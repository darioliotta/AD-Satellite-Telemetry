"""Frozen-weight history interventions with fixed scoring and calibration."""
import numpy as np
import torch

from tranad_plus.core.common import save_npz, write_json
from tranad_plus.core.evaluation.scoring import collect_outputs, reconstruction_scores

from .metrics import real_point_mask, ranking, raw_best_f1, fixed_threshold, fragment_ranking
from .progress import bar, record as progress_record

CONDITIONS = ("original", "reverse_past", "repeat_current")




def perturb(windows, condition):
    if condition == "original":
        return windows
    if condition == "reverse_past":
        result = windows.clone()
        result[:-1] = windows[:-1].flip(0)
        return result
    if condition == "repeat_current":
        return windows[-1:].expand_as(windows).clone()
    raise ValueError(f"Unknown intervention: {condition}")


class Intervention(torch.nn.Module):
    def __init__(self, model, condition):
        super().__init__()
        self.model, self.condition = model, condition
        self.output_count = model.output_count

    def forward(self, windows, target):
        changed = perturb(windows, self.condition)
        # Target and its position remain unchanged for every intervention.
        return self.model(changed, target)










def run_context(model, bundle, parent_scores, folder, profile, device, settings, data_root):
    alpha = .5 if model.output_count == 2 else None
    with np.load(parent_scores, allow_pickle=False) as saved:
        index = int(np.flatnonzero(saved["alpha_values"] == .5)[0]) if alpha is not None else 0
        original_train = saved["train_scores"][index].copy()
        original_test = saved["test_scores"][index].copy()
        np.testing.assert_array_equal(bundle.labels["test"], saved["test_labels"])
    # The original train threshold is held fixed across history interventions.
    threshold = float(np.quantile(original_train, settings["train_quantile"]))
    labels = bundle.labels["test"]
    real = real_point_mask(profile, data_root, len(labels))
    records, grids = [], []
    baseline_scores = None
    for step, condition in enumerate(bar(CONDITIONS, total=len(CONDITIONS), desc="V14 context | conditions",
                                         position=1, unit="condition", leave=False), 1):
        progress_record(folder, "context_evaluation", condition=condition, condition_index=step, conditions_total=len(CONDITIONS))
        outputs = collect_outputs(Intervention(model, condition), bundle.datasets["test"], device,
                                  settings["score_batch_size_by_dataset"][profile["dataset"]],
                                  f"CONTEXT {profile['dataset']} {condition}",
                                  folder / "numerical_diagnostics.json")
        scores = reconstruction_scores(outputs, alpha)
        if condition == "original":
            # A different numerical environment must not silently masquerade as
            # the intervention effect. CPU/GPU parity has a declared tolerance.
            np.testing.assert_allclose(scores, original_test, rtol=5e-4, atol=1e-7)
            baseline_scores = scores
        reconstruction = outputs["outputs"][0] if alpha is None else (
            alpha * outputs["outputs"][0] + (1-alpha) * outputs["outputs"][1])
        errors = (reconstruction - outputs["target"]) ** 2
        feature_metrics = [{"feature_index": i,
                            "normal_mean_mse": float(errors[labels == 0, i].mean()),
                            "anomaly_mean_mse": float(errors[labels == 1, i].mean()),
                            **ranking(labels, errors[:, i])} for i in range(errors.shape[1])]
        record = {"condition": condition, "pointwise": ranking(labels, scores),
                  "real_points": {"n": int(real.sum()), **ranking(labels[real], scores[real]),
                                  "raw_train_threshold": fixed_threshold(labels[real], scores[real], threshold)},
                  "raw_best_f1_test_oracle": raw_best_f1(labels, scores),
                  "raw_train_threshold": fixed_threshold(labels, scores, threshold),
                  "mean_absolute_score_change": float(np.abs(scores-baseline_scores).mean()),
                  "per_feature_association_with_global_labels": feature_metrics,
                  "performance": outputs["performance"]}
        if bundle.groups:
            record["fragment"] = fragment_ranking(scores, bundle.groups["test"])
        records.append(record)
        grids.append(scores)
    arrays = {"test_scores": np.stack(grids), "test_labels": labels,
              "conditions": np.array(CONDITIONS), "real_point_mask": real,
              "alpha": np.array([np.nan if alpha is None else alpha])}
    if bundle.groups:
        arrays["test_segment_ids"] = np.concatenate([
            np.repeat(g["id"], len(g["values"])) for g in bundle.groups["test"]])
    save_npz(folder / "scores.npz", **arrays)
    report = {"status": "complete", "profile": profile, "alpha": alpha,
              "point_adjustment": False, "conditions": records,
              "baseline_max_abs_score_difference": float(np.abs(baseline_scores-original_test).max()),
              "baseline_parity_tolerance": {"rtol": 5e-4, "atol": 1e-7},
              "notes": ["Frozen weights, unchanged target, fixed alpha; train threshold not recalibrated.",
                        "Perturbations can be out of distribution: sensitivity is not a proof of causal mechanism.",
                        "OPS-SAT point labels are weak; mean/max fragment aggregation is fixed, not selected on test.",
                        "Native real_points excludes padded target duplicates, retaining the same actual test points across W.",
                        "Feature AUC uses global labels; it is an association, not feature anomaly localization."]}
    write_json(folder / "metrics.json", report)
    return report
