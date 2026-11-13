from pathlib import Path
import json
import numpy as np

from ..common import hash_file, now, save_npz, write_json
from ..models import build_model
from ..training.engine import load_checkpoint
from .scoring import collect_outputs, reconstruction_scores
from .metrics import evaluate_oracle
from .selection import validation_selected
from .resources import measure


def verified_metrics(folder, signature):
    folder = Path(folder)
    path = folder / "metrics.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report.get("signature") != signature or report.get("status") != "complete":
        raise RuntimeError(f"Existing metrics have a different/incomplete signature: {path}")
    for relative, expected in report["artifacts"].items():
        artifact = folder / relative
        if not artifact.is_file() or hash_file(artifact) != expected:
            raise RuntimeError(f"Result artifact is missing or has changed: {artifact}")
    return report


def evaluate(profile, bundle, folder, signature, device, settings):
    folder = Path(folder)
    existing = verified_metrics(folder, signature)
    if existing is not None:
        print(f"EVALUATE SKIP verified metrics: {profile['identifier']}", flush=True)
        return existing
    final = folder / "checkpoints/final.ckpt"
    saved = load_checkpoint(final, profile, signature, final=True)
    model = build_model(profile).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    score_batch = settings["score_batch_size_by_dataset"][profile["dataset"]]
    outputs = {key: collect_outputs(model, dataset, device, score_batch,
                                   f"EVAL {profile['dataset']} {key}", folder / "numerical_diagnostics.json")
               for key, dataset in bundle.datasets.items()}
    alphas = [float(a) for a in np.linspace(0, 1, settings["alpha_steps"])] if model.output_count == 2 else [None]
    grids = {key: np.stack([reconstruction_scores(value, a) for a in alphas])
             for key, value in outputs.items()}
    arrays = {f"{key}_scores": grid for key, grid in grids.items()}
    arrays.update({f"{key}_labels": values for key, values in bundle.labels.items()})
    # Single-output models have one score row and an empty alpha list, not a
    # fabricated second decoder or a meaningless repeated alpha sweep.
    arrays["alpha_values"] = np.array([a for a in alphas if a is not None], dtype=np.float64)
    arrays["alpha_applicable"] = np.array([model.output_count == 2])
    for key, groups in bundle.groups.items():
        if groups:
            arrays[f"{key}_segment_ids"] = np.concatenate([np.repeat(g["id"], len(g["values"])) for g in groups])
            arrays[f"{key}_channels"] = np.concatenate([np.repeat(g["channel"], len(g["values"])) for g in groups])
    save_npz(folder / "scores.npz", **arrays)
    selection = validation_selected(outputs, bundle, settings["train_quantile"])
    write_json(folder / "validation_selection.json", selection)
    oracle = evaluate_oracle(grids["train"], grids["test"], bundle.labels["test"], alphas,
                             profile["pot_q"], settings, groups=bundle.groups.get("test"))
    sample = bundle.datasets["test"][0]
    sample = sample[0] if isinstance(sample, (tuple, list)) else sample
    resources = measure(model, profile, sample, device, settings)
    history = {}
    for line in (folder / "epochs.jsonl").read_text().splitlines():
        record = json.loads(line)
        history[record["epoch_one_based"]] = record
    artifacts = {name: hash_file(folder / name) for name in
                 ("checkpoints/final.ckpt", "scores.npz", "validation_selection.json", "epochs.jsonl", "config.json", "manifest.json")}
    result = {"schema_version": 1, "status": "complete", "created_at": now(),
              "identifier": profile["identifier"], "profile": profile, "signature": signature,
              "paper_oracle": oracle, "validation_selected": selection, "resources": resources,
              "evaluation_settings": settings,
              "scoring": {key: value["performance"] for key, value in outputs.items()},
              "training": {"epochs_logged": len(history),
                           "sum_epoch_seconds": sum(r["elapsed_seconds"] for r in history.values()),
                           "final_loss": saved["final_loss"],
                           "loss": "epoch-weighted two-reconstruction MSE" if model.output_count == 2 else "single-reconstruction MSE"},
              "dataset_inventory": bundle.inventory, "artifacts": artifacts,
              "score_layout": "rows follow alpha_values for two outputs; one row and no alpha for single outputs",
              "label_granularity": "weak_fragment_labels" if bundle.groups else "point_labels"}
    if len(history) != profile["epochs"]:
        raise RuntimeError("Incomplete epoch telemetry; do not label this run complete")
    write_json(folder / "metrics.json", result)
    print(f"EVALUATE COMPLETE {folder / 'metrics.json'}", flush=True)
    return result
