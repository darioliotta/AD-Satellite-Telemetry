"""Preserve OPS-SAT validation selection; single-output models have no alpha grid."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from ..bootstrap import activate_baseline
from .scoring import reconstruction_scores
activate_baseline()
from opssat_scoring import (selected_scoring, evaluate_selected, fit_calibrator, transform_scores,
                           point_channels, segment_statistics, auc_ap, CALIBRATIONS, AGGREGATIONS)


def validation_selected(outputs, bundle, quantile):
    if "validation" not in outputs:
        return {"status": "not_applicable", "reason": "Native SMAP/MSL official train/test protocol has no validation split"}
    tr, va, te = (outputs[k] for k in ("train", "validation", "test"))
    groups = bundle.groups
    if len(tr["outputs"]) == 2:
        def old_arrays(value):
            return tuple(a.reshape(-1) for a in value["outputs"] + [value["target"]])
        selection = selected_scoring(old_arrays(tr), old_arrays(va), groups["train"], groups["validation"])
        result = evaluate_selected(old_arrays(tr), old_arrays(te), groups["train"], groups["test"],
                                   selection, train_quantile=quantile)
        return {"status": "measured", "alpha_applicable": True, **result}
    train_scores = reconstruction_scores(tr, float64=True)
    val_scores = reconstruction_scores(va, float64=True)
    test_scores = reconstruction_scores(te, float64=True)
    train_channels, val_channels, test_channels = (point_channels(groups[k]) for k in ("train", "validation", "test"))
    candidates = []
    for calibration in CALIBRATIONS:
        fitted = fit_calibrator(train_scores, train_channels, calibration)
        calibrated = transform_scores(val_scores, val_channels, fitted)
        for aggregation in AGGREGATIONS:
            values, labels, _, _ = segment_statistics(calibrated, groups["validation"], aggregation)
            quality = auc_ap(labels, values)
            candidates.append({"calibration": calibration, "aggregation": aggregation,
                               "validation_fragment_auc": quality["roc_auc"],
                               "validation_fragment_ap": quality["average_precision"]})
    valid = [c for c in candidates if c["validation_fragment_auc"] is not None]
    if not valid:
        raise ValueError("No valid validation candidate")
    chosen = max(valid, key=lambda c: c["validation_fragment_auc"])
    fitted = fit_calibrator(train_scores, train_channels, chosen["calibration"])
    trc = transform_scores(train_scores, train_channels, fitted)
    tec = transform_scores(test_scores, test_channels, fitted)
    thresholds = {ch: float(np.quantile(trc[train_channels == ch], quantile)) for ch in fitted["channels"]}
    predictions = np.array([s > thresholds[ch] for s, ch in zip(tec, test_channels)], dtype=np.int8)
    labels = bundle.labels["test"]
    p, r, f, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    fragments, fragment_labels, _, fragment_channels = segment_statistics(tec, groups["test"], chosen["aggregation"])
    by_channel = {}
    for channel in sorted(fitted["channels"]):
        pmask, fmask = test_channels == channel, fragment_channels == channel
        by_channel[channel] = {"points": int(pmask.sum()), "fragments": int(fmask.sum()),
                               "pointwise": auc_ap(labels[pmask], tec[pmask]),
                               "fragment": auc_ap(fragment_labels[fmask], fragments[fmask])}
    return {"status": "measured", "alpha_applicable": False,
            "selection": {"chosen": chosen, "candidates": candidates,
                          "selection_target": "fragment_roc_auc_validation_only"},
            "calibrator": fitted, "thresholds_train_only_by_channel": thresholds,
            "pointwise": {**auc_ap(labels, tec), "precision": float(p), "recall": float(r),
                          "f1": float(f), "point_adjusted": False},
            "fragment": {**auc_ap(fragment_labels, fragments), "aggregation": chosen["aggregation"]},
            "by_channel": by_channel,
            "protocol": "Validation-only selection; train-only calibration and thresholds; test labels used for reporting only",
            "weak_labels": "Fragment labels repeated on points; fragment-level AUC is primary"}
