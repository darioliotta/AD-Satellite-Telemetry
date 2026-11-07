"""Audited baseline BF/PA routines; bounded original SPOT initialization."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from ..bootstrap import activate_baseline
activate_baseline()
from paper_metrics import paper_best_f1, dense_best_f1
from TranADPlus.src import omnianomaly
from TranADPlus.src.spot import SPOT
from opssat_pipeline import original_metrics, adjust_per_fragment, point_metrics


def continuous_metrics(scores, labels):
    if len(np.unique(labels)) != 2:
        return {"roc_auc_continuous_scores": None, "average_precision": None}
    return {"roc_auc_continuous_scores": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}


def fixed_metrics(scores, labels, threshold, prefix, groups=None):
    if groups is not None:
        return {**point_metrics(labels, adjust_per_fragment(scores, labels, groups, threshold), prefix),
                f"{prefix}_threshold": float(threshold)}
    if len(np.unique(labels)) != 2:
        raise ValueError("Native paper metrics require both test classes; do not fabricate labels")
    stats, latency = omnianomaly.calc_seq(scores, labels.copy(), threshold)
    keys = ("f1", "precision", "recall", "TP", "TN", "FP", "FN", "ROC/AUC")
    result = {f"{prefix}_{k}": int(v) if k in ("TP", "TN", "FP", "FN") else float(v)
              for k, v in zip(keys, stats)}
    return {**result, f"{prefix}_threshold": float(threshold), f"{prefix}_latency": float(latency)}


def bounded_pot(train_scores, scores, labels, q, max_attempts=35):
    level, error = 0.98, None
    for attempt in range(1, max_attempts + 1):
        try:
            spot = SPOT(q)
            spot.fit(train_scores, scores)
            spot.initialize(level=level, min_extrema=False, verbose=False)
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            level *= 0.999
    else:
        return {"status": "not_available", "pot_f1": None, "reason": error,
                "initialization_attempts": max_attempts}
    # As upstream, run only after a successful initialization. Do not silently
    # change the threshold algorithm if fitting/running fails.
    try:
        threshold = float(np.mean(spot.run(dynamic=False)["thresholds"]))
        if not np.isfinite(threshold):
            raise FloatingPointError("Nonfinite POT threshold")
        return {**fixed_metrics(scores, labels, threshold, "pot"), "status": "measured",
                "initialization_attempts": attempt, "initialization_level": float(level)}
    except Exception as exc:
        return {"status": "not_available", "pot_f1": None, "reason": f"{type(exc).__name__}: {exc}"}


def evaluate_oracle(train_grid, test_grid, labels, alphas, q, settings, groups=None):
    winners = {"pot": None, "bf": None, "bf_dense": None}
    records = []
    for index, alpha in enumerate(alphas):
        tr, te = train_grid[index], test_grid[index]
        if not np.isfinite(tr).all() or not np.isfinite(te).all():
            raise FloatingPointError("Nonfinite scores must not enter the metric functions")
        if groups is None:
            bf = paper_best_f1(te, labels)
            dense = dense_best_f1(te, labels)
            pot = bounded_pot(tr, te, labels, q, settings["pot_max_initializations"])
        else:
            # Exactly the OPS-SAT v10/v12 fragment-aware implementation.
            both, pot = original_metrics(tr, te, labels, groups, q)
            bf = {k: v for k, v in both.items() if not k.startswith("bf_dense_")}
            dense = {k: v for k, v in both.items() if k.startswith("bf_dense_")}
        dense.update(fixed_metrics(te, labels, dense["bf_dense_threshold"], "bf_dense", groups))
        for key, result, metric in (("pot", pot, "pot_f1"), ("bf", bf, "bf_f1"),
                                    ("bf_dense", dense, "bf_dense_f1")):
            result.update({"alpha": alpha, "alpha_applicable": alpha is not None,
                           "beta": None if alpha is None else 1.0 - alpha,
                           "point_adjustment": "fragment_aware" if groups is not None else "upstream_global",
                           "selection_uses_test_labels": True})
            current = winners[key]
            if current is None or (result.get(metric) is not None and
                                   (current.get(metric) is None or result[metric] > current[metric])):
                winners[key] = result
        records.append({"alpha": alpha, "pot_f1": pot.get("pot_f1"), "bf_f1": bf["bf_f1"],
                        "bf_dense_f1": dense["bf_dense_f1"], "pot_status": pot.get("status"),
                        **continuous_metrics(te, labels)})
        print(f"[EVAL] alpha={alpha} POT={pot.get('pot_f1')} BF200={bf['bf_f1']:.6f} "
              f"BFdense={dense['bf_dense_f1']:.6f}", flush=True)
    return {**winners, "alpha_grid": records,
            "pot_available_alphas": sum(r["pot_f1"] is not None for r in records),
            "auc_definition": "ROC-AUC on binary predictions after point adjustment, as in the authors' code",
            "bf_definition": "200 linear thresholds from min(score) to mean(score)+3*std(score)",
            "dense_definition": "all distinct score thresholds plus a threshold below the minimum",
            "warning": "Alpha and threshold selected using test labels: oracle diagnostics, not deployment performance"}
