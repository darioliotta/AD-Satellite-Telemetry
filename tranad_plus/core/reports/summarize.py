import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tranad_plus.core.bootstrap import PROJECT_ROOT
from tranad_plus.core.common import atomic_destination, digest, write_json


def summarize(root):
    root = Path(root)
    rows = []
    for path in sorted(root.glob("*/*/*/seed_*/metrics.json")):
        report = json.loads(path.read_text())
        if report.get("status") != "complete":
            continue
        p, oracle, resources = report["profile"], report["paper_oracle"], report["resources"]
        manifest = json.loads((path.parent / "manifest.json").read_text())
        row = {k: p[k] for k in ("identifier", "dataset", "variant", "configuration", "seed", "window_sz", "dim_feedforward")}
        row.update({"execution_sha256": digest(manifest["execution"]),
                    "parameters": resources["trainable_parameters"],
                    "weight_bytes": resources["parameter_bytes_fp32"],
                    "latency_median_ms": resources["batch_one_forward"]["median_ms"],
                    "analytical_macs": resources["analytical_matmul_macs_per_window"],
                    "training_seconds": report["training"]["sum_epoch_seconds"],
                    "metrics_file": str(path.relative_to(root))})
        for group, prefix in (("pot", "pot"), ("bf", "bf"), ("bf_dense", "bf_dense")):
            for metric in ("f1", "precision", "recall", "ROC/AUC"):
                row[f"{prefix}_{metric}"] = oracle[group].get(f"{prefix}_{metric}")
        selection = report["validation_selected"]
        row["validation_fragment_auc"] = selection.get("selection", {}).get("chosen", {}).get("validation_fragment_auc")
        row["validation_selected_test_fragment_auc"] = selection.get("fragment", {}).get("roc_auc")
        rows.append(row)
    controls = {(r["dataset"], r["configuration"], r["seed"], r["execution_sha256"]): r
                for r in rows if r["variant"] == "reference"}
    for row in rows:
        baseline = controls.get((row["dataset"], row["configuration"], row["seed"], row["execution_sha256"]))
        row["matched_reference_available"] = baseline is not None
        for metric in ("pot_f1", "bf_f1", "bf_dense_f1", "parameters", "latency_median_ms"):
            a, b = row.get(metric), baseline.get(metric) if baseline else None
            row[f"delta_{metric}_vs_reference"] = a - b if a is not None and b is not None else None
    if rows:
        with atomic_destination(root / "all_experiments.csv") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["variant"], row["configuration"], row["execution_sha256"])
        grouped.setdefault(key, []).append(row)
    aggregates = []
    metrics = ("pot_f1", "bf_f1", "bf_dense_f1", "parameters", "latency_median_ms",
               "validation_fragment_auc", "validation_selected_test_fragment_auc",
               "delta_pot_f1_vs_reference", "delta_bf_f1_vs_reference", "delta_bf_dense_f1_vs_reference")
    for key, values in sorted(grouped.items()):
        entry = dict(zip(("dataset", "variant", "configuration", "execution_sha256"), key))
        entry.update({"n_seeds": len(values), "seeds": sorted(v["seed"] for v in values)})
        for metric in metrics:
            numbers = [v[metric] for v in values if v.get(metric) is not None]
            entry[metric] = {"n": len(numbers), "mean": statistics.mean(numbers) if numbers else None,
                             "sample_std": statistics.stdev(numbers) if len(numbers) > 1 else None}
        aggregates.append(entry)
    write_json(root / "aggregate.json", {"completed_runs": len(rows), "groups": aggregates,
               "note": "Groups separated by execution identity; n=1 does not estimate seed variability. Oracle and validation-selected metrics remain distinct."})
    return rows, aggregates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results/v13/architectures")
    args = parser.parse_args()
    rows, groups = summarize(args.results_root)
    print(f"Completed runs: {len(rows)}; aggregate groups: {len(groups)}; {args.results_root}")
