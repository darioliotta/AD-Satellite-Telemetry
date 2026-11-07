"""One compact index of verified V14 context and continuation results."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics as st

from tranad_variants.common import hash_file, write_json


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    root = Path(root)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    context, continuation, inputs = [], [], []
    for stage, output in (("context", context), ("continuation", continuation)):
        for path in sorted((root / stage).glob("*/*/*/seed_*/metrics.json")):
            status = json.loads((path.parent / "diagnostic_status.json").read_text())
            if status["status"] != "complete":
                continue
            if hash_file(path) != status["artifacts"]["metrics.json"]:
                raise RuntimeError(f"Changed V14 metrics: {path}")
            report = json.loads(path.read_text())
            p = report["profile"]
            meta = {k: p[k] for k in ("dataset", "variant", "configuration", "seed")}
            inputs.append({"path": str(path.relative_to(root)), "sha256": hash_file(path)})
            if stage == "context":
                baseline = report["conditions"][0]
                for condition in report["conditions"]:
                    row = {**meta, "condition": condition["condition"],
                           "point_auc": condition["pointwise"]["auc"],
                           "real_point_auc": condition["real_points"]["auc"],
                           "real_point_auc_delta_pp": 100*(condition["real_points"]["auc"]-baseline["real_points"]["auc"]),
                           "raw_best_f1_fixed_alpha": condition["raw_best_f1_test_oracle"]["f1"],
                           "raw_train_threshold_f1": condition["raw_train_threshold"]["f1"],
                           "mean_absolute_score_change": condition["mean_absolute_score_change"]}
                    if "fragment" in condition:
                        row.update(fragment_mean_auc=condition["fragment"]["mean"]["auc"],
                                   fragment_max_auc=condition["fragment"]["max"]["auc"])
                    output.append(row)
            else:
                comp_path = path.parent / "comparison.json"
                if hash_file(comp_path) != status["artifacts"]["comparison.json"]:
                    raise RuntimeError(f"Changed V14 comparison: {comp_path}")
                comp = json.loads(comp_path.read_text())
                row = {**meta, "condition": "epoch50_to_100"}
                for label, values in comp["steps"].items():
                    row[label+"_raw_auc"] = values["raw_fixed_alpha"]["auc"]
                    row[label+"_raw_ap"] = values["raw_fixed_alpha"]["average_precision"]
                    for method in ("pot", "bf", "bf_dense"):
                        row[label+"_"+method+"_pa_f1"] = values["paper_oracle"][method][method+"_f1"]
                    selected = values["validation_selected"]
                    if selected["status"] == "measured":
                        row[label+"_validation_selected_fragment_auc"] = selected["fragment"]["roc_auc"]
                for key in list(row):
                    if key.startswith("before_") and "after_"+key[7:] in row:
                        row[key[7:]+"_delta_pp"] = 100*(row["after_"+key[7:]]-row[key])
                output.append(row)
    write_csv(out / "context_runs.csv", context)
    write_csv(out / "continuation_runs.csv", continuation)
    groups = defaultdict(list)
    for stage, rows in (("context", context), ("continuation", continuation)):
        for row in rows:
            groups[stage, row["dataset"], row["variant"], row["configuration"], row["condition"]].append(row)
    summaries = []
    for (stage, ds, variant, config, condition), rows in sorted(groups.items()):
        summary = dict(study=stage, dataset=ds, variant=variant, configuration=config, condition=condition, n=len(rows))
        for key in dict.fromkeys(k for r in rows for k in r):
            if key == "seed" or not all(isinstance(r.get(key), (float, int)) for r in rows):
                continue
            values = [r[key] for r in rows]
            summary[key+"_mean"] = st.mean(values)
            summary[key+"_sd"] = st.stdev(values) if len(values)>1 else None
        summaries.append(summary)
    write_csv(out / "mean_std.csv", summaries)
    write_json(out / "index.json", {"context_conditions": len(context), "continuations": len(continuation),
                                    "groups": len(summaries), "inputs": inputs})
    (out / "README.md").write_text(
        "# V14 summary\n\n"
        "`context_runs.csv`: one row per seed/condition. `continuation_runs.csv`: paired epoch 50/100 endpoints. "
        "`mean_std.csv`: mean and sample SD over seeds, with n explicitly reported. "
        "Metrics are fractions, deltas are percentage points. No maximum over seeds is selected.\n\n"
        "The first-ablation window figures/tables are in `../window_study/`.\n")
    return summaries
