#!/usr/bin/env python3
"""OPS-SAT context diagnostics from existing V13/V14 artifacts; no training.

Published-style results retain each run's saved global BF200/POT alpha and
threshold. Cohorts never reselect a threshold. Morphology is descriptive,
with cutoffs fitted on nominal optimization fragments only, not fault labels.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics as st
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "opssat-context-mpl"))
import numpy as np
from tqdm import tqdm

METHODS = ("bf", "pot")
METRICS = ("f1_pa", "precision_pa", "recall_pa", "auc_pa", "fpr",
           "anomalous_fragment_recall", "nominal_fragment_alarm_rate")
MAGNETIC = {"CADC0872", "CADC0873", "CADC0874"}
COLORS = {"reference": "#333333", "encoder_only": "#9B0014"}
NAMES = {"reference": "TranAD+", "encoder_only": "Encoder-only"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def blocks(ids):
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(ids)]
    if len(set(ids[starts].tolist())) != len(starts):
        raise ValueError("Fragment appears in disjoint score blocks")
    return list(zip(starts.tolist(), ends.tolist()))


def fragment_counts(scores, labels, ids, threshold):
    """PA within each weakly labelled fragment; preserve nominal false alarms."""
    pred = scores > threshold
    result = []
    for a, b in blocks(ids):
        y = int(labels[a])
        if not np.all(labels[a:b] == y):
            raise ValueError("Inconsistent weak labels inside fragment")
        hit = bool(pred[a:b].any())
        tp = b-a if y and hit else 0
        fp = int(pred[a:b].sum()) if not y else 0
        result.append({"segment": int(ids[a]), "label": y, "points": b-a,
                       "tp": tp, "fp": fp, "tn": b-a-fp if not y else 0,
                       "fn": b-a-tp if y else 0, "hit": int(hit),
                       "direct_hits": int(pred[a:b].sum()),
                       "score_mean": float(scores[a:b].mean()),
                       "score_max": float(scores[a:b].max())})
    return result


def metrics(rows):
    counts = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "tn", "fn")}
    tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    recall = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    return {**counts, "fragments": len(rows), "anomalous_fragments": len(pos),
            "nominal_fragments": len(neg), "points": sum(r["points"] for r in rows),
            "f1_pa": 2*tp/(2*tp+fp+fn) if pos else None,
            "precision_pa": tp/(tp+fp) if tp+fp else 0., "recall_pa": recall,
            "auc_pa": (recall+specificity)/2 if recall is not None and specificity is not None else None,
            "fpr": 1-specificity if specificity is not None else None,
            "anomalous_fragment_recall": st.mean(r["hit"] for r in pos) if pos else None,
            "nominal_fragment_alarm_rate": st.mean(r["hit"] for r in neg) if neg else None}


def load_fragments(path, train_ids):
    raw = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            raw[int(row["segment"])].append(row)
    channel_training = defaultdict(list)
    for segment in train_ids:
        rows = raw[segment]
        if any(int(r["anomaly"]) != 0 or int(r["train"]) != 1 for r in rows):
            raise ValueError("Morphology calibration must use nominal optimization data")
        channel_training[rows[0]["channel"]].extend(float(r["value"]) for r in rows)
    scales = {c: max(float(np.quantile(v, .95)-np.quantile(v, .05)), 1e-12)
              for c, v in channel_training.items()}
    out = {}
    for segment, rows in raw.items():
        channel = rows[0]["channel"]
        for k in ("channel", "anomaly", "train"):
            if len({r[k] for r in rows}) != 1:
                raise ValueError(f"Nonconstant {k} in fragment {segment}")
        dates = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in rows]
        t = np.array([(d-dates[0]).total_seconds() for d in dates])
        if len(t) < 2 or np.any(np.diff(t) < 0):
            raise ValueError(f"Decreasing/insufficient timestamps: {segment}")
        x = np.array([float(r["value"]) for r in rows])
        sampling = {float(r["sampling"]) for r in rows}
        cadence = float(next(iter(sampling))) if len(sampling) == 1 else float(np.median(np.diff(t)))
        # A sample-index descriptor, not a time derivative: timestamps can
        # repeat for distinct values and must not be deduplicated or invented.
        expected = (x[:-2]+x[2:])/2
        q = max(1, len(x)//4)
        spans = {f"history_span_w{w}_median_s": float(np.median(t[w-1:]-t[:len(t)-w+1]))
                 if len(t) >= w else None for w in (5, 10)}
        out[segment] = {"segment": segment, "channel": channel,
                        "sensor": "magnetometer" if channel in MAGNETIC else "sun_angle",
                        "label": int(rows[0]["anomaly"]), "official_train": int(rows[0]["train"]),
                        "optimization_fragment": int(segment in train_ids), "n_samples": len(x),
                        "duration_s": float(t[-1]), "sampling": f"{cadence:g}s" if len(sampling) == 1 else "mixed",
                        "median_dt_s": float(np.median(np.diff(t))), "max_dt_s": float(np.diff(t).max()),
                        "large_gap": int(np.any(np.diff(t) > 3*cadence)),
                        "repeated_timestamp_fraction": float(np.mean(np.diff(t) == 0)),
                        "irregular_interval_fraction": float(np.mean(np.abs(np.diff(t)-cadence) > .25*cadence)),
                        "local_departure": float(np.abs(x[1:-1]-expected).max()/scales[channel]) if len(x)>2 else 0.,
                        "endpoint_drift": float(abs(np.median(x[-q:])-np.median(x[:q]))/scales[channel]),
                        **spans}
    cutoffs = {}
    for channel in sorted(scales):
        fit = [out[i] for i in train_ids if out[i]["channel"] == channel]
        cutoffs[channel] = {"n_nominal_optimization_fragments": len(fit), "channel_scale_p95_minus_p05": scales[channel],
                            "duration_tertiles_s": np.quantile([r["duration_s"] for r in fit], [1/3, 2/3]).tolist(),
                            **{k+"_q75": float(np.quantile([r[k] for r in fit], .75))
                               for k in ("local_departure", "endpoint_drift")}}
    for row in out.values():
        c = cutoffs[row["channel"]]
        row["duration_bin"] = ("short", "middle", "long")[int(np.searchsorted(c["duration_tertiles_s"], row["duration_s"], side="right"))]
        for name in ("local_departure", "endpoint_drift"):
            row[name+"_bin"] = "high" if row[name] > c[name+"_q75"] else "ordinary"
    return out, cutoffs


def cohorts(rows, fragments):
    result = defaultdict(list)
    for row in rows:
        f = fragments[row["segment"]]
        for kind, value in (("all", "all"), ("channel", f["channel"]), ("sensor", f["sensor"]),
                            ("sampling", f["sampling"]), ("duration", f["duration_bin"]),
                            ("local_departure", f["local_departure_bin"]),
                            ("endpoint_drift", f["endpoint_drift_bin"]), ("large_gap", str(f["large_gap"])),
                            ("repeated_timestamps", str(int(f["repeated_timestamp_fraction"] > 0)))):
            result[kind, value].append(row)
    return result


def summaries(rows, context=False):
    keys = ["variant", "configuration", "epochs", "method", "cohort", "value"]
    if context:
        keys.append("condition")
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, part in sorted(grouped.items()):
        row = dict(zip(keys, key), n_seeds=len(part))
        for k in ("fragments", "anomalous_fragments", "nominal_fragments", "points"):
            row[k] = part[0][k]
            if len({r[k] for r in part}) != 1:
                raise ValueError("Subgroup populations differ across seeds")
        for k in METRICS:
            values = [r[k] for r in part if r[k] is not None]
            row[k+"_mean"] = st.mean(values) if values else None
            row[k+"_sd"] = st.stdev(values) if len(values)>1 else None
        out.append(row)
    return out


def select_scores(arrays, method):
    if method.get("alpha_applicable"):
        found = np.flatnonzero(np.isclose(arrays["alpha_values"], method["alpha"], atol=1e-12, rtol=0))
        if len(found) != 1:
            raise ValueError("Cannot identify saved winning alpha")
        return arrays["test_scores"][int(found[0])]
    if arrays["test_scores"].shape[0] != 1:
        raise ValueError("Unexpected single-output score layout")
    return arrays["test_scores"][0]


def paired_windows(rows):
    index = {(r["variant"], r["configuration"], r["epochs"], r["seed"], r["method"], r["cohort"], r["value"]): r for r in rows}
    out = []
    for short in rows:
        long_config = {"window_only": "reference", "compact": "ff_only"}.get(short["configuration"])
        if long_config is None:
            continue
        long = index.get((short["variant"], long_config, short["epochs"], short["seed"], short["method"], short["cohort"], short["value"]))
        if long is None:
            continue
        if short["ff"] != long["ff"]:
            raise ValueError("Window contrast changed FF")
        out.append({**{k: short[k] for k in ("variant", "epochs", "seed", "method", "cohort", "value", "ff")},
                    "short_configuration": short["configuration"], "long_configuration": long_config,
                    "short_window": short["window"], "long_window": long["window"],
                    "anomalous_fragments": short["anomalous_fragments"], "nominal_fragments": short["nominal_fragments"],
                    **{k+"_delta_pp": 100*(short[k]-long[k]) if short[k] is not None and long[k] is not None else None for k in METRICS}})
    return out


def report_text(out, summary, context, inventory, experiment_count):
    lines = ["# OPS-SAT: temporal context and signal families", "",
             f"Read-only analysis of {experiment_count} trained endpoints, on the same 529 test fragments.", "",
             "Only original Best F1 (200 thresholds) and POT are reported. F1, precision, recall and AUC-PA use fragment-aware point adjustment. AUC-PA is calculated from binary predictions. Scores, trained models and source campaigns are unchanged.", "",
             "Each endpoint retains its saved full-test global alpha and threshold. They are not optimized again for individual channels or cohorts. These are the original test-oracle conventions, not deployment estimates. Values below are percentages and means over the five seeds; sample SDs are in the CSV files.", "",
             "## Results by sensor family", "",
             "Both configurations have FF=8. Each subgroup uses the global threshold of its parent run.", "",
             "| Model | Epochs | Method | Sensor | Anomalous / nominal fragments | F1 W10 | F1 W5 |",
             "|---|---:|---|---|---:|---:|---:|"]
    lookup = {(r['variant'], r['configuration'], r['epochs'], r['method'], r['cohort'], r['value']): r for r in summary}
    for variant in ('reference', 'encoder_only'):
        for epochs in (50, 100):
            for method in METHODS:
                for sensor in ('magnetometer', 'sun_angle'):
                    pair = [lookup[variant, c, epochs, method, 'sensor', sensor] for c in ('reference', 'window_only')]
                    a, b = pair
                    lines.append(f"| {NAMES[variant]} | {epochs} | {'BF200' if method == 'bf' else 'POT'} | {sensor} | {a['anomalous_fragments']} / {a['nominal_fragments']} | {100*a['f1_pa_mean']:.2f} | {100*b['f1_pa_mean']:.2f} |")
    lines += ["", "## Frozen-weight context interventions", "",
              "Encoder-only checkpoints at epoch 50. The saved original BF200/POT thresholds stay fixed across conditions. `reverse_past` reverses the past while preserving the final sample; `repeat_current` replaces every past input with the current sample. No model is retrained and no threshold is reselected. Repeated-current input can be out of distribution, so this tests sensitivity, not a causal physical mechanism.", "",
              "| Window | Method | Condition | F1-PA | AUC-PA |", "|---:|---|---|---:|---:|"]
    for r in context:
        if r['cohort'] == 'all':
            w = 10 if r['configuration'] == 'reference' else 5
            lines.append(f"| {w} | {'BF200' if r['method'] == 'bf' else 'POT'} | {r['condition']} | {100*r['f1_pa_mean']:.2f} | {100*r['auc_pa_mean']:.2f} |")
    test = [r for r in inventory.values() if not r['official_train']]
    repeats = [r for r in test if r['repeated_timestamp_fraction'] > 0]
    lines += ["", "## Physical-time and morphology controls", "",
              f"The source contains {len(repeats)} test fragments with repeated timestamps for distinct sample rows. Original row order is retained. No row is removed or assigned an invented timestamp. Duration and actual W5/W10 spans are derived from timestamps, including repeats and gaps.", "",
              "Duration tertiles and the high local-departure/endpoint-drift cutoffs are fitted per channel using nominal optimization fragments only. Local departure is the maximum deviation from the average of adjacent samples, divided by the channel's nominal-training p95-p05 range. Endpoint drift compares medians of the first and last quarters using that same scale. These describe signal shape; they do not identify the location or physical cause of an anomaly. A large gap is an interval longer than three times the declared sampling step.", "",
              "## Limits and next decisions", "",
              "- Fragment labels do not identify which individual points are anomalous. PA credits a whole positive fragment after one crossing.",
              "- Channels contain different class counts and numbers of points. Single-class AUC and F1 without positive examples are left unavailable, not interpreted as detection performance. Sparse channels should not drive physical conclusions.",
              "- The five seeds share one test set; they are training replicates, not five independent telemetry datasets.",
              "- A global pointwise F1 objective weights anomalous fragments by their lengths. Inspect fragment recall and subgroup populations alongside F1.",
              "- Differences between W5 and W10 compare separately trained checkpoints. Frozen interventions probe sensitivity of the same weights, but can change the input distribution.",
              "- The initial V10/V12 six-window grid is not re-scored here: its fixed-alpha exports cannot reproduce all globally selected-alpha subgroup metrics. This analysis covers the V13/V14 W5/W10, FF4/FF8 settings.",
              "- These results can identify candidate mechanisms and useful follow-ups; they do not establish orbit, attitude or hardware-fault causes.", "",
              "## Files", "", "- `fragment_inventory.csv`: original units, timestamps, physical spans and descriptors.",
              "- `subgroup_mean_std.csv` / `subgroup_runs.csv`: all methods, channels and cohorts.",
              "- `paired_window_differences.csv`: short-minus-long percentage-point changes, matched by architecture, FF, epoch and seed.",
              "- `fragment_predictions.csv`: per-fragment counts and direct crossings; supports case review.",
              "- `frozen_context_mean_std.csv` / `frozen_context_runs.csv`: same-weight history interventions.",
              "- `figures/channel_comparisons.pdf`: BF200/POT channel plots at 50 and 100 epochs.",
              "- `morphology_cutoffs.json` and `manifest.json`: definitions, calibration populations and hashes.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def figures(out, summary, inventory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    path = out / "figures"
    path.mkdir(exist_ok=True)
    with PdfPages(path / "channel_comparisons.pdf") as pdf:
        for method in METHODS:
            for epoch in (50, 100):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
                for ax, variant in zip(axes, ("reference", "encoder_only")):
                    for config, marker, color in (("reference", "o", "#333333"), ("window_only", "s", "#9B0014")):
                        part = [r for r in summary if r["variant"] == variant and r["epochs"] == epoch and r["method"] == method and r["cohort"] == "channel" and r["configuration"] == config]
                        part.sort(key=lambda r: r["value"])
                        xs, ys, sds, ticks = [], [], [], []
                        for i, r in enumerate(part):
                            xs.append(i); ys.append(np.nan if r["f1_pa_mean"] is None else 100*r["f1_pa_mean"])
                            sds.append(0 if r["f1_pa_sd"] is None else 100*r["f1_pa_sd"])
                            ticks.append(r["value"]+f"\nA={r['anomalous_fragments']}, N={r['nominal_fragments']}")
                        ax.errorbar(xs, ys, yerr=sds, marker=marker, color=color, label=f"W={10 if config=='reference' else 5}, FF=8", capsize=3, lw=1)
                    ax.set_title(NAMES[variant]); ax.set_xticks(range(len(ticks)), ticks, rotation=60, ha="right", fontsize=8)
                    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=.2); ax.legend(fontsize=9)
                axes[0].set_ylabel("F1 after fragment-aware PA (%)")
                fig.suptitle(f"OPS-SAT | {'Best F1 (200 thresholds)' if method=='bf' else 'POT'} | epoch {epoch}\nSaved global alpha/threshold; mean ± SD over five seeds", color="#9B0014")
                fig.tight_layout(rect=(0, 0, 1, .91)); pdf.savefig(fig)
                fig.savefig(path / f"channels_{method}_e{epoch}.png", dpi=180)
                plt.close(fig)
    test = [r for r in inventory.values() if not r["official_train"]]
    channels = sorted({r["channel"] for r in test})
    fig, ax = plt.subplots(figsize=(10, 6))
    for w, offset, color in ((5, -.18, "#9B0014"), (10, .18, "#777777")):
        field = f"history_span_w{w}_median_s"
        values = [[r[field] for r in test if r["channel"] == c and r[field] is not None] for c in channels]
        box = ax.boxplot(values, positions=np.arange(len(channels))+offset, vert=False,
                         widths=.3, patch_artist=True, showfliers=False, manage_ticks=False)
        for patch in box["boxes"]:
            patch.set(facecolor=color, alpha=.5)
        for line in box["medians"]:
            line.set(color=color, linewidth=2)
        ax.plot([], [], color=color, linewidth=8, alpha=.5, label=f"W={w}")
    ax.set_yticks(np.arange(len(channels)), channels)
    ax.set_xlabel("Within-fragment median observed history span (seconds)")
    ax.set_title("OPS-SAT | Actual temporal context", color="#9B0014")
    ax.grid(axis="x", alpha=.2); ax.legend()
    fig.text(.12, .025, "Test fragments; first-to-last timestamps; padding excluded. Boxes summarize fragment medians; outliers hidden.", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(path / "physical_context.pdf"); fig.savefig(path / "physical_context.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    out = args.output_root.resolve() if args.output_root else project / "results/analysis_opssat_context"
    if any(out.is_relative_to((project / p).resolve()) for p in ("results/v13", "results/v14", "data")):
        raise ValueError("Output must be separate from original campaigns")
    out.mkdir(parents=True, exist_ok=True)
    parents = sorted((project / "results/v13/architectures/OPS-SAT").glob("*/*/seed_*/metrics.json"))
    parents += sorted((project / "results/v14/continuation/OPS-SAT").glob("*/*/seed_*/metrics.json"))
    if not parents:
        raise ValueError("No existing OPS-SAT experiments found")
    first = json.loads(parents[0].read_text())
    split = first["dataset_inventory"]["split"]
    csv_path = project / "data/raw/OPSSAT/segments.csv"
    csv_sha = sha(csv_path)
    if csv_sha != split["source_sha256"]:
        raise ValueError("Raw OPS-SAT differs from experiment source")
    train_ids = set(map(int, split["train_fragment_ids"]))
    fragments, cutoffs = load_fragments(csv_path, train_ids)
    write_csv(out / "fragment_inventory.csv", list(fragments.values()))
    (out / "morphology_cutoffs.json").write_text(json.dumps(cutoffs, indent=2, allow_nan=False)+"\n")
    rows, fragment_rows, context_rows, inputs = [], [], [], []
    report_map = {}
    for file in tqdm(parents, desc="OPS-SAT | saved experiments", unit="run"):
        report = json.loads(file.read_text())
        p = report["profile"]
        if report["status"] != "complete" or report["dataset_inventory"]["split"] != split:
            raise ValueError(f"Incomplete run or different input/split: {file}")
        score_file = file.parent / "scores.npz"
        score_sha = sha(score_file)
        if score_sha != report["artifacts"]["scores.npz"]:
            raise ValueError(f"Score hash mismatch: {score_file}")
        status_file = file.parent / ("diagnostic_status.json" if p["epochs"] == 100 else "status.json")
        status = json.loads(status_file.read_text())
        if status["status"] != "complete":
            raise ValueError(f"Incomplete status: {file}")
        if p["epochs"] == 100 and status["artifacts"]["metrics.json"] != sha(file):
            raise ValueError(f"Metrics hash mismatch: {file}")
        meta = {k: p[k] for k in ("variant", "configuration", "seed", "epochs")}
        meta.update(window=p["window_sz"], ff=p["dim_feedforward"])
        report_map[p["variant"], p["configuration"], p["seed"], p["epochs"]] = report
        with np.load(score_file, allow_pickle=False) as arrays:
            ids, labels = arrays["test_segment_ids"], arrays["test_labels"]
            if set(ids.tolist()) != set(map(int, split["test_fragment_ids"])):
                raise ValueError("Wrong test fragment population")
            for a, b in blocks(ids):
                f = fragments[int(ids[a])]
                if f["official_train"] or f["n_samples"] != b-a or not np.all(labels[a:b] == f["label"]):
                    raise ValueError("Raw file and saved score alignment disagree")
            for method in METHODS:
                saved = report["paper_oracle"][method]
                scores = select_scores(arrays, saved)
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite scores")
                per_fragment = fragment_counts(scores, labels, ids, saved[method+"_threshold"])
                whole = metrics(per_fragment)
                for key, original in (("f1_pa", "f1"), ("precision_pa", "precision"), ("recall_pa", "recall"), ("auc_pa", "ROC/AUC")):
                    if not np.isclose(whole[key], saved[method+"_"+original], atol=1e-10, rtol=0):
                        raise ValueError(f"Cannot reproduce original {method} {key}: {file}")
                for (kind, value), part in cohorts(per_fragment, fragments).items():
                    rows.append({**meta, "method": method, "cohort": kind, "value": value,
                                 "alpha": saved["alpha"], "threshold": saved[method+"_threshold"], **metrics(part)})
                fragment_rows.extend({**meta, "method": method, **r} for r in per_fragment)
        inputs.append({"metrics": str(file.relative_to(project)), "metrics_sha256": sha(file), "scores_sha256": score_sha})

    # Single-output controls have exactly the same score definition as their
    # original BF/POT reports. Thresholds stay fixed for all three conditions.
    controls = sorted((project / "results/v14/context/OPS-SAT/encoder_only").glob("*/seed_*/metrics.json"))
    for file in tqdm(controls, desc="OPS-SAT | frozen context", unit="run"):
        d = json.loads(file.read_text()); p = d["profile"]
        status = json.loads((file.parent / "diagnostic_status.json").read_text())
        if sha(file) != status["artifacts"]["metrics.json"] or sha(file.parent / "scores.npz") != status["artifacts"]["scores.npz"]:
            raise ValueError("Frozen context artifact changed")
        parent = report_map[p["variant"], p["configuration"], p["seed"], 50]
        with np.load(file.parent / "scores.npz", allow_pickle=False) as a:
            for i, condition in enumerate(a["conditions"]):
                for method in METHODS:
                    chosen = parent["paper_oracle"][method]
                    parts = fragment_counts(a["test_scores"][i], a["test_labels"], a["test_segment_ids"], chosen[method+"_threshold"])
                    if str(condition) == "original":
                        original = metrics(parts)
                        for key, suffix in (("f1_pa", "f1"), ("precision_pa", "precision"), ("recall_pa", "recall"), ("auc_pa", "ROC/AUC")):
                            if not np.isclose(original[key], chosen[method+"_"+suffix], atol=1e-10, rtol=0):
                                raise ValueError(f"Frozen original differs from parent {key}: {file}")
                    for (kind, value), part in cohorts(parts, fragments).items():
                        context_rows.append({"variant": p["variant"], "configuration": p["configuration"], "epochs": 50,
                                             "seed": p["seed"], "window": p["window_sz"], "ff": p["dim_feedforward"],
                                             "method": method, "condition": str(condition), "cohort": kind, "value": value, **metrics(part)})
        inputs.append({"metrics": str(file.relative_to(project)), "metrics_sha256": sha(file), "scores_sha256": sha(file.parent / "scores.npz")})
    summary = summaries(rows)
    write_csv(out / "subgroup_runs.csv", rows)
    write_csv(out / "subgroup_mean_std.csv", summary)
    write_csv(out / "fragment_predictions.csv", fragment_rows)
    write_csv(out / "paired_window_differences.csv", paired_windows(rows))
    write_csv(out / "frozen_context_runs.csv", context_rows)
    context_summary = summaries(context_rows, context=True)
    write_csv(out / "frozen_context_mean_std.csv", context_summary)
    report_text(out, summary, context_summary, fragments, len(parents))
    if not args.no_figures:
        figures(out, summary, fragments)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
                "source_csv_sha256": csv_sha, "experiments": len(parents), "frozen_context_runs": len(controls),
                "original_metric_parity_checks": 8*len(parents), "frozen_original_parity_checks": 8*len(controls),
                "training_fragment_ids": sorted(train_ids), "test_fragments": len(split["test_fragment_ids"]),
                "metrics": "BF200 and POT; fragment-aware PA; AUC-PA is binary ROC-AUC",
                "threshold_selection": "Saved full-test global winners; no subgroup-specific selection",
                "morphology": "Observable signal descriptors; not fault diagnoses or pointwise anomaly labels",
                "inputs": inputs, "outputs": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(out), "experiments": len(parents), "frozen_context_runs": len(controls),
                      "original_metric_parity_checks": 8*len(parents), "subgroup_rows": len(rows)}))


if __name__ == "__main__":
    main()
