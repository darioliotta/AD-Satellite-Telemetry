"""Matched, common-population window diagnostics on all saved v10/v12 runs."""
import csv
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import statistics as st
import time
import numpy as np

from tranad_variants.common import hash_file, write_json, digest, now
from .metrics import ranking, raw_best_f1, fixed_threshold, real_point_mask, dense_pa_f1, event_metrics
from .progress import bar, record


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def entries(results_root, settings, datasets=None, seeds=None):
    records = []
    for version in settings["versions"]:
        root = results_root / version / "study"
        with (root / "all_experiments.csv").open() as stream:
            source_rows = list(csv.DictReader(stream))
        for row in source_rows:
            if row["dataset"] not in (datasets or settings["datasets"]) or int(row["seed"]) not in (seeds or settings["seeds"]):
                continue
            report_path = (root / row["report_file"]).resolve()
            if not report_path.is_relative_to(root.resolve()):
                raise ValueError("Window report outside its source study")
            report = json.loads(report_path.read_text())
            if (report["identifier"], report["dataset"], report["checkpoint"]["sha256"]) != (
                    row["identifier"], row["dataset"], row["checkpoint_sha256"]):
                raise ValueError(f"Window CSV/report identity mismatch: {report_path}")
            if (report['profile']['identifier'], report['profile']['dataset'], report['profile']['seed']) != (
                    row['identifier'], row['dataset'], int(row['seed'])):
                raise ValueError(f'Window profile identity mismatch: {report_path}')
            records.append((version, report_path, report))
    groups = defaultdict(list)
    seen = set()
    for version, path, report in records:
        p = report["profile"]
        key = (p["dataset"], p["window_sz"], p["dim_feedforward"], p["seed"])
        if key in seen:
            raise ValueError(f"Duplicate first-ablation setting: {key}")
        seen.add(key)
        groups[p["dataset"]].append(p)
    for ds in datasets or settings["datasets"]:
        part = groups[ds]
        expected_seeds = [s for s in (seeds or settings["seeds"]) if s in settings["seeds"]]
        if (not expected_seeds or len(part) != settings["windows_per_dataset"]*settings["ff_widths_per_dataset"]*len(expected_seeds)
                or len({r["window_sz"] for r in part}) != settings["windows_per_dataset"]
                or len({r["dim_feedforward"] for r in part}) != settings["ff_widths_per_dataset"]):
            raise ValueError(f"Incomplete first-ablation grid for {ds}; expected 6 windows x 5 FF x {len(expected_seeds)} seeds")
    return sorted(records, key=lambda r: (r[2]["dataset"], r[2]["profile"]["dim_feedforward"],
                                         r[2]["profile"]["window_sz"], r[2]["profile"]["seed"]))


def verify(version, path, report, data_root, models_sha, *, verify_checkpoint=True):
    if report["source"]["models_sha256"] != models_sha:
        raise ValueError(f"Window model implementation differs: {path}")
    if report["profile"]["epochs"] != 50:
        raise ValueError("Window parents must be 50-epoch runs")
    raw = report["raw_pointwise_evaluation"]
    alpha = raw["alpha_fixed_pre_test"]
    if alpha != .5:
        raise ValueError("Window score export does not use fixed alpha 0.5")
    source = path.parent / raw["scores_file"]
    if not source.resolve().is_relative_to(path.parent.resolve()) or hash_file(source) != raw["scores_sha256"]:
        raise ValueError(f"Window score export missing/changed: {source}")
    checkpoint = path.parent / report["checkpoint"]["file"]
    if verify_checkpoint and (not checkpoint.resolve().is_relative_to(path.parent.resolve())
                              or hash_file(checkpoint) != report["checkpoint"]["sha256"]):
        raise ValueError(f"Window checkpoint missing/changed: {checkpoint}")
    return source


def fragment_metrics(scores, labels, ids):
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1]
    ends = np.r_[starts[1:], len(ids)]
    segment_labels = labels[starts]
    if any(not np.all(labels[a:b] == labels[a]) for a, b in zip(starts, ends)):
        raise ValueError("OPS-SAT labels are not constant inside saved fragments")
    out = {}
    for name, fun in (("mean", np.mean), ("max", np.max)):
        values = np.array([fun(scores[a:b]) for a, b in zip(starts, ends)])
        out[f"fragment_{name}_auc"] = ranking(segment_labels, values)["auc"]
    return out, starts[1:]


def interior_mask(profile, data_root, maximum_window, count, ids=None):
    """Same physical points with a full real history for the longest window."""
    if ids is not None:
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1]
        ends = np.r_[starts[1:], len(ids)]
        mask = np.zeros(len(ids), dtype=bool)
        for a, b in zip(starts, ends):
            mask[min(a+maximum_window-1, b):b] = True
    else:
        paths = sorted((data_root / 'processed/tranad_plus' / profile['dataset'] / 'Test/Tracks').glob('*.npy'))
        mask = np.concatenate([np.arange(len(np.load(p, mmap_mode='r'))) >= maximum_window-1 for p in paths])
    if len(mask) != count:
        raise ValueError('Interior mask population mismatch')
    return mask


def run_window_study(results_root, data_root, folder, settings, models_sha, *, datasets=None, seeds=None, check=False):
    results_root, data_root = results_root.resolve(), data_root.resolve()
    parents = entries(results_root, settings, datasets, seeds)
    inputs, rows, label_hashes = [], [], {}
    started = time.monotonic()
    source_csv_sha = None
    maximum_windows = {ds: max(r[2]['profile']['window_sz'] for r in parents if r[2]['dataset'] == ds)
                       for ds in (datasets or settings['datasets'])}
    for n, (version, path, report) in enumerate(bar(parents, total=len(parents), desc="V14 window study | v10/v12",
                                                  position=1, unit="run", leave=False), 1):
        source = verify(version, path, report, data_root, models_sha)
        p, raw = report["profile"], report["raw_pointwise_evaluation"]
        if p["dataset"] == "OPS-SAT":
            if source_csv_sha is None:
                source_csv_sha = hash_file(data_root / "raw/OPSSAT/segments.csv")
            if source_csv_sha != report["dataset_inventory"]["source_sha256"]:
                raise ValueError("OPS-SAT source differs from first-ablation input")
        if check:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(source, allow_pickle=False) as z:
            scores, labels, train = z["scores"], z["labels"], z["train_scores"]
            if not np.isfinite(scores).all() or not np.isfinite(train).all():
                raise ValueError("Nonfinite scores in first-ablation export")
            mask = real_point_mask(p, data_root, len(labels))
            s, y = scores[mask], labels[mask]
            signature = digest({"labels_sha256": hashlib.sha256(y.tobytes()).hexdigest(), "n": len(y)})
            if p["dataset"] in label_hashes and label_hashes[p["dataset"]] != signature:
                raise ValueError("Different physical test populations between window sizes")
            label_hashes[p["dataset"]] = signature
            boundaries = None
            row = {"dataset": p["dataset"], "version": version, "identifier": p["identifier"],
                   "window": p["window_sz"], "ff": p["dim_feedforward"], "seed": p["seed"],
                   "points_original": len(labels), "points_real": len(y), "removed_padding_targets": int((~mask).sum())}
            if p["dataset"] == "OPS-SAT":
                fragments, boundaries = fragment_metrics(s, y, z["segment_ids"])
                row.update(fragments)
            interior = interior_mask(p, data_root, maximum_windows[p['dataset']], len(y),
                                     z['segment_ids'] if p['dataset'] == 'OPS-SAT' else None)
            row.update(points_interior=int(interior.sum()),
                       interior_raw_auc=ranking(y[interior], s[interior])['auc'],
                       interior_raw_best_f1=raw_best_f1(y[interior], s[interior])['f1'])
            rank = ranking(y, s)
            if not np.isclose(ranking(labels, scores)["auc"], raw["roc_auc_continuous_scores"], atol=1e-10):
                raise ValueError("Saved score AUC does not reproduce its original report")
            original = report["original_paper_evaluation"]["metrics"]
            row.update(original_pot_pa_f1=original["pot"]["pot_f1"],
                       original_bf200_pa_f1=original["bf"]["bf_f1"],
                       original_dense_pa_f1=original["bf"]["bf_dense_f1"],
                       raw_auc_real=rank["auc"], raw_ap_real=rank["average_precision"],
                       raw_best_f1_real=raw_best_f1(y, s)["f1"],
                       fixed_alpha_dense_pa_f1_real=dense_pa_f1(s, y, boundaries)["f1"])
            threshold = float(np.quantile(train, .995))
            row["raw_train_quantile_f1_real"] = fixed_threshold(y, s, threshold)["f1"]
            fpr_threshold = float(np.quantile(s[y == 0], 1-settings["diagnostic_test_normal_fpr"], method="higher"))
            row.update({"matched_fpr_"+k: v for k, v in event_metrics(s, y, fpr_threshold, boundaries).items()})
            for label, values in (("normal", s[y == 0]), ("anomaly", s[y == 1])):
                for q in (.5, .95, .99):
                    row[f"{label}_q{int(q*100)}"] = float(np.quantile(values, q))
            rows.append(row)
        inputs.append({"version": version, "report": str(path.relative_to(results_root)),
                       "report_sha256": hash_file(path), "scores_sha256": raw["scores_sha256"],
                       "checkpoint_sha256": report["checkpoint"]["sha256"], "source": report["source"]})
        record(folder, "window_analysis", completed_runs=n, total_runs=len(parents),
               elapsed_seconds=time.monotonic()-started,
               remaining_seconds_estimate=(time.monotonic()-started)/n*(len(parents)-n))
    if check:
        return {"checked_runs": len(parents)}
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["ff"], row["window"]].append(row)
    fields = [k for k, v in rows[0].items() if isinstance(v, (float, int)) and k not in ("seed", "window", "ff")]
    summaries = []
    for (ds, ff, w), part in sorted(groups.items()):
        summary = {"dataset": ds, "ff": ff, "window": w, "n": len(part)}
        for k in fields:
            if all(k in r for r in part):
                values = [r[k] for r in part]
                summary[k+"_mean"] = st.mean(values)
                summary[k+"_sd"] = st.stdev(values) if len(values)>1 else None
        for k in ("fragment_mean_auc", "fragment_max_auc"):
            if k in part[0]:
                values = [r[k] for r in part]
                summary[k+"_mean"] = st.mean(values)
                summary[k+"_sd"] = st.stdev(values) if len(values)>1 else None
        summaries.append(summary)
    lookup = {(r["dataset"], r["ff"], r["window"], r["seed"]): r for r in rows}
    paired = []
    for row in rows:
        maximum = max(r["window"] for r in rows if r["dataset"] == row["dataset"])
        if row["window"] == maximum:
            continue
        baseline = lookup[row["dataset"], row["ff"], maximum, row["seed"]]
        pair = {k: row[k] for k in ("dataset", "ff", "seed", "window")}
        pair["reference_window"] = maximum
        for k in ("original_dense_pa_f1", "fixed_alpha_dense_pa_f1_real", "raw_auc_real", "raw_best_f1_real",
                  "matched_fpr_direct_point_recall", "matched_fpr_event_recall", "fragment_mean_auc", "interior_raw_auc"):
            if k in row:
                pair[k+"_delta_pp"] = 100*(row[k]-baseline[k])
        paired.append(pair)
    csv_write(folder / "runs.csv", rows)
    csv_write(folder / "mean_std.csv", summaries)
    csv_write(folder / "paired_window_effects.csv", paired)
    make_figures(folder, summaries, settings)
    write_json(folder / "provenance.json", {"created": now(), "settings": settings, "inputs": inputs,
               "rows": len(rows), "cells": len(summaries), "paired_comparisons": len(paired),
               "physical_label_populations": label_hashes,
               "notes": ["Original best metrics retain the original alpha/test oracle selections.",
                         "New diagnostics use fixed alpha=.5 and the same actual test points for all windows.",
                         "Interior metrics exclude the first max(W)-1 real points in every track/fragment for every window.",
                         "Matched FPR uses normal test labels: diagnostic only, not deployment calibration.",
                         "PA is recomputed with native global intervals or OPS-SAT fragment boundaries.",
                         "Pairs match FF/seed; campaign versions and source hashes remain recorded, not assumed identical."]})
    (folder / "README.md").write_text(
        "# V14 — first-ablation window study\n\n"
        f"{len(rows)} saved runs; {len(summaries)} W/FF cells; {len(paired)} comparisons with the longest window, matched by FF and seed.\n\n"
        "Open `window_effects.pdf`. Plots hold FF at the original dataset baseline and show mean ± sample SD across seeds. "
        "`mean_std.csv` contains all FF widths. `paired_window_effects.csv` reports individual seed differences.\n\n"
        "Original PA-oracle metrics and new fixed-alpha diagnostics are explicitly separate. "
        "All newly computed window comparisons use identical physical test points; padded target duplicates are removed. "
        "Interior metrics additionally exclude every track/fragment's initial max(W)-1 points for all window sizes, testing sensitivity to padded history. "
        "OPS-SAT retains fragment boundaries and fixed mean/max fragment AUC. "
        "FPR matching uses test-normal labels and is diagnostic. No model has been trained.\n")
    return {"runs": len(rows), "cells": len(summaries), "paired_comparisons": len(paired)}


def make_figures(folder, summaries, settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(folder / "window_effects.pdf") as pdf:
        for ds in settings["datasets"]:
            part = sorted([r for r in summaries if r["dataset"] == ds and r["ff"] == settings["baseline_ff"][ds]],
                          key=lambda r: r["window"])
            if not part:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
            fig.suptitle(f"{ds} | Window-size control | FF = {settings['baseline_ff'][ds]}",
                         color="#9B0014", fontsize=18, weight="bold")
            panels = [
                [("original_dense_pa_f1", "Original dense F1 + PA", "#777777"),
                 ("fixed_alpha_dense_pa_f1_real", "Fixed-alpha dense F1 + PA", "#9B0014"),
                 ("raw_best_f1_real", "Fixed-alpha best F1, no PA", "#222222")],
                [("raw_auc_real", "Point-score AUC", "#9B0014"),
                 ("interior_raw_auc", "AUC excluding history padding", "#777777")],
                [("matched_fpr_direct_point_recall", "Direct point recall", "#222222"),
                 ("matched_fpr_event_recall", "Intervals with a hit", "#9B0014")]]
            if ds == "OPS-SAT":
                panels[1].append(("fragment_mean_auc", "Mean-fragment AUC", "#222222"))
            for ax, panel, title in zip(axes, panels, ("Threshold / PA sensitivity", "Ranking without PA", "Same 0.5% normal-point FPR")):
                for k, label, color in panel:
                    x = [r["window"] for r in part]
                    y = np.array([r[k+"_mean"] for r in part])*100
                    sd = np.array([r[k+"_sd"] or 0 for r in part])*100
                    ax.plot(x, y, "o-", color=color, label=label, linewidth=1.7, markersize=4)
                    ax.fill_between(x, y-sd, y+sd, color=color, alpha=.10)
                ax.set(title=title, xlabel="Window size", ylabel="%")
                ax.grid(alpha=.15)
                ax.legend(fontsize=8)
            fig.supxlabel("Mean ± sample SD across matched seeds. Diagnostic test-label searches are not deployment estimates.", fontsize=9)
            pdf.savefig(fig)
            fig.savefig(folder / f"{ds.lower()}_window_effects.png", dpi=200)
            plt.close(fig)
