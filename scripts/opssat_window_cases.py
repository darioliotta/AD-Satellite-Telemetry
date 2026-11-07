#!/usr/bin/env python3
"""Matched W5/W10 signal review and exact score/alpha/threshold bookkeeping.

Uses existing FF8, 50/100 epoch, five-seed TranAD+/encoder-only endpoints.
No training, dense search, subgroup threshold optimization or model inference.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from itertools import permutations
import json
import os
from pathlib import Path
import statistics as st
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "opssat-case-review-mpl"))
import numpy as np
from tqdm import tqdm

from opssat_context import blocks, load_fragments, sha, write_csv

MODELS = ("reference", "encoder_only")
METHODS = ("bf", "pot")
SEEDS = tuple(range(2026, 2031))
FACTORS = ("model_window", "alpha", "threshold")
KINDS = ("short_only", "long_only", "both", "neither", "seed_dependent")
RED, GRAY = "#9B0014", "#444444"
NAMES = {"reference": "TranAD+", "encoder_only": "Encoder-only", "bf": "Best F1 (200 thresholds)", "pot": "POT"}


def shapley(cube):
    """Average exact finite differences over all six orders of three changes."""
    result = {k: np.zeros_like(np.asarray(cube[0], dtype=float)) for k in FACTORS}
    for order in permutations(range(3)):
        mask = 0
        for bit in order:
            nxt = mask | (1 << bit)
            result[FACTORS[bit]] += (np.asarray(cube[nxt])-np.asarray(cube[mask]))/6
            mask = nxt
    np.testing.assert_allclose(sum(result.values()), np.asarray(cube[7])-np.asarray(cube[0]), atol=1e-12, rtol=0)
    return result


def score_row(data, alpha):
    if alpha is None:
        if data["scores"].shape[0] != 1:
            raise ValueError("Single-output model has multiple score rows")
        return data["scores"][0]
    found = np.flatnonzero(np.isclose(data["alphas"], alpha, atol=1e-12, rtol=0))
    if len(found) != 1:
        raise ValueError("Requested alpha is absent from saved score grid")
    return data["scores"][int(found[0])]


def confusion(hit, direct, labels, lengths):
    positive = labels.astype(bool)
    tp = int(lengths[positive & hit].sum())
    fp = int(direct[~positive].sum())
    fn = int(lengths[positive].sum())-tp
    tn = int(lengths[~positive].sum())-fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": 2*tp/(2*tp+fp+fn), "precision": tp/(tp+fp) if tp+fp else 0.,
            "recall": tp/(tp+fn), "auc_pa": .5*(tp/(tp+fn)+tn/(tn+fp))}


def cube_decisions(long, short, method):
    """Cross saved score grids, saved alpha choices and saved thresholds.

    bit 0: W10 -> W5 trained endpoint; bit 1: alpha10 -> alpha5;
    bit 2: threshold10 -> threshold5. Endpoints 0 and 7 are actual reports.
    """
    choices = [d["report"]["paper_oracle"][method] for d in (long, short)]
    starts = long["starts"]
    hits, direct, metrics = {}, {}, {}
    for mask in range(8):
        data = short if mask & 1 else long
        alpha = choices[int(bool(mask & 2))]["alpha"]
        threshold = choices[int(bool(mask & 4))][method+"_threshold"]
        score = score_row(data, alpha)
        hit = np.maximum.reduceat(score, starts) > threshold
        count = np.add.reduceat((score > threshold).astype(np.int64), starts)
        hits[mask], direct[mask] = hit, count
        metrics[mask] = confusion(hit, count, long["fragment_labels"], long["lengths"])
    for mask, data in ((0, long), (7, short)):
        saved = data["report"]["paper_oracle"][method]
        for metric, suffix in (("f1", "f1"), ("precision", "precision"), ("recall", "recall"), ("auc_pa", "ROC/AUC")):
            if not np.isclose(metrics[mask][metric], saved[method+"_"+suffix], atol=1e-10, rtol=0):
                raise ValueError(f"Actual endpoint does not reproduce {method}/{metric}")
    return hits, direct, metrics


def category(short, long):
    return "both" if short and long else "short_only" if short else "long_only" if long else "neither"


def consensus(rows):
    counts = {k: sum(r["decision_category"] == k for r in rows) for k in KINDS[:-1]}
    stable = next((k for k, n in counts.items() if n >= 4), "seed_dependent")
    return {"category": stable, **{k+"_votes": n for k, n in counts.items()},
            "short_hits": sum(r["short_hit"] for r in rows), "long_hits": sum(r["long_hit"] for r in rows),
            "hit_delta_mean": st.mean(r["short_hit"]-r["long_hit"] for r in rows),
            **{k+"_contribution_mean": st.mean(r[k+"_contribution"] for r in rows) for k in FACTORS}}


def attribution(values, changed):
    if not changed:
        return "unchanged"
    total = sum(abs(values[k]) for k in FACTORS)
    best = max(FACTORS, key=lambda k: abs(values[k]))
    return best if abs(values[best]) > total/2+1e-12 else "mixed"


def load_endpoint(project, model, config, epoch, seed, source_sha, inputs):
    version = "v13/architectures" if epoch == 50 else "v14/continuation"
    folder = project / "results" / version / "OPS-SAT" / model / config / f"seed_{seed}"
    file = folder / "metrics.json"
    report = json.loads(file.read_text())
    p = report["profile"]
    if report["status"] != "complete" or p["dim_feedforward"] != 8 or p["epochs"] != epoch:
        raise ValueError(f"Unexpected profile: {file}")
    if report["dataset_inventory"]["split"]["source_sha256"] != source_sha:
        raise ValueError("Dataset source differs")
    expected_w = 10 if config == "reference" else 5
    if (p["dataset"], p["variant"], p["seed"], p["window_sz"]) != ("OPS-SAT", model, seed, expected_w):
        raise ValueError("Endpoint identity mismatch")
    saved_sha = sha(folder / "scores.npz")
    if saved_sha != report["artifacts"]["scores.npz"]:
        raise ValueError("Saved scores changed")
    with np.load(folder / "scores.npz", allow_pickle=False) as a:
        data = {"report": report, "folder": folder, "scores": a["test_scores"].copy(),
                "alphas": a["alpha_values"].copy(), "ids": a["test_segment_ids"].copy(),
                "labels": a["test_labels"].copy()}
    if not np.isfinite(data["scores"]).all():
        raise ValueError("Nonfinite scores")
    ranges = blocks(data["ids"])
    data["starts"] = np.array([a for a, b in ranges])
    data["ends"] = np.array([b for a, b in ranges])
    data["lengths"] = data["ends"]-data["starts"]
    data["fragment_ids"] = data["ids"][data["starts"]]
    data["fragment_labels"] = data["labels"][data["starts"]]
    for a, b in ranges:
        if not np.all(data["labels"][a:b] == data["labels"][a]):
            raise ValueError("Fragment label changed inside a saved block")
    inputs.append({"metrics": str(file.relative_to(project)), "metrics_sha256": sha(file), "scores_sha256": saved_sha})
    return data


def raw_signals(path):
    groups = defaultdict(list)
    with path.open() as stream:
        for r in csv.DictReader(stream):
            groups[int(r["segment"])].append((r["timestamp"], float(r["value"])))
    result = {}
    for i, rows in groups.items():
        dates = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t, v in rows]
        result[i] = {"t": np.array([(t-dates[0]).total_seconds() for t in dates]),
                     "x": np.array([v for t, v in rows])}
    return result


def peak_alignment(signal, short_scores, long_scores):
    """Error-peak proximity to the two largest sample-to-sample transitions.

    Both score peaks and transition endpoints are restricted to index >=9,
    the common region with fully observed W10 history. This is a signal-shape
    diagnostic, not localization against an expert anomaly boundary.
    """
    x=signal['x']
    if len(x)<=9:
        return {'peak_alignment_available':0}
    indices=np.arange(9,len(x))
    order=np.argsort(-np.abs(np.diff(x))[indices-1],kind='stable')
    transitions=indices[order[:2]]
    result={'peak_alignment_available':1}
    for name,scores in (('short',short_scores),('long',long_scores)):
        if np.ptp(scores[9:]) == 0:
            result['peak_alignment_available']=0
            return result
        peak=9+int(np.argmax(scores[9:]))
        distance=int(np.min(np.abs(transitions-peak)))
        result[name+'_peak_index']=peak
        result[name+'_peak_step_distance']=distance
        for radius in (0,2,5):
            result[f'{name}_peak_near_step_r{radius}']=int(distance<=radius)
    return result


def peak_summary(decisions):
    groups=defaultdict(list)
    for r in decisions:
        if not r['peak_alignment_available'] or r['label']!=1:
            continue
        groups[r['variant'],r['epochs'],r['method'],r['sensor'],r['seed']].append(r)
    per_seed=[]
    for (model,epoch,method,sensor,seed),part in sorted(groups.items()):
        per_seed.append(dict(variant=model,epochs=epoch,method=method,sensor=sensor,seed=seed,n_fragments=len(part),
                            **{f'{w}_r{rad}':st.mean(r[f'{w}_peak_near_step_r{rad}'] for r in part)
                               for w in ('short','long') for rad in (0,2,5)}))
    grouped=defaultdict(list)
    for r in per_seed:grouped[r['variant'],r['epochs'],r['method'],r['sensor']].append(r)
    result=[]
    for (model,epoch,method,sensor),part in sorted(grouped.items()):
        result.append(dict(variant=model,epochs=epoch,method=method,sensor=sensor,n_seeds=len(part),
                           n_fragments_min=min(r['n_fragments'] for r in part),
                           n_fragments_max=max(r['n_fragments'] for r in part),
                           **{f'{w}_r{rad}_mean':st.mean(r[f'{w}_r{rad}'] for r in part)
                              for w in ('short','long') for rad in (0,2,5)}))
    return result


def group_effects(decisions, fragments):
    grouped = defaultdict(list)
    for r in decisions:
        f = fragments[r["segment"]]
        for scope, value in (("all", "all"), ("sensor", f["sensor"]), ("channel", f["channel"])):
            descriptors = [("all", "all")]
            descriptors += [(k, str(f[k])) for k in ("duration_bin", "local_departure_bin", "endpoint_drift_bin", "large_gap")]
            for descriptor, level in descriptors:
                grouped[r["variant"], r["epochs"], r["method"], r["label"], scope, value, descriptor, level, r["seed"]].append(r)
    seed_rows = []
    for key, part in sorted(grouped.items()):
        variant, epoch, method, label, scope, value, desc, level, seed = key
        seed_rows.append(dict(variant=variant, epochs=epoch, method=method, label=label, scope=scope, value=value,
                              descriptor=desc, level=level, seed=seed, n_fragments=len(part),
                              short_rate=st.mean(r["short_hit"] for r in part), long_rate=st.mean(r["long_hit"] for r in part),
                              delta_pp=100*st.mean(r["short_hit"]-r["long_hit"] for r in part)))
    grouped = defaultdict(list)
    for r in seed_rows:
        key = tuple(r[k] for k in ("variant", "epochs", "method", "label", "scope", "value", "descriptor", "level"))
        grouped[key].append(r)
    result = []
    for key, part in sorted(grouped.items()):
        row = dict(zip(("variant", "epochs", "method", "label", "scope", "value", "descriptor", "level"), key))
        if len(part) != 5 or len({r["n_fragments"] for r in part}) != 1:
            raise ValueError("Group comparison lost matched test fragments or seeds")
        n = part[0]["n_fragments"]
        result.append({**row, "n_fragments": n, "n_seeds": 5, "sparse_group": n < 5,
                       "rate_definition": "detected anomalous fragments" if row["label"] else "nominal fragments with any false alarm",
                       **{k+"_mean": st.mean(r[k] for r in part) for k in ("short_rate", "long_rate", "delta_pp")},
                       "delta_pp_sd": st.stdev(r["delta_pp"] for r in part)})
    return result


def select_cases(consensus_rows, decisions, fragments):
    """Median-duration examples per complete, predeclared outcome category."""
    selected = []
    eligible = {i for i, f in fragments.items() if f["official_train"] == 0}
    support = defaultdict(lambda: [0, 0])
    for i in eligible:
        f = fragments[i]; support[f["channel"]][f["label"]] += 1
    for variant in MODELS:
        for epoch in (50, 100):
            for method in METHODS:
                for label in (1, 0):
                    for cat in KINDS[:-1]:
                        pool = [r for r in consensus_rows if (r["variant"], r["epochs"], r["method"], r["label"], r["category"]) == (variant, epoch, method, label, cat)
                                and min(support[r["channel"]]) >= 5]
                        if not pool:
                            continue
                        med = float(np.median([fragments[r["segment"]]["duration_s"] for r in pool]))
                        chosen = min(pool, key=lambda r: (abs(fragments[r["segment"]]["duration_s"]-med), r["segment"]))
                        seed_rows = [r for r in decisions if r["segment"] == chosen["segment"] and
                                     (r["variant"], r["epochs"], r["method"], r["decision_category"]) == (variant, epoch, method, cat)]
                        seed = min((r["seed"] for r in seed_rows), key=lambda s: (abs(s-2028), s))
                        selected.append({**chosen, "representative_seed": seed, "eligible_fragments": len(pool),
                                         "pool_median_duration_s": med, "selection_rule": "median duration; segment ID breaks ties; closest agreeing seed to 2028"})
    return selected


def draw_case(axs, case, cache, raw, fragments):
    if case is None:
        for ax in axs:
            ax.axis("off")
        axs[0].text(.1, .5, "No case meets the 4/5-seed rule\nin a channel with >=5 examples per class.", transform=axs[0].transAxes)
        return
    i, seed, method = case["segment"], case["representative_seed"], case["method"]
    f = fragments[i]; signal = raw[i]; t = signal["t"]
    long = cache[case["variant"], case["epochs"], seed, 10]
    short = cache[case["variant"], case["epochs"], seed, 5]
    j = int(np.flatnonzero(long["fragment_ids"] == i)[0]); a, b = long["starts"][j], long["ends"][j]
    names = {"short_only": "W5 only", "long_only": "W10 only", "both": "Both", "neither": "Neither"}
    axs[0].set_title(f"{names[case['category']]} | {f['channel']} | segment {i}\n{f['sensor']} | {case[case['category']+'_votes']}/5 seeds agree; shown seed {seed}", fontsize=10)
    axs[0].plot(t, signal["x"], color="#333333", lw=1, marker=".", ms=2)
    axs[0].set_ylabel("Telemetry\n(raw CSV scale)")
    predictions = []
    for w, data, color in ((5, short, RED), (10, long, GRAY)):
        chosen = data["report"]["paper_oracle"][method]
        s = score_row(data, chosen["alpha"])[a:b]
        threshold = chosen[method+"_threshold"]
        label = f"W{w}; alpha={chosen['alpha']}" if chosen['alpha'] is not None else f"W{w}"
        axs[1].plot(t, s, color=color, lw=1, label=label)
        axs[1].axhline(threshold, color=color, ls="--", lw=1)
        predictions.append((s > threshold).astype(int))
    axs[1].set_ylabel("Reconstruction score\n(dashed: threshold)")
    axs[1].legend(fontsize=8, loc="best")
    for row, pred, color, name in zip((1, 0), predictions, (RED, GRAY), ("W5", "W10")):
        axs[2].plot(t, row+.65*pred, color=color, drawstyle="steps-post", lw=1)
    axs[2].set_yticks([.325, 1.325], ["W10", "W5"])
    axs[2].set_ylim(-.15, 1.85); axs[2].set_ylabel("Direct alarms\n(before PA)")
    axs[2].set_xlabel("Seconds from fragment start")
    for ax in axs:
        ax.grid(alpha=.15)
    if len(t) >= 10:
        for ax in axs:
            ax.axvspan(t[0], t[9], color="#CCCCCC", alpha=.16)


def figures(out, consensus_rows, global_rows, selections, cache, raw, fragments):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    dest = out / "figures"; dest.mkdir(exist_ok=True)
    with PdfPages(dest / "outcomes_and_decomposition.pdf") as pdf:
        for method in METHODS:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            for ax, (variant, epoch) in zip(axs.flat, ((m, e) for m in MODELS for e in (50, 100))):
                part = [r for r in consensus_rows if (r["variant"], r["epochs"], r["method"], r["label"]) == (variant, epoch, method, 1)]
                counts = [sum(r["category"] == k for r in part) for k in KINDS]
                ax.bar(range(5), counts, color=[RED, GRAY, "#A9A9A9", "#DDDDDD", "#D7959E"])
                ax.set_xticks(range(5), ["W5 only", "W10 only", "Both", "Neither", "Seed-dependent"], rotation=15, ha="right")
                for x, n in enumerate(counts): ax.text(x, n+.6, str(n), ha="center")
                ax.set_title(f"{NAMES[variant]} | {epoch} epochs"); ax.set_ylim(0, max(counts)+12)
                ax.set_ylabel("Anomalous fragments (113 total)"); ax.grid(axis="y", alpha=.15)
            fig.suptitle(f"OPS-SAT | {NAMES[method]} | W5 versus W10, FF8\nStable outcomes require agreement in at least 4/5 seeds", color=RED)
            fig.tight_layout(rect=(0, 0, 1, .92)); pdf.savefig(fig); plt.close(fig)
        for method in METHODS:
            fig, ax = plt.subplots(figsize=(10, 5))
            for k, color in zip(FACTORS, (RED, "#C78691", GRAY)):
                xs=[]; means=[]; std=[]
                for j, (variant, epoch) in enumerate((m, e) for m in MODELS for e in (50, 100)):
                    part = [r for r in global_rows if (r["variant"], r["epochs"], r["method"]) == (variant, epoch, method)]
                    v = [r["f1_"+k+"_pp"] for r in part]
                    xs.append(j+(FACTORS.index(k)-1)*.22); means.append(st.mean(v)); std.append(st.stdev(v))
                ax.bar(xs, means, width=.2, color=color, label=k.replace('_', ' '), yerr=std, capsize=3)
            ax.axhline(0, color="#777777", lw=.8)
            ax.set_xticks(range(4), [f"{NAMES[m]}\n{e} epochs" for m in MODELS for e in (50, 100)])
            ax.set_ylabel("Contribution to W5-minus-W10 F1-PA (percentage points)")
            ax.set_title(f"{NAMES[method]} | exact decision bookkeeping, not physical causality", color=RED)
            ax.legend(); ax.grid(axis="y", alpha=.15); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    by_key = {(r["variant"], r["epochs"], r["method"], r["label"], r["category"]): r for r in selections}
    plotted=[]
    with PdfPages(dest / "signal_score_examples.pdf") as pdf:
        pages=[]
        for variant in MODELS:
            for epoch in (50, 100):
                for method in METHODS:
                    pages.append((variant, epoch, method, 1, ("short_only", "long_only")))
                pages.append((variant, epoch, "bf", 0, ("short_only", "long_only")))
                pages.append((variant, epoch, "bf", 1, ("both", "neither")))
        page_number=0
        for variant, epoch, method, label, cats in pages:
            cases = [by_key.get((variant, epoch, method, label, cat)) for cat in cats]
            if not any(cases):
                continue
            available=[c for c in cases if c is not None]
            if len(available)==1:
                fig, axs=plt.subplots(3,1,figsize=(11,8),sharex=True,gridspec_kw={"height_ratios":[2,2,1]})
                draw_case(axs,available[0],cache,raw,fragments)
                fig.text(.08,.043,"No eligible stable example exists for the opposite outcome category.",fontsize=8,color="#666666")
            else:
                fig, axs = plt.subplots(3, 2, figsize=(13, 8), sharex="col", gridspec_kw={"height_ratios": [2, 2, 1]})
                for j, case in enumerate(cases): draw_case(axs[:,j], case, cache, raw, fragments)
            label_name = "whole fragment labelled anomalous" if label else "nominal fragments: crossings are false alarms"
            fig.suptitle(f"{NAMES[variant]} | {epoch} epochs | {NAMES[method]}\n{label_name}", color=RED, fontsize=13)
            fig.text(.08, .02, "Median-duration cases among stable outcomes; thresholds are saved global choices. Grey band: first 10 samples. Labels do not localize anomaly onset.", fontsize=8)
            fig.tight_layout(rect=(0, .05, 1, .91)); pdf.savefig(fig)
            page_number+=1
            if page_number <= 4:
                fig.savefig(dest / f"example_page_{page_number:02d}.png", dpi=160)
            for c in cases:
                if c: plotted.append({**c, "pdf_page": page_number})
            plt.close(fig)
    write_csv(out / "plotted_examples.csv", plotted)
    return page_number


def write_report(out, consensus_rows, global_rows, effects, peaks, pages):
    lines=["# OPS-SAT: matched window case study", "",
           "W5 versus W10 at FF8; TranAD+ and encoder-only; 50 and 100 epochs; seeds 2026–2030. All 529 test fragments are included. Only original BF200 and POT are used, with fragment-aware PA and AUC-PA.", "",
           "## Full-population results", "",
           "| Model | Epochs | Method | F1 W10 | F1 W5 | Stable W5-only anomalies | Stable W10-only anomalies | Seed-dependent anomalies |",
           "|---|---:|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for epoch in (50, 100):
            for method in METHODS:
                r=[x for x in global_rows if (x['variant'],x['epochs'],x['method'])==(model,epoch,method)]
                c=[x for x in consensus_rows if (x['variant'],x['epochs'],x['method'],x['label'])==(model,epoch,method,1)]
                lines.append(f"| {NAMES[model]} | {epoch} | {method.upper()} | {100*st.mean(x['long_f1'] for x in r):.2f} | {100*st.mean(x['short_f1'] for x in r):.2f} | {sum(x['category']=='short_only' for x in c)} | {sum(x['category']=='long_only' for x in c)} | {sum(x['category']=='seed_dependent' for x in c)} |")
    lines += ["", "## Detected fragments and false alarms", "",
              "These are fractions of fragments, not pointwise recall/FPR. A nominal fragment counts as alarmed after any direct crossing. Means are over the same five seeds.", "",
              "| Model | Epochs | Method | Anomalous detected W10 | Anomalous detected W5 | Nominal with any alarm W10 | Nominal with any alarm W5 |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for model in MODELS:
        for epoch in (50,100):
            for method in METHODS:
                selected=[r for r in effects if (r['variant'],r['epochs'],r['method'],r['scope'],r['descriptor'])==(model,epoch,method,'all','all')]
                pos=next(r for r in selected if r['label']==1); neg=next(r for r in selected if r['label']==0)
                lines.append(f"| {NAMES[model]} | {epoch} | {method.upper()} | {100*pos['long_rate_mean']:.2f} | {100*pos['short_rate_mean']:.2f} | {100*neg['long_rate_mean']:.2f} | {100*neg['short_rate_mean']:.2f} |")
    lines += ["", "## Alignment with abrupt signal transitions", "",
              "Exploratory shape diagnostic: within the common region starting at sample index 9, compare each error-score maximum with the two largest absolute sample-to-sample changes of the raw signal. Each table cell reports the percentage exactly aligned / within 2 / within 5 sample indices. All three tolerances are shown; no radius is optimized against detection performance. These transition positions are not expert anomaly-onset labels. Repeated timestamps preclude interpreting sample-index distance as a uniform physical delay.", "",
              "| Model | Epochs | Sensor | Eligible anomalous fragments per seed | W10: exact / ±2 / ±5 | W5: exact / ±2 / ±5 |",
              "|---|---:|---|---:|---:|---:|"]
    for r in peaks:
        if r['method']=='bf':
            lv=' / '.join(f"{100*r[f'long_r{rad}_mean']:.2f}" for rad in (0,2,5))
            sv=' / '.join(f"{100*r[f'short_r{rad}_mean']:.2f}" for rad in (0,2,5))
            lines.append(f"| {NAMES[r['variant']]} | {r['epochs']} | {r['sensor']} | {r['n_fragments_min']}–{r['n_fragments_max']} | {lv} | {sv} |")
    lines += ["", "For TranAD+, the score here uses each model's saved BF200 alpha. Encoder-only has no alpha, so its peak-alignment result is identical for BF200 and POT. The diagnostic does not use a threshold or PA.", "",
              "## Score, alpha and threshold contributions", "",
              "For each pair, all eight combinations of the two saved score grids, the two saved alpha choices and the two saved thresholds are evaluated. Their exact finite differences are averaged over all six orders of changing these factors. The three contributions sum to the actual W5-minus-W10 F1 difference. This is algebraic bookkeeping, not a causal physical attribution. The model/window term includes the separately trained weights; alpha and threshold are evaluation choices. Encoder-only has no alpha term. Contributions depend on score scale: rescaling a model's scores and threshold together can change these shares while preserving its actual predictions. They must not be read as intrinsic importance or physical causes.", "",
              "| Model | Epochs | Method | Model/window contribution | Alpha contribution | Threshold contribution | Total F1 change |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for model in MODELS:
        for epoch in (50, 100):
            for method in METHODS:
                r=[x for x in global_rows if (x['variant'],x['epochs'],x['method'])==(model,epoch,method)]
                v=[st.mean(x['f1_'+k+'_pp'] for x in r) for k in FACTORS]
                delta=100*st.mean(x['short_f1']-x['long_f1'] for x in r)
                lines.append(f"| {NAMES[model]} | {epoch} | {method.upper()} | {v[0]:+.2f} | {v[1]:+.2f} | {v[2]:+.2f} | {delta:+.2f} |")
    lines += ["", "## Morphology associations", "",
              "The following differences are percentage points of anomalous-fragment detection rate, W5 minus W10, averaged over seeds. They are not F1 differences. Descriptors use cutoffs fitted on nominal optimization fragments only. `local_departure` measures deviation from adjacent samples in sample order; `endpoint_drift` compares first/last-quarter medians. Neither is an expert fault type. The complete table includes within-channel comparisons to expose composition effects.", "",
              "| Model | Epochs | Method | Sensor | Descriptor | Group | Anomalous fragments | Detection-rate change |",
              "|---|---:|---|---|---|---|---:|---:|"]
    for r in effects:
        if r['label']==1 and r['scope']=='sensor' and r['descriptor'] in ('local_departure_bin','endpoint_drift_bin'):
            lines.append(f"| {NAMES[r['variant']]} | {r['epochs']} | {r['method'].upper()} | {r['value']} | {r['descriptor']} | {r['level']} | {r['n_fragments']} | {r['delta_pp_mean']:+.2f} |")
    lines += ["", "## Examples and selection", "",
              f"`figures/signal_score_examples.pdf` contains {pages} pages. The complete outcomes are in `fragment_consensus.csv` and `fragment_seed_decisions.csv`. An outcome is stable if at least four of five seeds agree; remaining fragments are explicitly seed-dependent. This is a descriptive rule, not a statistical significance test.", "",
              "Examples are chosen by median duration within each outcome category, not by largest performance gain or most striking morphology. Only channels with at least five nominal and five anomalous test fragments are eligible for examples. The displayed seed is the closest agreeing seed to 2028. Candidate counts, selected IDs and PDF pages are saved in `selected_cases.csv` and `plotted_examples.csv`. Missing stable categories are left empty. All fragments, including sparse channels, remain in the quantitative tables.", "",
              "The first row is raw telemetry on its supplied scale; the second shows actual continuous reconstruction scores and the two saved thresholds; the third shows direct binary alarms before PA. Whole-fragment annotations are not displayed as precise anomaly onset/offset. No units, orbital state or hardware-fault causes are invented.", "",
              "## Scope and limits", "",
              "- The same fragment identities, labels, split, normalization and sample ordering are required across each pair. Original global F1, precision, recall and AUC-PA are reproduced before analysis.",
              "- More detections are desirable on anomalous fragments; more alarms are undesirable on nominal fragments. Both populations are analyzed separately.",
              "- Each run retains its original full-test BF200/POT winner. No threshold or alpha is optimized for a subgroup or example. Crossed settings are diagnostic combinations, not newly selected winners.",
              "- Morphology associations can reflect channel composition, duration, score calibration, or training effects; they do not prove a physical mechanism. Small groups and seed variation remain visible.",
              "- Different seeds share the same test set. Their variability is not uncertainty over independent telemetry acquisitions.",
              "- Repeated timestamps are preserved. Shape descriptors are sample-index quantities, not time derivatives. Physical window spans use observed timestamps and omit padded history.",
              "- The study can support statements about signal morphology and temporal scales. Attributing a specific event to rotation, eclipse or a hardware fault requires independent operational information.", "",
              "## Reproduce", "", "```bash", "python -B scripts/opssat_window_cases.py", "```", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args=parser.parse_args(); project=args.project_root.resolve()
    out=args.output_root.resolve() if args.output_root else project / "results/analysis_opssat_window_cases"
    if any(out.is_relative_to(project/p) for p in ("data","results/v13","results/v14")):
        raise ValueError("Output must be separate from source campaigns")
    out.mkdir(parents=True,exist_ok=True)
    source=project / "data/raw/OPSSAT/segments.csv"; source_sha=sha(source)
    first=json.loads((project / "results/v13/architectures/OPS-SAT/reference/reference/seed_2026/metrics.json").read_text())
    split=first['dataset_inventory']['split']
    fragments,cutoffs=load_fragments(source,set(map(int,split['train_fragment_ids'])))
    cache={}; inputs=[]; decisions=[]; global_rows=[]; raw=raw_signals(source)
    for model, epoch in tqdm([(m,e) for m in MODELS for e in (50,100)],desc="OPS-SAT matched window pairs",unit="group"):
        for seed in SEEDS:
            for w,config in ((10,"reference"),(5,"window_only")):
                cache[model,epoch,seed,w]=load_endpoint(project,model,config,epoch,seed,source_sha,inputs)
            long,short=cache[model,epoch,seed,10],cache[model,epoch,seed,5]
            for key in ('ids','labels'):
                np.testing.assert_array_equal(long[key],short[key])
            if long['report']['dataset_inventory']['split'] != split or short['report']['dataset_inventory']['split'] != split:
                raise ValueError("Split/normalization population changed")
            for j,i in enumerate(long['fragment_ids']):
                f=fragments[int(i)]
                if f['official_train'] or f['label']!=long['fragment_labels'][j] or f['n_samples']!=long['lengths'][j]:
                    raise ValueError("Original signal and score alignment differ")
            for method in METHODS:
                hits,direct,m=cube_decisions(long,short,method)
                # Cast binary arrays before subtracting: numpy bool subtraction is invalid.
                hit_effects=shapley({k:v.astype(float) for k,v in hits.items()})
                f1_effects=shapley({k:v['f1'] for k,v in m.items()})
                meta=dict(variant=model,epochs=epoch,seed=seed,method=method)
                global_rows.append({**meta,'long_f1':m[0]['f1'],'short_f1':m[7]['f1'],
                                    'long_auc_pa':m[0]['auc_pa'],'short_auc_pa':m[7]['auc_pa'],
                                    'long_scores_at_short_threshold_f1':m[4]['f1'],
                                    'short_scores_at_long_threshold_f1':m[3]['f1'],
                                    **{'f1_'+k+'_pp':100*float(v) for k,v in f1_effects.items()}})
                short_method_scores=score_row(short,short['report']['paper_oracle'][method]['alpha'])
                long_method_scores=score_row(long,long['report']['paper_oracle'][method]['alpha'])
                for j,i in enumerate(long['fragment_ids']):
                    values={k:float(hit_effects[k][j]) for k in FACTORS}
                    a,b=long['starts'][j],long['ends'][j]
                    alignment=peak_alignment(raw[int(i)],short_method_scores[a:b],long_method_scores[a:b])
                    decisions.append({**meta,'segment':int(i),'label':int(long['fragment_labels'][j]),
                                      'channel':fragments[int(i)]['channel'],'sensor':fragments[int(i)]['sensor'],
                                      'short_hit':int(hits[7][j]),'long_hit':int(hits[0][j]),
                                      'decision_category':category(hits[7][j],hits[0][j]),
                                      'attribution':attribution(values,hits[7][j]!=hits[0][j]),
                                      **alignment,
                                      **{k+'_contribution':v for k,v in values.items()}})
    grouped=defaultdict(list)
    for r in decisions:grouped[r['variant'],r['epochs'],r['method'],r['segment']].append(r)
    consensus_rows=[]
    for (model,epoch,method,i),part in sorted(grouped.items()):
        if {r['seed'] for r in part}!=set(SEEDS):raise ValueError("Missing matched seed")
        f=fragments[i]
        consensus_rows.append(dict(variant=model,epochs=epoch,method=method,segment=i,label=f['label'],channel=f['channel'],sensor=f['sensor'],**consensus(part)))
    effects=group_effects(decisions,fragments)
    peaks=peak_summary(decisions)
    selections=select_cases(consensus_rows,decisions,fragments)
    write_csv(out/'fragment_seed_decisions.csv',decisions)
    write_csv(out/'fragment_consensus.csv',consensus_rows)
    write_csv(out/'global_decomposition.csv',global_rows)
    write_csv(out/'morphology_associations.csv',effects)
    write_csv(out/'peak_transition_alignment.csv',peaks)
    write_csv(out/'selected_cases.csv',selections)
    pages=0 if args.no_figures else figures(out,consensus_rows,global_rows,selections,cache,raw,fragments)
    write_report(out,consensus_rows,global_rows,effects,peaks,pages)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
                  shared_analysis_sha256=sha(Path(__file__).with_name('opssat_context.py')),
                  source_csv_sha256=source_sha,endpoints=len(inputs),matched_seed_method_pairs=len(global_rows),
                  fragment_seed_cases=len(decisions),unique_test_fragments=len({r['segment'] for r in decisions}),
                  source_metric_parity_checks=8*len(global_rows),numpy=np.__version__,inputs=inputs,
                  selection='4/5 seed agreement; median duration; adequate class support; deterministic agreeing seed',
                  attribution='Exact symmetric finite-difference allocation; no causal physical interpretation',
                  outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:manifest[k] for k in ('endpoints','matched_seed_method_pairs','fragment_seed_cases','unique_test_fragments')}))
    print('Report:',out/'REPORT.md')


if __name__=='__main__':main()
