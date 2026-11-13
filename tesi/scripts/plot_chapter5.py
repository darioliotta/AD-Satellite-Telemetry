#!/usr/bin/env python3
"""Plot the verified BF200/POT ablation grid without fitting or evaluating models.

Default input: tesi/data/chapter5/runs.csv. All plotted statistics are computed
from the three recorded seeds, never selected seed maxima. PDF output uses
vector cells, lines, markers and text; PNG files are inspection previews.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "toast-thesis-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np


THESIS = Path(__file__).resolve().parents[1]
DATASETS = ("OPS-SAT", "SMAP", "MSL")
SEEDS = {2026, 2027, 2028}
RED = "#9B0014"
GRAY = "#424242"
FORMATS = ("pdf", "png")


def read_runs(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 270:
        raise ValueError(f"Expected 270 runs, received {len(rows)}")
    seen = set()
    for row in rows:
        for key in ("window", "feedforward", "seed", "epochs"):
            row[key] = int(row[key])
        for key in ("bf_f1", "pot_f1"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"Invalid {key}: {row['identifier']}")
        identity = tuple(row[k] for k in ("dataset", "window", "feedforward", "seed"))
        if identity in seen or row["epochs"] != 50:
            raise ValueError(f"Duplicate or unexpected run: {identity}")
        seen.add(identity)
    for dataset in DATASETS:
        part = [row for row in rows if row["dataset"] == dataset]
        windows = sorted({row["window"] for row in part})
        widths = sorted({row["feedforward"] for row in part})
        if len(part) != 90 or len(windows) != 6 or len(widths) != 5:
            raise ValueError(f"Incomplete dataset grid: {dataset}")
        for w in windows:
            for h in widths:
                group = [r for r in part if (r["window"], r["feedforward"]) == (w, h)]
                if len(group) != 3 or {r["seed"] for r in group} != SEEDS:
                    raise ValueError(f"Incomplete seed group: {dataset}, W={w}, FF={h}")
    return rows


def settings() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, output: Path, name: str) -> list[str]:
    paths = []
    for extension in FORMATS:
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.06)
        paths.append(str(path))
    plt.close(fig)
    return paths


def grid_figure(rows: list[dict], output: Path) -> list[str]:
    red_map = LinearSegmentedColormap.from_list("padova", ["#fff8f8", "#e5b6be", RED, "#470009"])
    gray_map = LinearSegmentedColormap.from_list("spread", ["#fafafa", "#a7a7a7", "#303030"])
    parts, spreads = {}, []
    for dataset in DATASETS:
        part = [r for r in rows if r["dataset"] == dataset]
        windows = sorted({r["window"] for r in part})
        widths = sorted({r["feedforward"] for r in part})
        means, sds = [], []
        for h in widths:
            group_means, group_sds = [], []
            for w in windows:
                values = [100*r["bf_f1"] for r in part if (r["window"], r["feedforward"]) == (w, h)]
                group_means.append(statistics.mean(values))
                group_sds.append(statistics.stdev(values))
            means.append(group_means)
            sds.append(group_sds)
            spreads.extend(group_sds)
        parts[dataset] = (windows, widths, np.asarray(means), np.asarray(sds))
    norms = (Normalize(0, 100), Normalize(0, max(5, 5*np.ceil(max(spreads)/5))))
    cmaps = (red_map, gray_map)
    fig = plt.figure(figsize=(6.4, 7.35), layout="constrained")
    gs = fig.add_gridspec(4, 2, height_ratios=(1, 1, 1, .085), hspace=.16, wspace=.09)
    meshes = []
    for i, dataset in enumerate(DATASETS):
        windows, widths, means, sds = parts[dataset]
        for j, values in enumerate((means, sds)):
            ax = fig.add_subplot(gs[i, j])
            mesh = ax.pcolormesh(np.arange(len(windows)+1), np.arange(len(widths)+1), values,
                                 norm=norms[j], cmap=cmaps[j], edgecolors="white",
                                 linewidth=.6, rasterized=False)
            if i == 0:
                meshes.append(mesh)
            ax.set_xticks(np.arange(len(windows))+.5, windows)
            ax.set_yticks(np.arange(len(widths))+.5, widths)
            ax.set_xlabel("Window W")
            ax.set_ylabel("FF width")
            ax.set_title(dataset, loc="left", pad=5)
            ax.tick_params(length=0, pad=3)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for y in range(len(widths)):
                for x in range(len(windows)):
                    rgba = cmaps[j](norms[j](values[y, x]))
                    luminance = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                    ax.text(x+.5, y+.5, f"{values[y, x]:.1f}", ha="center", va="center",
                            fontsize=7.5, color="white" if luminance < .45 else "#171717")
            # Largest W and FF are the reference geometry for each dataset.
            ax.add_patch(Rectangle((len(windows)-1, len(widths)-1), 1, 1,
                                   fill=False, edgecolor="#111111", linewidth=1.25))
    for j, label in enumerate(("Mean BF200 F1 (%)", "Sample SD (pp)")):
        cbax = fig.add_subplot(gs[3, j])
        cb = fig.colorbar(meshes[j], cax=cbax, orientation="horizontal")
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.outline.set_linewidth(.5)
        if cb.solids is not None:
            cb.solids.set_rasterized(False)
            cb.solids.set_edgecolor("face")
    return save(fig, output, "grid_bf200")


def slices_figure(rows: list[dict], output: Path) -> list[str]:
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 7.3))
    fig.subplots_adjust(left=.10, right=.985, top=.945, bottom=.11, wspace=.25, hspace=.42)
    handles = None
    for i, dataset in enumerate(DATASETS):
        part = [r for r in rows if r["dataset"] == dataset]
        max_w = max(r["window"] for r in part)
        max_h = max(r["feedforward"] for r in part)
        slices = (("window", [r for r in part if r["feedforward"] == max_h]),
                  ("feedforward", [r for r in part if r["window"] == max_w]))
        for j, (key, selected) in enumerate(slices):
            ax = axes[i, j]
            xs = sorted({r[key] for r in selected})
            series = []
            for metric, color, style, marker, label in (
                    ("bf_f1", RED, "-", "o", "BF200"),
                    ("pot_f1", GRAY, "--", "s", "POT")):
                samples = [[100*r[metric] for r in selected if r[key] == x] for x in xs]
                means = np.asarray([statistics.mean(v) for v in samples])
                low = means-np.asarray([min(v) for v in samples])
                high = np.asarray([max(v) for v in samples])-means
                artist = ax.errorbar(xs, means, yerr=np.vstack((low, high)), color=color,
                                     linestyle=style, marker=marker, markersize=3.6,
                                     markerfacecolor="white", markeredgewidth=.9,
                                     linewidth=1.2, elinewidth=.8, capsize=2.1,
                                     capthick=.8, label=label, zorder=3)
                series.append(artist)
            if handles is None:
                handles = series
            ax.set_title(dataset, loc="left", pad=4)
            ax.set_xticks(xs)
            ax.set_ylim(20, 100)
            ax.set_yticks((20, 40, 60, 80, 100))
            ax.set_xlim(min(xs)-.05*(max(xs)-min(xs)), max(xs)+.05*(max(xs)-min(xs)))
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#dedede", linewidth=.6)
            ax.spines[["top", "right"]].set_visible(False)
            for name in ("left", "bottom"):
                ax.spines[name].set_color("#777777")
                ax.spines[name].set_linewidth(.65)
            ax.tick_params(length=2.5, width=.65)
            if j == 0:
                ax.set_ylabel("F1 (%)")
            if i == 2:
                ax.set_xlabel("Window W" if j == 0 else "FF width")
    fig.text(.30, .982, "Window sweep", ha="center", va="top", fontsize=9)
    fig.text(.795, .982, "Feed-forward sweep", ha="center", va="top", fontsize=9)
    fig.legend(handles, ("BF200", "POT"), loc="lower center", bbox_to_anchor=(.545, .025),
               ncol=2, frameon=False, handlelength=2.6, columnspacing=2.3)
    return save(fig, output, "slices_bf200_pot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=THESIS / "data/chapter5/runs.csv")
    parser.add_argument("--output", type=Path, default=THESIS / "figures/chapter5")
    args = parser.parse_args()
    rows = read_runs(args.input)
    settings()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = grid_figure(rows, args.output) + slices_figure(rows, args.output)
    print(json.dumps({"input": str(args.input), "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                      "runs": len(rows), "cells": 90, "seeds_per_cell": 3,
                      "grid_statistic": "mean and sample standard deviation in separate panels",
                      "slice_statistic": "mean; error bars span recorded minimum to maximum",
                      "baseline_outline": "largest W and FF in each dataset",
                      "slice_y_limits_percent": [20, 100], "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
