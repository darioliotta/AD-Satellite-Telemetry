"""Standalone PNG/PDF figures from seed aggregates; never pool devices silently."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tranad_variants.bootstrap import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results/v13/architectures")
    parser.add_argument("--metric", choices=("pot_f1", "bf_f1", "bf_dense_f1", "validation_selected_test_fragment_auc"), default="bf_dense_f1")
    parser.add_argument("--cost", choices=("parameters", "latency_median_ms"), default="parameters")
    parser.add_argument("--execution", help="Execution SHA from aggregate.json, required when several environments are present")
    args = parser.parse_args()
    rows = json.loads((args.results_root / "aggregate.json").read_text())["groups"]
    identities = {r["execution_sha256"] for r in rows}
    if len(identities) > 1 and args.execution is None:
        raise ValueError("Multiple execution environments: select one with --execution")
    rows = [r for r in rows if args.execution is None or r["execution_sha256"] == args.execution]
    if not rows:
        raise ValueError("No completed results for this execution")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = args.results_root / "figures"
    destination.mkdir(exist_ok=True)
    colors = {"reference": "#444444", "shared_decoder": "#0072B2", "single_pass": "#D55E00", "encoder_only": "#009E73"}
    for dataset in sorted({r["dataset"] for r in rows}):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        used = set()
        for r in rows:
            if r["dataset"] != dataset or r[args.metric]["mean"] is None:
                continue
            x, y = r[args.cost], r[args.metric]
            label = r["variant"] if r["variant"] not in used else None
            used.add(r["variant"])
            ax.errorbar(x["mean"], y["mean"], xerr=x["sample_std"], yerr=y["sample_std"],
                        fmt="o" if r["configuration"] == "reference" else "s",
                        color=colors[r["variant"]], label=label, capsize=3)
            ax.annotate(f"{r['configuration']} (n={y['n']})", (x["mean"], y["mean"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Trainable parameters" if args.cost == "parameters" else "Batch-one forward median (ms)",
               ylabel=args.metric, title=f"{dataset}: architecture trade-offs")
        ax.grid(alpha=.25)
        if used:
            ax.legend(fontsize=8)
        fig.text(.02, .01, "Error bars: sample SD across seeds when n > 1; oracle metrics use test labels.", fontsize=7)
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"{dataset}_{args.metric}_vs_{args.cost}.{extension}", dpi=200)
        plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
