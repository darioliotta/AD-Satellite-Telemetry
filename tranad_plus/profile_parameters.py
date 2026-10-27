from collections import defaultdict
from pathlib import Path
import csv
import sys

import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "TranADPlus"))

from src.models import TranAD

configurations = [
    ("OPS-SAT-3F-H8", 3, 8, 20),
    ("OPS-SAT-3F-H16", 3, 16, 20),
    ("OPS-SAT-3F-H32", 3, 32, 20),
    ("OPS-SAT-3F-H64", 3, 64, 20),
    ("SMAP-25F-H50", 25, 50, 10),
    ("SMD-38F-H16", 38, 16, 20),
]

results = []

for name, features, hidden, window in configurations:
    model = TranAD(
        n_feats=features,
        window_sz=window,
        batch_sz=64,
        dim_feedforward=hidden,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )

    total = sum(p.numel() for p in model.parameters())

    theoretical = (
        82 * features**2
        + 12 * features * hidden
        + 47 * features
        + 3 * hidden
    )

    components = defaultdict(int)
    categories = defaultdict(int)

    for parameter_name, parameter in model.named_parameters():
        count = parameter.numel()
        component = parameter_name.split(".")[0]
        components[component] += count

        if "self_attn" in parameter_name or "multihead_attn" in parameter_name:
            category = "attention"
        elif "linear1." in parameter_name or "linear2." in parameter_name:
            category = "feedforward"
        elif "norm" in parameter_name:
            category = "normalization"
        elif parameter_name.startswith("fcn."):
            category = "output"
        else:
            category = "other"

        categories[category] += count

    print(f"\n{name}")
    print(f"Features: {features}")
    print(f"Feed-forward width: {hidden}")
    print(f"PyTorch parameters: {total:,}")
    print(f"Theoretical parameters: {theoretical:,}")
    print(f"Formula matches: {total == theoretical}")
    print(f"FP32 weight memory: {total * 4 / 1024:.2f} KiB")

    print("Components:")
    for component, count in sorted(components.items()):
        print(f"  {component}: {count:,} ({count / total * 100:.1f}%)")

    print("Parameter categories:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count:,} ({count / total * 100:.1f}%)")

    row = {
        "model": name,
        "features": features,
        "feedforward": hidden,
        "window": window,
        "parameters": total,
        "theoretical": theoretical,
        "formula_matches": total == theoretical,
        "fp32_kib": total * 4 / 1024,
        **{f"component_{k}": v for k, v in components.items()},
        **{f"category_{k}": v for k, v in categories.items()},
    }
    results.append(row)

output = root / "parameter_report.csv"
fieldnames = sorted(set().union(*(row.keys() for row in results)))

with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nReport saved to: {output}")
