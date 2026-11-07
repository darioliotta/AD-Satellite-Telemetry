"""Predeclared matched comparisons; pilot jobs are reusable in the main campaign."""
from dataclasses import dataclass
import itertools
import json
from pathlib import Path

from .bootstrap import ROOT, PROJECT_ROOT

VARIANTS = ("reference", "shared_decoder", "single_pass", "encoder_only")
DATASETS = ("OPS-SAT", "SMAP", "MSL")
CAMPAIGNS = ("pilot", "main", "compact", "encoder_only", "all", "additional")
CONFIGURATIONS = ("reference", "compact", "window_only", "ff_only")


@dataclass(frozen=True)
class Experiment:
    profile: dict

    @property
    def identifier(self):
        return self.profile["identifier"]

    def folder(self, output_root):
        p = self.profile
        return (Path(output_root) / "v13" / "architectures" / p["dataset"] /
                p["variant"] / p["configuration"] / f"seed_{p['seed']}")


def load_campaign(campaign="pilot", plan_path=None):
    path = Path(plan_path) if plan_path else ROOT / "configs/campaigns" / f"{campaign}.json"
    plan = json.loads(path.read_text())
    base = json.loads((ROOT / "configs/baselines.json").read_text())
    evaluation = base["evaluation"]
    if (evaluation["alpha_steps"] != 21 or evaluation["paper_threshold_steps"] != 200
            or evaluation["dense_thresholds"] != "all_distinct_scores"
            or evaluation["pot_max_initializations"] != 35):
        raise ValueError("Changing the pinned evaluation protocol requires an explicit new implementation/study")
    if (evaluation["warmup"] < 0 or evaluation["iterations"] < 1
            or not 0 < evaluation["train_quantile"] < 1
            or any(type(v) is not int or v < 1 for v in evaluation["score_batch_size_by_dataset"].values())):
        raise ValueError("Invalid evaluation measurement settings")
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported campaign schema")
    if not set(plan["datasets"]).issubset(DATASETS) or not plan["datasets"]:
        raise ValueError("Unsupported/empty datasets")
    if not set(plan["variants"]).issubset(VARIANTS) or not plan["variants"]:
        raise ValueError("Unsupported/empty variants")
    # Blocks allow the extension to add two seeds to the old settings while
    # giving the new controls all five seeds, without scheduling the old 72.
    blocks = plan.get("blocks")
    if blocks is None:
        blocks = [{"configurations": plan["configurations"], "seeds": plan["seeds"]}]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Campaign blocks must be a nonempty list")
    for block in blocks:
        if (not block["configurations"] or
                not set(block["configurations"]).issubset(CONFIGURATIONS)):
            raise ValueError("Unsupported/empty configuration")
        if not block["seeds"] or any(type(s) is not int or s < 0 for s in block["seeds"]):
            raise ValueError("Seeds must be nonnegative integers")
    if type(plan.get("dataset_first", False)) is not bool:
        raise ValueError("dataset_first must be a boolean")
    rows = []
    combinations = itertools.chain.from_iterable(
        itertools.product(block["configurations"], plan["datasets"], block["seeds"], plan["variants"])
        for block in blocks)
    for setting, dataset, seed, variant in combinations:
        info = base["datasets"][dataset]
        profile = {**base["training"], **info["training"], **info["configurations"][setting],
                   "dataset": dataset, "features": info["features"], "variant": variant,
                   "configuration": setting, "seed": seed,
                   "identifier": f"toast13_{dataset.lower().replace('-', '')}_{variant}_{setting}_s{seed}"}
        for key in ("epochs", "batch_sz", "window_sz", "dim_feedforward", "num_layers", "features"):
            if type(profile[key]) is not int or profile[key] < 1:
                raise ValueError(f"{key} must be a positive effective integer")
        if profile["shuffle"] or not profile["padding"] or profile["downsample"] is not None:
            raise ValueError("These campaigns preserve ordered batches, padding and no downsampling")
        if profile["lr"] <= 0 or not 0 < profile["pot_q"] < 1:
            raise ValueError("Invalid LR/POT risk")
        rows.append(Experiment(profile))
    if len({r.identifier for r in rows}) != len(rows):
        raise ValueError("Duplicate experiments in campaign")
    if plan.get("dataset_first", False):
        order = {dataset: index for index, dataset in enumerate(plan["datasets"])}
        rows.sort(key=lambda experiment: order[experiment.profile["dataset"]])
    return plan, rows, base["evaluation"]


def validate_output_root(output_root, data_root):
    value = Path(output_root).expanduser().resolve()
    protected = (ROOT, PROJECT_ROOT / "tranad_plus", Path(data_root).expanduser().resolve())
    if any(value == p or p in value.parents for p in protected):
        raise ValueError("Results must live outside model source and data directories")
    return value
