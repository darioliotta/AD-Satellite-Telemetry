"""Fork a verified recovery state and continue the unchanged trainer."""
import json
from pathlib import Path
import torch

from tranad_plus.core.common import hash_file, save_torch, write_json, atomic_destination
from tranad_plus.core.training.engine import train, TrainingInterrupted, load_checkpoint
from tranad_plus.core.evaluation.engine import evaluate
from .metrics import ranking, raw_best_f1
from .progress import training as training_progress, evaluation as evaluation_progress


def validate_recovery(parent, parent_profile, parent_signature):
    """Read-only final/recovery agreement check, also used by preflight."""
    saved = load_checkpoint(parent / "checkpoints/resume.ckpt", parent_profile, parent_signature)
    final = load_checkpoint(parent / "checkpoints/final.ckpt", parent_profile, parent_signature, final=True)
    if saved["epoch"] != parent_profile["epochs"] - 1 or "rng_state" not in saved:
        raise RuntimeError("Parent recovery must be at the final epoch and include RNG state")
    for key, value in final["model_state"].items():
        torch.testing.assert_close(saved["model_state"][key], value, rtol=0, atol=0)
    def same_state(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
        return a == b
    if not same_state(saved["optimizer_state"], final["optimizer_state"]):
        raise RuntimeError("Parent final/recovery optimizer mismatch")
    if saved["scheduler_state"] != final["scheduler_state"]:
        raise RuntimeError("Parent final/recovery scheduler mismatch")
    if saved["scheduler_state"]["last_epoch"] != parent_profile["epochs"]:
        raise RuntimeError("Parent scheduler is not at the final training epoch")
    history = {}
    for line in (parent / "epochs.jsonl").read_text().splitlines():
        row = json.loads(line)
        history[row["epoch_one_based"]] = row
    if sorted(history) != list(range(1, parent_profile["epochs"] + 1)):
        raise RuntimeError("Incomplete parent epoch history")
    return saved, history


def initialize(parent, folder, parent_profile, parent_signature, signature, extra_epochs):
    """Copy state into a new namespace, preserving optimizer/scheduler/RNG."""
    target = parent_profile['epochs'] + extra_epochs
    identifier = 'toast14_' + parent_profile['identifier'].removeprefix('toast13_') + f'_e{target}'
    profile = {**parent_profile, "epochs": target, "identifier": identifier}
    saved, history = validate_recovery(parent, parent_profile, parent_signature)
    resume = folder / "checkpoints/resume.ckpt"
    if not resume.exists() and not (folder / "checkpoints/final.ckpt").exists():
        # Epoch log first: an interrupted initialization can be safely repeated.
        with atomic_destination(folder / "epochs.jsonl") as stream:
            for epoch in sorted(history):
                stream.write(json.dumps(history[epoch]) + "\n")
        imported = {**saved, "profile": profile, "signature": signature,
                    "continuation_parent": {"path": str(parent), "recovery_sha256": hash_file(parent / "checkpoints/resume.ckpt")}}
        save_torch(resume, imported)
    return profile


def run_continuation(parent, folder, parent_report, signature, bundle, device, settings,
                     extra_epochs=50, stop_after_epoch=None):
    profile = initialize(parent, folder, parent_report["profile"], parent_report["signature"],
                         signature, extra_epochs)
    write_json(folder / "config.json", profile)
    # Full original loss weights keep the absolute epoch index: 1/51 ... 1/100.
    # Adam moments, StepLR and all RNG state come from the recovery checkpoint.
    with training_progress(folder, parent_report["profile"]["epochs"], profile["epochs"], profile["dataset"]):
        train(profile, bundle, folder, signature, device, stop_after_epoch=stop_after_epoch)
    with evaluation_progress(folder, 21 if profile["variant"] == "reference" else 1):
        result = evaluate(profile, bundle, folder, signature, device, settings)
    import numpy as np
    comparison = {"parent_epochs": parent_report["profile"]["epochs"], "final_epochs": profile["epochs"],
                  "parent_identifier": parent_report["identifier"], "steps": {}}
    for label, report, path in (("before", parent_report, parent / "scores.npz"),
                                ("after", result, folder / "scores.npz")):
        with np.load(path, allow_pickle=False) as scores:
            i = int(np.flatnonzero(scores["alpha_values"] == .5)[0]) if len(scores["alpha_values"]) else 0
            raw = ranking(scores["test_labels"], scores["test_scores"][i])
            raw["best_f1_fixed_alpha_test_oracle"] = raw_best_f1(scores["test_labels"], scores["test_scores"][i])["f1"]
        comparison["steps"][label] = {
            "paper_oracle": {k: report["paper_oracle"][k] for k in ("pot", "bf", "bf_dense")},
            "raw_fixed_alpha": raw, "validation_selected": report["validation_selected"]}
    write_json(folder / "comparison.json", comparison)
    return result
