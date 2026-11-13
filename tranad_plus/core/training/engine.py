"""Ordered FP32/Adam training, durable epoch recovery and strict resume checks."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch

from ..bootstrap import activate_baseline
from ..common import append_json, save_torch
from ..models import build_model
from .objectives import reconstruction_loss
activate_baseline()
from gpu_batches import training_batches
from console_telemetry import emit_epoch


class TrainingInterrupted(Exception):
    """An intentional interruption after a durable checkpoint (CLI exit 75)."""


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    n = np.random.get_state()
    state = {"python": random.getstate(), "torch": torch.get_rng_state(),
             "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4])}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def load_checkpoint(path, profile, signature, *, final=False):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("schema_version") != 1 or saved.get("signature") != signature
            or saved.get("profile") != profile):
        raise RuntimeError(f"Checkpoint configuration/source/data mismatch: {path}")
    if not 0 <= saved["epoch"] < profile["epochs"]:
        raise RuntimeError("Invalid checkpoint epoch")
    if final and (saved.get("kind") != "final" or saved["epoch"] != profile["epochs"] - 1):
        raise RuntimeError("Expected a completed final checkpoint")
    return saved


def train(profile, bundle, folder, signature, device, *, stop_after_epoch=None):
    folder = Path(folder)
    final = folder / "checkpoints/final.ckpt"
    recovery = folder / "checkpoints/resume.ckpt"
    if final.exists():
        load_checkpoint(final, profile, signature, final=True)
        print(f"TRAIN SKIP verified final checkpoint: {profile['identifier']}", flush=True)
        return final
    seed_all(profile["seed"])
    batches, transfer = training_batches(bundle.datasets["train"], batch_size=profile["batch_sz"],
                                        shuffle=False, device=device, num_workers=0)
    model = build_model(profile).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=profile["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)
    start_epoch, final_loss = 0, None
    if recovery.exists():
        saved = load_checkpoint(recovery, profile, signature)
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        restore_rng(saved["rng_state"])
        start_epoch = saved["epoch"] + 1
        final_loss = saved["final_loss"]
        if scheduler.last_epoch != start_epoch:
            raise RuntimeError("Recovery scheduler/epoch mismatch")
        print(f"TRAIN RESUME {profile['identifier']} epoch {start_epoch + 1}/{profile['epochs']}", flush=True)
    cuda = torch.device(device).type == "cuda"
    for epoch in range(start_epoch, profile["epochs"]):
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model.train()
        total = torch.zeros((), device=device, dtype=torch.float64)
        count = 0
        for batch in batches:
            windows = batch[0] if isinstance(batch, (tuple, list)) else batch
            windows = windows.to(device, non_blocking=(transfer == "cpu_pinned")).permute(1, 0, 2)
            target = windows[-1:, :, :]
            loss = reconstruction_loss(model(windows, target), target, epoch + 1,
                                       implementation="square" if profile["dataset"] == "OPS-SAT" else "mse_loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total.add_(loss.detach().double())
            count += 1
        if not count:
            raise RuntimeError("No training batches")
        final_loss = float((total / count).item())
        if not np.isfinite(final_loss):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}; recovery preserved")
        scheduler.step()
        if cuda:
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        record = {"epoch_one_based": epoch + 1, "loss_mean_of_batch_means": final_loss,
                  "learning_rate_after_scheduler_step": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds": seconds, "batches": count,
                  "dataset_windows": len(bundle.datasets["train"]),
                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None}
        append_json(folder / "epochs.jsonl", record)
        save_torch(recovery, {"schema_version": 1, "kind": "resume", "signature": signature,
                   "profile": profile, "epoch": epoch, "final_loss": final_loss,
                   "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "scheduler_state": scheduler.state_dict(), "rng_state": rng_state()})
        emit_epoch(dataset=profile["dataset"], identifier=profile["identifier"],
                   epoch=epoch + 1, epochs=profile["epochs"], loss=final_loss,
                   lr=optimizer.param_groups[0]["lr"], seconds=seconds, batches=count,
                   windows=len(bundle.datasets["train"]),
                   peak_allocated_bytes=record["gpu_peak_allocated_bytes"])
        if stop_after_epoch and epoch + 1 >= stop_after_epoch:
            raise TrainingInterrupted(f"Recovery saved after epoch {epoch + 1}")
    save_torch(final, {"schema_version": 1, "kind": "final", "signature": signature,
               "profile": profile, "epoch": profile["epochs"] - 1, "final_loss": final_loss,
               "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "scheduler_state": scheduler.state_dict(), "data_inventory": bundle.inventory})
    print(f"TRAIN COMPLETE {final}", flush=True)
    return final
