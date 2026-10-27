import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import NATIVE_PACKAGE, RESULTS_DIR, checkpoint_path, get_profile


def save_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def checkpoint_info(path, profile):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
    if not isinstance(saved, dict) or not expected.issubset(saved):
        raise RuntimeError("Expected an intermediate native TranAD+ checkpoint with optimizer and scheduler states")
    epoch = saved["epoch"]
    if not isinstance(epoch, int) or not 0 <= epoch < profile["epochs"]:
        raise RuntimeError(f"Checkpoint epoch outside expected range: {epoch}")
    if not saved["optimizer_state_dict"].get("state"):
        raise RuntimeError("Checkpoint has an empty optimizer state")
    return saved


def build_metadata(profile, feature_count, final_loss, final_lr):
    from TranADPlus.src import helpers

    return {
        "dataset_str": profile["dataset"],
        "model_str": "TranAD",
        "identifier": profile["identifier"],
        "scaling": True,
        "features": feature_count,
        "lr": profile["lr"],
        "epochs": profile["epochs"],
        "batch_sz": profile["batch_sz"],
        "window_sz": profile["window_sz"],
        "shuffle": profile["shuffle"],
        "padding": profile["padding"],
        "num_workers": profile["num_workers"],
        "downsample": profile["downsample"],
        "pot_q": profile["pot_q"],
        "dim_feedforward": helpers.parse_dim_feedforward(feature_count, profile["dim_feedforward"]),
        "num_layers": profile["num_layers"],
        "final_loss": final_loss,
        "final_lr": final_lr,
    }


def resume(profile, device, saved, checkpoint):
    from TranADPlus.src import helpers, models
    from TranADPlus.src.datasets import TSDataset_tracks

    folder = NATIVE_PACKAGE / "Datasets" / "Preprocessed" / profile["dataset"] / "Train" / "Tracks"
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    track_paths = [str(folder / name) for name in os.listdir(folder) if name.endswith(".npy")]
    if not track_paths:
        raise RuntimeError(f"No training tracks in {folder}")
    dataset = TSDataset_tracks(
        track_paths=track_paths,
        window_size=profile["window_sz"],
        padding=profile["padding"],
        downsample=profile["downsample"],
        scaling=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=profile["batch_sz"],
        shuffle=profile["shuffle"],
        num_workers=profile["num_workers"],
    )
    feature_count = int(dataset.big_track.shape[1])
    parsed_feedforward = helpers.parse_dim_feedforward(feature_count, profile["dim_feedforward"])
    model = models.TranAD(
        n_feats=feature_count,
        dim_feedforward=parsed_feedforward,
        batch_sz=profile["batch_sz"],
        window_sz=profile["window_sz"],
        num_encoder_layers=profile["num_layers"],
        num_decoder_layers=profile["num_layers"],
    ).to(device).to(torch.float32)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=profile["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    scheduler.load_state_dict(saved["scheduler_state_dict"])
    last_epoch = saved["epoch"]
    start_epoch = last_epoch + 1
    if scheduler.last_epoch != start_epoch:
        raise RuntimeError(f"Scheduler epoch mismatch: {scheduler.last_epoch} vs {start_epoch}")

    backup = checkpoint.with_name(f"{checkpoint.stem}_origin_epoch_{start_epoch}.ckpt")
    if not backup.exists():
        shutil.copy2(checkpoint, backup)
    print(f"Original checkpoint preserved: {backup}", flush=True)
    print(f"Resuming {profile['identifier']} from epoch {start_epoch + 1}/{profile['epochs']}", flush=True)
    print(f"Dataset windows: {len(dataset)} | features: {feature_count} | device: {device}", flush=True)

    started = time.monotonic()
    final_loss = None
    model.train()
    for epoch in range(start_epoch, profile["epochs"]):
        epoch_losses = []
        for windows in dataloader:
            windows = windows.to(device).permute(1, 0, 2)
            element = windows[-1:, :, :]
            output1, output2 = model(windows, element)
            weight = 1.0 / (epoch + 1)
            loss = (weight * (output1 - element).square() + (1.0 - weight) * (output2 - element).square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if not epoch_losses:
            raise RuntimeError("No batches processed")
        final_loss = float(np.mean(epoch_losses))
        if not np.isfinite(final_loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
        scheduler.step()
        save_atomic(checkpoint, {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        })
        print(f"Epoch {epoch} avg. loss: {final_loss:.6f}", flush=True)

    if final_loss is None:
        final_loss = float("nan")
    metadata = build_metadata(profile, feature_count, final_loss, optimizer.param_groups[0]["lr"])
    final_checkpoint = {
        "hp_dict": metadata,
        "model_states": model.state_dict(),
        "optimizer_states": optimizer.state_dict(),
        "scheduler_states": scheduler.state_dict(),
    }
    save_atomic(checkpoint, final_checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    record = {
        "dataset": profile["dataset"],
        "identifier": profile["identifier"],
        "checkpoint_origin_epoch": last_epoch,
        "first_resumed_epoch": start_epoch,
        "final_epoch": profile["epochs"] - 1,
        "checkpoint_sha256": digest,
        "duration_seconds": round(time.monotonic() - started, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "warning": "RNG state was not stored in the original checkpoint; continuation is not bitwise-identical to uninterrupted training",
    }
    destination = RESULTS_DIR / profile["dataset"] / f"{profile['identifier']}_resume_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"Completed. Final native checkpoint: {checkpoint}", flush=True)
    print(f"Resume record: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Resume a native TranAD+ intermediate checkpoint")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    profile = get_profile(args.profile)
    checkpoint = checkpoint_path(profile["identifier"])
    saved = checkpoint_info(checkpoint, profile)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Completed epochs: {saved['epoch'] + 1}/{profile['epochs']}", flush=True)
    print(f"Next epoch: {saved['epoch'] + 2}/{profile['epochs']}", flush=True)
    if args.check_only:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Keep --check-only until a GPU is assigned")
    resume(profile, args.device, saved, checkpoint)


if __name__ == "__main__":
    main()
