import argparse
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from TranADPlus.scripts import training_loop as native


ORIGINAL_SAVE = torch.save


def save_atomic(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            ORIGINAL_SAVE(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def random_state():
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        value["mps"] = torch.mps.get_rng_state()
    return value


def restore_random_state(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if "cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in value["cuda"]])
    if "mps" in value and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.set_rng_state(value["mps"].cpu())


def parse_arguments():
    parser = argparse.ArgumentParser(description="TranAD+ resumable training")
    parser.add_argument("--model_str", default="TranAD")
    parser.add_argument("--dataset_str", required=True)
    parser.add_argument("--identifier", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scaling", default="True")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_sz", type=int)
    parser.add_argument("--window_sz", type=int)
    parser.add_argument("--hidden_sz", type=int)
    parser.add_argument("--dim_feedforward")
    parser.add_argument("--pot_q", type=float)
    parser.add_argument("--shuffle", type=int)
    parser.add_argument("--padding", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--downsample", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--stop-after-epoch", type=int)
    return parser, parser.parse_args()


def main():
    parser, args = parse_arguments()
    if args.model_str != "TranAD":
        parser.error("This runner supports TranAD only")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be at least 1")

    base = Path(native.global_config.config.top_dir)
    final_path = base / "Checkpoints" / f"{args.identifier}.ckpt"
    resume_path = base / "Checkpoints" / f"{args.identifier}.resume.ckpt"

    def train(model, epochs, dataloader, optimizer, scheduler, device, save_location=None):
        if save_location is None or Path(save_location) != final_path:
            raise RuntimeError("Unexpected training checkpoint destination")
        if final_path.exists():
            raise RuntimeError(
                f"Existing original or final checkpoint: {final_path}. "
                "Use resume_native.py for original intermediate checkpoints."
            )

        folder = base / "Datasets" / "Preprocessed" / args.dataset_str / "Train" / "Tracks"
        inventory = [
            (name, (folder / name).stat().st_size)
            for name in os.listdir(folder)
            if name.endswith(".npy")
        ]
        configuration = {
            "dataset": args.dataset_str,
            "identifier": args.identifier,
            "epochs": int(epochs),
            "features": int(model.n_feats),
            "window_size": int(model.window_sz),
            "batch_size": int(dataloader.batch_size),
            "feedforward": int(model.transformer_encoder.layers[0].linear1.out_features),
            "encoder_layers": int(model.num_encoder_layers),
            "decoder_layers": int(model.num_decoder_layers),
            "initial_lr": float(optimizer.defaults["lr"]),
            "padding": bool(dataloader.dataset.padding),
            "shuffle": bool(args.shuffle) if args.shuffle is not None else False,
            "num_workers": int(dataloader.num_workers),
            "downsample": args.downsample,
            "scaling": str(args.scaling),
            "track_inventory_ordered": inventory,
        }
        start_epoch = 0
        final_loss = None

        if resume_path.exists():
            saved = torch.load(resume_path, map_location=device, weights_only=False)
            needed = {
                "resume_schema", "configuration", "epoch", "model_state_dict",
                "optimizer_state_dict", "scheduler_state_dict", "rng_state"
            }
            if not isinstance(saved, dict) or not needed.issubset(saved):
                raise RuntimeError(f"Invalid resumable checkpoint: {resume_path}")
            if saved["resume_schema"] != 1 or saved["configuration"] != configuration:
                raise RuntimeError(
                    "Checkpoint configuration differs from this run. "
                    "Do not overwrite the existing checkpoint."
                )
            start_epoch = int(saved["epoch"]) + 1
            if not 0 <= start_epoch <= epochs:
                raise RuntimeError("Resume epoch is out of range")
            model.load_state_dict(saved["model_state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            scheduler.load_state_dict(saved["scheduler_state_dict"])
            if scheduler.last_epoch != start_epoch:
                raise RuntimeError(
                    f"Scheduler epoch mismatch: {scheduler.last_epoch} vs {start_epoch}"
                )
            restore_random_state(saved["rng_state"])
            final_loss = saved.get("last_loss")
            print(f"RESUME: {args.identifier}, epoch {start_epoch + 1}/{epochs}", flush=True)
        else:
            print(f"NEW TRAINING: {args.identifier}", flush=True)

        loss_fn = nn.MSELoss(reduction="none")
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_losses = []
            for windows in dataloader:
                windows = windows.to(device).permute(1, 0, 2)
                target = windows[-1:, :, :]
                first, second = model(windows, target)
                weight = 1.0 / (epoch + 1)
                loss = (
                    weight * loss_fn(first, target)
                    + (1.0 - weight) * loss_fn(second, target)
                ).mean()
                epoch_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            if not epoch_losses:
                raise RuntimeError("Training dataloader produced no batches")
            final_loss = float(np.mean(epoch_losses))
            if not np.isfinite(final_loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            scheduler.step()
            print(f"Epoch {epoch} avg. loss: {final_loss:.4f}", flush=True)
            completed = epoch + 1
            stop = args.stop_after_epoch is not None and completed >= args.stop_after_epoch
            if completed % args.checkpoint_every == 0 or completed == epochs or stop:
                save_atomic({
                    "resume_schema": 1,
                    "configuration": configuration,
                    "epoch": epoch,
                    "last_loss": final_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": random_state(),
                }, resume_path)
                print(f"Saved recovery checkpoint: {completed}/{epochs}", flush=True)
            if stop:
                print("Test interruption requested; checkpoint preserved", flush=True)
                raise SystemExit(75)

        return final_loss, optimizer.param_groups[0]["lr"]

    original_train = native.helpers.train_TranAD

    def protected_save(payload, destination, *args_save, **kwargs_save):
        if isinstance(destination, (str, os.PathLike)):
            if Path(destination) == final_path and isinstance(payload, dict) and "hp_dict" in payload:
                save_atomic(payload, final_path)
                return
        return ORIGINAL_SAVE(payload, destination, *args_save, **kwargs_save)

    try:
        native.helpers.train_TranAD = train
        torch.save = protected_save
        native.main(args)
    finally:
        native.helpers.train_TranAD = original_train
        torch.save = ORIGINAL_SAVE

    if not final_path.is_file():
        raise RuntimeError("Training returned without a final checkpoint")
    print(f"Final native checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
