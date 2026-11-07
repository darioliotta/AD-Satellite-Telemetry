import argparse
import os
import random
import tempfile
import time
from epoch_telemetry import write_epoch
from console_telemetry import emit_epoch
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
    parser.add_argument('--seed', type=int, default=None)
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
    from gpu_settings import configure
    configure()
    parser, args = parse_arguments()
    if args.model_str != "TranAD":
        parser.error("This runner supports TranAD only")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be at least 1")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    native_cfg = native.global_config.config
    final_path = Path(native_cfg.checkpoint_dir) / f"{args.identifier}.ckpt"
    resume_path = Path(native_cfg.checkpoint_dir) / f"{args.identifier}.resume.ckpt"

    def train(model, epochs, dataloader, optimizer, scheduler, device, save_location=None):
        if save_location is None or Path(save_location) != final_path:
            raise RuntimeError("Unexpected training checkpoint destination")
        if final_path.exists():
            raise RuntimeError(
                f"Existing original or final checkpoint: {final_path}. "
                "Use resume_native.py for original intermediate checkpoints."
            )

        folder = Path(native_cfg.preprocess_save_dir) / args.dataset_str / "Train" / "Tracks"
        inventory = [
            (name, (folder / name).stat().st_size)
            for name in sorted(os.listdir(folder))
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
            "seed": args.seed,
            "adam_impl": os.environ.get("TRANADPLUS_ADAM_IMPL", "standard"),
            "precision": os.environ.get("TRANADPLUS_CUDA_PRECISION", "faithful"),
            "track_inventory_ordered": inventory,
        }
        from gpu_settings import select_optimizer
        upstream_optimizer = optimizer
        optimizer, scheduler = select_optimizer(model, optimizer, scheduler)
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

        from gpu_batches import training_batches
        batches, transfer_path = training_batches(dataloader.dataset, batch_size=dataloader.batch_size,
            shuffle=bool(args.shuffle), device=device, num_workers=dataloader.num_workers)
        loss_fn = nn.MSELoss(reduction="none")
        for epoch in range(start_epoch, epochs):
            if torch.device(device).type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.perf_counter()
            model.train()
            batch_count=0
            loss_total=torch.zeros((),device=device,dtype=torch.float64)
            for windows in batches:
                windows = windows.to(device, non_blocking=(transfer_path == 'cpu_pinned')).permute(1, 0, 2)
                target = windows[-1:, :, :]
                first, second = model(windows, target)
                weight = 1.0 / (epoch + 1)
                loss = (
                    weight * loss_fn(first, target)
                    + (1.0 - weight) * loss_fn(second, target)
                ).mean()
                loss_total.add_(loss.detach().double())
                batch_count += 1
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if not batch_count:
                raise RuntimeError("Training dataloader produced no batches")
            final_loss = float((loss_total / batch_count).item())
            if not np.isfinite(final_loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            scheduler.step()
            cuda = torch.device(device).type == 'cuda'
            if cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - epoch_start
            write_epoch(args.dataset_str, args.identifier, epoch+1, final_loss,
                        optimizer.param_groups[0]['lr'], elapsed, batch_count,
                        len(dataloader.dataset), device,
                        torch.cuda.max_memory_allocated(device) if cuda else None,
                        torch.cuda.max_memory_reserved(device) if cuda else None)
            emit_epoch(dataset=args.dataset_str, identifier=args.identifier,
                       epoch=epoch+1, epochs=epochs, loss=final_loss,
                       lr=optimizer.param_groups[0]['lr'], seconds=elapsed,
                       batches=batch_count, windows=len(dataloader.dataset),
                       peak_allocated_bytes=(torch.cuda.max_memory_allocated(device) if cuda else None))
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

        # The original upstream training_loop.py saves `optimizer` from its
        # own local scope after this function returns. In fused mode we have
        # trained with a replacement optimizer: copy its complete state back
        # so the final upstream checkpoint contains the ACTUAL trained Adam
        # moments/step counters instead of an empty original optimizer.
        from gpu_settings import sync_upstream_optimizer
        sync_upstream_optimizer(upstream_optimizer, optimizer)
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
