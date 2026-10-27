
import argparse
import os
import shutil
from uuid import uuid4

from config import (
    DATASETS,
    NATIVE_PACKAGE,
    PROFILES,
    checkpoint_path,
    get_profile,
    result_directory,
)
from run_native import check_device, native_command, run_and_log


def archive_checkpoint(dataset, identifier):
    source = checkpoint_path(identifier)

    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(
            f"Training checkpoint missing or empty: {source}"
        )

    directory = result_directory(dataset)
    directory.mkdir(parents=True, exist_ok=True)

    destination = directory / f"{identifier}.ckpt"

    if destination.exists():
        raise FileExistsError(
            f"Archived checkpoint already exists: {destination}"
        )

    temporary = directory / (
        f".{identifier}_{uuid4().hex}.tmp"
    )

    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return destination


def validate_training_data(dataset):
    directory = (
        NATIVE_PACKAGE
        / "Datasets"
        / "Preprocessed"
        / dataset
        / "Train"
        / "Tracks"
    )

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Preprocessed training data not found: {directory}"
        )

    tracks = list(directory.glob("*.npy"))

    if not tracks:
        raise RuntimeError(
            f"No training tracks found: {directory}"
        )

    if dataset == "SMAP" and len(tracks) != 54:
        raise RuntimeError(
            f"Expected 54 SMAP tracks, found {len(tracks)}."
        )

    return len(tracks)


def build_command(profile, device):
    options = [
        "--model_str", "TranAD",
        "--dataset_str", profile["dataset"],
        "--identifier", profile["identifier"],
        "--device", device,
        "--scaling", "True",
        "--lr", str(profile["lr"]),
        "--epochs", str(profile["epochs"]),
        "--batch_sz", str(profile["batch_sz"]),
        "--window_sz", str(profile["window_sz"]),
        "--dim_feedforward", str(
            profile["dim_feedforward"]
        ),
        "--num_layers", str(profile["num_layers"]),
        "--pot_q", str(profile["pot_q"]),
        "--shuffle", (
            "1" if profile["shuffle"] else "0"
        ),
        "--padding", (
            "1" if profile["padding"] else "0"
        ),
        "--num_workers", str(profile["num_workers"]),
    ]

    if profile["downsample"] is not None:
        options.extend([
            "--downsample",
            str(profile["downsample"]),
        ])

    return native_command("training_loop.py", options)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda:0"),
        default="cpu",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        profile = get_profile(args.profile)
    except ValueError as exc:
        parser.error(str(exc))

    dataset = profile["dataset"]
    identifier = profile["identifier"]
    checkpoint = checkpoint_path(identifier)
    archived = (
        result_directory(dataset)
        / f"{identifier}.ckpt"
    )

    command = build_command(profile, args.device)

    print(f"Dataset: {dataset}", flush=True)
    print(f"Profile: {identifier}", flush=True)
    print(f"Epochs: {profile['epochs']}", flush=True)
    print(
        f"Window size: {profile['window_sz']}",
        flush=True,
    )
    print(f"Device: {args.device}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)

    if args.dry_run:
        print("Dry run: " + " ".join(command))
        return

    check_device(args.device, parser)

    if checkpoint.exists() or checkpoint.is_symlink():
        parser.error(
            f"Checkpoint already exists: {checkpoint}"
        )

    if archived.exists():
        parser.error(
            f"Archived checkpoint already exists: {archived}"
        )

    tracks = validate_training_data(dataset)
    print(f"Training tracks: {tracks}", flush=True)

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_and_log(
        command,
        dataset,
        identifier,
        "train",
    )

    archived = archive_checkpoint(dataset, identifier)

    print(
        f"Checkpoint archived: {archived}",
        flush=True,
    )


if __name__ == "__main__":
    main()
