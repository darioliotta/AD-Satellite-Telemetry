
import argparse
import math
import shutil

from config import (
    NATIVE_PACKAGE,
    checkpoint_path,
    get_profile,
    result_directory,
)
from run_native import check_device, native_command, run_and_log


def restore_checkpoint(dataset, identifier):
    native = checkpoint_path(identifier)

    if native.is_file():
        return native

    archived = (
        result_directory(dataset)
        / f"{identifier}.ckpt"
    )

    if not archived.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {native} or {archived}"
        )

    native.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archived, native)

    return native


def validate_checkpoint(path, profile):
    import torch

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid checkpoint format.")

    metadata = checkpoint.get("hp_dict")

    if not isinstance(metadata, dict):
        raise RuntimeError(
            "Checkpoint hyperparameters are missing."
        )

    if metadata.get("dataset_str") != profile["dataset"]:
        raise RuntimeError(
            "Checkpoint dataset does not match profile."
        )

    if "features" not in metadata:
        raise RuntimeError(
            "Checkpoint feature count is missing."
        )

    from TranADPlus.src import helpers

    feedforward = helpers.parse_dim_feedforward(
        metadata["features"],
        profile["dim_feedforward"],
    )

    expected = {
        "window_sz": profile["window_sz"],
        "dim_feedforward": feedforward,
        "num_layers": profile["num_layers"],
        "pot_q": profile["pot_q"],
    }

    for key, value in expected.items():
        actual = metadata.get(key)

        if (
            not isinstance(actual, (int, float))
            or not math.isclose(
                actual,
                value,
                rel_tol=1e-8,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"Checkpoint parameter mismatch: {key}. "
                f"Expected {value!r}, found {actual!r}."
            )

    return metadata


def validate_evaluation_data(dataset):
    base = (
        NATIVE_PACKAGE
        / "Datasets"
        / "Preprocessed"
        / dataset
    )

    for split in ("Train", "Test"):
        directory = base / split / "Tracks"

        if (
            not directory.is_dir()
            or not any(directory.glob("*.npy"))
        ):
            raise FileNotFoundError(
                f"Evaluation data missing: {directory}"
            )


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

    command = native_command(
        "evaluation.py",
        [
            "--identifier",
            identifier,
            "--device",
            args.device,
            "--alpha_search",
            "True",
        ],
    )

    print(f"Dataset: {dataset}", flush=True)
    print(f"Profile: {identifier}", flush=True)

    if args.dry_run:
        print("Dry run: " + " ".join(command))
        return

    check_device(args.device, parser)

    checkpoint = restore_checkpoint(
        dataset,
        identifier,
    )

    validate_checkpoint(checkpoint, profile)
    validate_evaluation_data(dataset)

    log_path = run_and_log(
        command,
        dataset,
        identifier,
        "evaluate",
    )

    from report import write_report

    destination = write_report(
        dataset,
        identifier,
        log_path,
    )

    print(
        f"Evaluation archived: {destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()
