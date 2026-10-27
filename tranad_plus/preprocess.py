
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import (
    DATASETS,
    NATIVE_PACKAGE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    result_directory,
)

from TranADPlus.src import helpers, parse_data, preprocess_data
from TranADPlus.src.global_config import config as native_config


def raw_directory(dataset):
    if dataset in ("SMAP", "MSL"):
        return RAW_DATA_DIR / "SMAP_MSL"

    return RAW_DATA_DIR / dataset


def validate_raw(dataset, directory):
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {directory}"
        )

    required = []

    if dataset in ("SMAP", "MSL"):
        required = [
            directory / "labeled_anomalies.csv",
            directory / "train" / "P-1.npy",
            directory / "test" / "P-1.npy",
            directory / "train" / "C-1.npy",
            directory / "test" / "C-1.npy",
        ]

    elif dataset == "SMD":
        required = [
            directory / "train",
            directory / "test",
            directory / "test_label",
        ]

    elif dataset == "SWaT":
        required = [
            directory / native_config.swat_normal_fn,
            directory / native_config.swat_attack_fn,
        ]

    elif dataset == "WADI":
        required = [
            directory / native_config.wadi_data_fn,
            directory / native_config.wadi_label_fn,
        ]

    elif dataset == "DSN_1k":
        required = [
            directory / native_config.mons_1k_pkl_fn,
            directory / native_config.drs_1k_pkl_fn,
        ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset input not found: {path}"
            )


def ensure_link(destination, source):
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink():
        if destination.resolve() == source:
            return

        raise FileExistsError(
            f"Unexpected symbolic link: {destination}"
        )

    if destination.exists():
        raise FileExistsError(
            f"Path already exists and cannot be replaced: "
            f"{destination}"
        )

    destination.symlink_to(
        source,
        target_is_directory=True,
    )


def connect_datasets(dataset, raw, processed):
    raw_base = NATIVE_PACKAGE / "Datasets" / "Raw"

    linked = (
        ("SMAP", "MSL")
        if dataset in ("SMAP", "MSL")
        else (dataset,)
    )

    for name in linked:
        ensure_link(raw_base / name, raw)

    processed.mkdir(parents=True, exist_ok=True)

    ensure_link(
        NATIVE_PACKAGE / "Datasets" / "Preprocessed",
        processed,
    )


def initialize(dataset):
    if dataset in ("SMAP", "MSL"):
        preprocess_data.initialize_smap_msl_data_channels(
            helpers.get_all_channels()
        )
    else:
        preprocess_data.initialize_dataset(dataset)


def inspect_split(dataset, processed, split):
    base = processed / dataset / split
    tracks_directory = base / "Tracks"
    labels_directory = base / "Labels"

    if not tracks_directory.is_dir():
        raise FileNotFoundError(tracks_directory)

    if not labels_directory.is_dir():
        raise FileNotFoundError(labels_directory)

    tracks = {
        path.stem: path
        for path in tracks_directory.glob("*.npy")
    }

    labels = {
        path.stem: path
        for path in labels_directory.glob("*.npy")
    }

    if not tracks or tracks.keys() != labels.keys():
        raise RuntimeError(
            f"Invalid processed track/label inventory: "
            f"{dataset}/{split}"
        )

    samples = 0
    anomalies = 0
    features = set()

    for name, path in sorted(tracks.items()):
        track = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        label = np.load(
            labels[name],
            mmap_mode="r",
            allow_pickle=False,
        )

        if track.ndim != 2 or label.shape != track.shape:
            raise RuntimeError(
                f"Invalid processed dimensions: {path}"
            )

        if not all(track.shape):
            raise RuntimeError(f"Empty track: {path}")

        samples += track.shape[0]
        features.add(track.shape[1])

        if split == "Test":
            anomalies += int(
                np.count_nonzero(
                    np.any(label > 0, axis=1)
                )
            )

    return {
        "tracks": len(tracks),
        "samples": samples,
        "anomalous_samples": anomalies,
        "feature_dimensions": sorted(features),
    }


def verify_dataset(dataset, processed):
    summary = {
        split: inspect_split(dataset, processed, split)
        for split in ("Train", "Test", "Both")
    }

    train = summary["Train"]
    test = summary["Test"]
    both = summary["Both"]

    if dataset == "DSN_1k":
        root = processed / dataset
        train_paths = {
            path.stem: path
            for path in (root / "Train" / "Tracks").glob("*.npy")
        }
        test_paths = {
            path.stem: path
            for path in (root / "Test" / "Tracks").glob("*.npy")
        }
        both_paths = {
            path.stem: path
            for path in (root / "Both" / "Tracks").glob("*.npy")
        }

        if set(train_paths) & set(test_paths):
            raise RuntimeError(
                "DSN_1k training and testing tracks overlap."
            )

        if not (set(train_paths) | set(test_paths)) <= set(both_paths):
            raise RuntimeError(
                "DSN_1k split tracks are missing from Both."
            )

        for name, path in (*train_paths.items(), *test_paths.items()):
            split_shape = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            ).shape
            both_shape = np.load(
                both_paths[name],
                mmap_mode="r",
                allow_pickle=False,
            ).shape

            if split_shape != both_shape:
                raise RuntimeError(
                    f"DSN_1k track shape mismatch: {name}"
                )
    else:
        if not (
            train["tracks"]
            == test["tracks"]
            == both["tracks"]
        ):
            raise RuntimeError(
                f"Inconsistent track counts for {dataset}"
            )

        if both["samples"] != (
            train["samples"] + test["samples"]
        ):
            raise RuntimeError(
                f"Inconsistent sample counts for {dataset}"
            )

    if dataset == "SMAP":
        actual = (
            train["tracks"],
            train["samples"],
            test["samples"],
            test["anomalous_samples"],
        )

        if actual != (54, 138004, 435826, 55922):
            raise RuntimeError(
                f"Unexpected SMAP preprocessing counts: {actual}"
            )

    return summary


def save_summary(dataset, raw, processed, splits):
    directory = result_directory(dataset)
    directory.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": dataset,
        "raw_directory": str(raw),
        "processed_directory": str(processed / dataset),
        "verified_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "splits": splits,
    }

    destination = directory / "preprocessing.json"

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=".preprocessing_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2)
        handle.write("\n")

    os.replace(temporary, destination)
    return destination


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        required=True,
    )

    parser.add_argument("--raw", type=Path)

    parser.add_argument(
        "--processed",
        type=Path,
        default=PROCESSED_DATA_DIR,
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    args = parser.parse_args()

    raw = (
        args.raw
        if args.raw is not None
        else raw_directory(args.dataset)
    ).expanduser().resolve()

    processed = args.processed.expanduser().resolve()

    if not args.verify_only:
        validate_raw(args.dataset, raw)

        output = processed / args.dataset

        if output.exists() and any(
            output.rglob("*.npy")
        ):
            parser.error(
                f"Preprocessed data already exist: {output}. "
                "Use --verify-only."
            )

        connect_datasets(
            args.dataset,
            raw,
            processed,
        )

        initialize(args.dataset)
        parse_data.parse_tracks(args.dataset)
        preprocess_data.preprocess_data(
            args.dataset,
            byte_mod="byte-dict",
        )

    splits = verify_dataset(args.dataset, processed)

    destination = save_summary(
        args.dataset,
        raw,
        processed,
        splits,
    )

    print(
        f"{args.dataset} preprocessing verified.",
        flush=True,
    )

    print(
        f"Train samples: {splits['Train']['samples']}",
        flush=True,
    )

    print(
        f"Test samples: {splits['Test']['samples']}",
        flush=True,
    )

    print(
        f"Summary: {destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()
