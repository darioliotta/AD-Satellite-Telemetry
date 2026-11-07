"""Adapters to the audited dataset-specific baseline protocols."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch

from .bootstrap import activate_baseline
from .common import hash_file
activate_baseline()
from TranADPlus.src.datasets import TSDataset_tracks
from opssat_pipeline import prepare_data, FragmentWindows

_HASH_CACHE = {}


@dataclass
class DataBundle:
    datasets: dict
    labels: dict
    inventory: dict
    groups: dict = field(default_factory=dict)


def _fingerprint(path, root):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = hash_file(path)
    return {"file": str(path.relative_to(root)), "bytes": stat.st_size, "sha256": _HASH_CACHE[key]}


def inventory(profile, data_root):
    root = Path(data_root).resolve()
    if profile["dataset"] == "OPS-SAT":
        paths = [root / "raw/OPSSAT/segments.csv"]
    else:
        paths = []
        for split in ("Train", "Test"):
            parent = root / "processed/tranad_plus" / profile["dataset"] / split
            tracks = sorted((parent / "Tracks").glob("*.npy"))
            labels = sorted((parent / "Labels").glob("*.npy"))
            if not tracks or [p.name for p in tracks] != [p.name for p in labels]:
                raise ValueError(f"Missing or unmatched track/label files: {parent}")
            for track, label in zip(tracks, labels):
                a, b = (np.load(p, mmap_mode="r", allow_pickle=False) for p in (track, label))
                if a.ndim != 2 or a.shape[1] != profile["features"] or b.shape != a.shape:
                    raise ValueError(f"Feature/label dimensions differ from profile: {track}")
                if not len(a):
                    raise ValueError(f"Empty track: {track}")
            paths.extend(tracks + labels)
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError(f"Missing input files under {root}")
    return {"files": [_fingerprint(p, root) for p in paths], "file_order": "sorted_by_filename"}


def load_data(profile, data_root, files=None):
    root = Path(data_root).resolve()
    files = files or inventory(profile, root)
    if profile["dataset"] == "OPS-SAT":
        p = {**profile, "segments_csv": str(root / "raw/OPSSAT/segments.csv")}
        tr, te, scalers, info, va = prepare_data(p, return_validation=True)
        groups = {"train": tr, "test": te, "validation": va}
        datasets = {key: FragmentWindows(g, scalers, profile["window_sz"])
                    for key, g in groups.items() if g}
        return DataBundle(datasets, {k: d.point_labels for k, d in datasets.items()},
                          {**files, "split": info}, groups)
    parent = root / "processed/tranad_plus" / profile["dataset"]
    datasets = {}
    for split, key in (("Train", "train"), ("Test", "test")):
        datasets[key] = TSDataset_tracks(
            track_paths=[str(p) for p in sorted((parent / split / "Tracks").glob("*.npy"))],
            anom_paths=[str(p) for p in sorted((parent / split / "Labels").glob("*.npy"))],
            window_size=profile["window_sz"], padding=True, scaling=True,
            scaler=None if key == "train" else datasets["train"].scaler, downsample=None)
    labels = {}
    for key, ds in datasets.items():
        if len(ds) <= 0 or not bool(torch.isfinite(ds.big_track).all()):
            raise ValueError(f"Empty/nonfinite normalized {key} data")
        if not bool(torch.isfinite(ds.big_anom).all()):
            raise ValueError(f"Nonfinite {key} labels")
        labels[key] = (ds.big_anom[profile["window_sz"] - 1:].sum(dim=1) > 0).numpy().astype(np.int8)
        if len(labels[key]) != len(ds):
            raise AssertionError("Window/label alignment mismatch")
    return DataBundle(datasets, labels,
                      {**files, "split": {"policy": "official_train_test",
                       "window_policy": "upstream_global_concatenation_with_per_track_padding",
                       "point_adjustment": "upstream_global",
                       "train_windows": len(datasets["train"]), "test_windows": len(datasets["test"])}})
