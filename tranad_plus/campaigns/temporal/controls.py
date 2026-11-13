"""Deterministic within-window interventions; current sample is never altered."""
import numpy as np
import torch


def window_view(dataset):
    if hasattr(dataset, "windows"):
        return dataset.windows
    return dataset.big_track.unfold(0, dataset.window_size, 1).permute(0, 2, 1)


def transform(windows, condition):
    """Input [point, time, feature]; permute time jointly across all features.

    Each window gets its own permutation, drawn once from a private generator.
    The same transformed training dataset is reused in every epoch. This is
    a static order control, not fresh random augmentation at each epoch.
    No Python/NumPy/PyTorch global RNG state is consumed.
    """
    if condition == "original":
        return windows
    if condition == "repeat_current":
        return windows[:, -1:, :].expand_as(windows).contiguous()
    if condition == "reverse_past":
        result = windows.clone()
        result[:, :-1] = windows[:, :-1].flip(1)
        return result
    if condition.startswith("shuffle_"):
        seed = int(condition.split("_")[1])
        result = windows.clone()
        # Chunking limits peak temporary RAM on the native multivariate datasets.
        rng = np.random.default_rng(seed)
        past = windows.shape[1] - 1
        for start in range(0, len(windows), 8192):
            stop = min(start + 8192, len(windows))
            order = rng.random((stop-start, past)).argsort(axis=1)
            indices = torch.as_tensor(order, device=windows.device).unsqueeze(-1)
            result[start:stop, :-1] = windows[start:stop, :-1].gather(
                1, indices.expand(-1, -1, windows.shape[2]))
        return result
    raise ValueError(f"Unknown history condition: {condition}")


class Windows(torch.utils.data.Dataset):
    """Expose materialized windows to the existing A100 resident-data loader."""
    def __init__(self, dataset, condition):
        original = window_view(dataset)
        self.windows = transform(original, condition)
        if not torch.equal(self.windows[:, -1], original[:, -1]):
            raise AssertionError("History intervention changed reconstruction targets")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


def changed_bundle(bundle, condition):
    from tranad_plus.core.data import DataBundle
    return DataBundle({key: Windows(ds, condition) for key, ds in bundle.datasets.items()},
                      bundle.labels, {**bundle.inventory, "history_condition": condition}, bundle.groups)
