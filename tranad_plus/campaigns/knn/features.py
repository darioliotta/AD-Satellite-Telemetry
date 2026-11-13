"""Elementary signal statistics for fragments (OPS-SAT) and causal windows (SMAP/MSL)."""
import numpy as np

FRAGMENT_FEATURES = ('std', 'range', 'q01', 'q99', 'kurtosis', 'diff_std', 'diff_absmax', 'zero_step_fraction')
WINDOW_FEATURES = ('mean', 'std', 'min', 'max', 'last', 'last_diff', 'diff_std', 'diff_absmax', 'zero_step_fraction')


def fragment_features(values):
    """Eight statistics of one univariate fragment (raw units)."""
    v = np.asarray(values, dtype=np.float64)
    d = np.diff(v) if len(v) > 1 else np.zeros(1)
    std = v.std()
    kurt = float(np.mean((v - v.mean())**4) / std**4 - 3.0) if std > 0 and len(v) > 3 else 0.0
    return np.array([std, np.ptp(v), np.quantile(v, .01), np.quantile(v, .99), kurt,
                     d.std(), np.abs(d).max(), np.mean(d == 0)])


def window_features(series, window):
    """Nine statistics of every causal window ending at each real sample of one track.

    The first sample is repeated to supply missing history, so a track of length L
    gives L rows; no window crosses into another track.
    """
    x = np.asarray(series, dtype=np.float64)
    padded = np.concatenate([np.full(window - 1, x[0]), x])
    w = np.lib.stride_tricks.sliding_window_view(padded, window)          # (L, window)
    d = np.diff(w, axis=1) if window > 1 else np.zeros((len(x), 1))
    return np.column_stack([w.mean(1), w.std(1), w.min(1), w.max(1), w[:, -1], d[:, -1],
                            d.std(1), np.abs(d).max(1), (d == 0).mean(1)])


def standardiser(reference):
    """Median/std of nominal reference rows; std floored to keep constant statistics finite."""
    med = np.median(reference, axis=0)
    std = reference.std(axis=0)
    return med, np.maximum(std, 1e-6 * (np.abs(med) + 1.0))
