"""Shared V14 diagnostic metrics; no torch import or model changes."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def anomaly_segments(labels, boundaries=None):
    y = np.asarray(labels, dtype=bool)
    start = y & ~np.r_[False, y[:-1]]
    end = y & ~np.r_[y[1:], False]
    if boundaries is not None:
        boundaries = np.asarray(boundaries, dtype=int)
        boundaries = boundaries[(boundaries > 0) & (boundaries < len(y))]
        start[boundaries] = y[boundaries]
        end[boundaries-1] = y[boundaries-1]
    return np.flatnonzero(start), np.flatnonzero(end)+1


def pa_stats(scores, labels, thresholds, boundaries=None):
    """Same global/fragment PA convention, with native index-zero behavior."""
    y = np.asarray(labels, dtype=bool)
    thresholds = np.atleast_1d(thresholds).astype(np.float64)
    a, b = anomaly_segments(y, boundaries)
    maxima = np.array([scores[i:j].max() for i, j in zip(a, b)])
    order = np.argsort(maxima)
    cumulative = np.r_[0, np.cumsum((b-a)[order])]
    tp = int(y.sum()) - cumulative[np.searchsorted(maxima[order], thresholds, side="right")]
    if boundaries is None and len(a) and y[0]:
        tp -= ((maxima[0] > thresholds) & (scores[0] <= thresholds)).astype(int)
    normal = np.sort(scores[~y])
    fp = len(normal) - np.searchsorted(normal, thresholds, side="right")
    f1 = 2*tp / (int(y.sum()) + tp + fp)
    return {"f1": f1, "tp": tp, "fp": fp}


def dense_pa_f1(scores, labels, boundaries=None):
    below = np.nextafter(scores.min(), np.array(-np.inf, dtype=scores.dtype))
    thresholds = np.concatenate((np.array([below], dtype=np.float64), np.unique(scores).astype(np.float64)))
    values = pa_stats(scores, labels, thresholds, boundaries)
    k = int(np.argmax(values["f1"]))
    return {"f1": float(values["f1"][k]), "threshold": float(thresholds[k])}


def event_metrics(scores, labels, threshold, boundaries=None):
    y = np.asarray(labels, dtype=bool)
    a, b = anomaly_segments(y, boundaries)
    pred = scores > threshold
    hit = np.array([bool(pred[i:j].any()) for i, j in zip(a, b)])
    adjusted = pa_stats(scores, y, [threshold], boundaries)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    return {"actual_fpr": fp/int((~y).sum()), "direct_point_recall": tp/int(y.sum()),
            "event_recall": float(hit.mean()), "pa_point_recall": float(adjusted["tp"][0])/int(y.sum()),
            "pa_f1": float(adjusted["f1"][0]), "events": len(a),
            "direct_point_f1": 2*tp/(int(y.sum())+tp+fp)}


def real_point_mask(profile, data_root, count):
    """Exclude extra padded target rows when comparing native window lengths."""
    if profile["dataset"] == "OPS-SAT":
        return np.ones(count, dtype=bool)
    paths = sorted((data_root / "processed/tranad_plus" / profile["dataset"] / "Test/Tracks").glob("*.npy"))
    pieces = []
    w = profile["window_sz"]
    for path in paths:
        n = len(np.load(path, mmap_mode="r", allow_pickle=False))
        pieces.append(np.r_[np.zeros(w-1, dtype=bool), np.ones(n, dtype=bool)])
    result = np.concatenate(pieces)[w-1:]
    if len(result) != count:
        raise RuntimeError("Native real-point mask does not match scored windows")
    return result



def ranking(labels, scores):
    if len(np.unique(labels)) < 2:
        return {"auc": None, "average_precision": None}
    return {"auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}



def raw_best_f1(labels, scores):
    """Exact unadjusted, test-oracle threshold sweep; alpha is already fixed."""
    order = np.argsort(scores)
    s, y = scores[order], labels[order].astype(bool)
    positives = int(y.sum())
    if not positives:
        return {"f1": 0., "threshold": float(s[-1])}
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]) + 1, len(s)]
    tp = np.r_[positives, positives - np.cumsum(y)[ends - 1]]
    counts = np.r_[len(s), len(s) - ends]
    f1 = 2 * tp / (positives + counts)
    k = int(np.argmax(f1))
    # The lower boundary must survive NumPy 1.x scalar casting to float32.
    threshold = float(np.nextafter(s[0], np.array(-np.inf, dtype=s.dtype))) if k == 0 else float(s[ends[k - 1] - 1])
    return {"f1": float(f1[k]), "threshold": float(threshold)}



def fixed_threshold(labels, scores, threshold):
    pred, y = scores > threshold, labels.astype(bool)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}



def fragment_ranking(scores, groups):
    result = {}
    for aggregation in ("mean", "max"):
        offset, values, labels = 0, [], []
        for group in groups:
            n = len(group["values"])
            values.append(float(getattr(np, aggregation)(scores[offset:offset+n])))
            labels.append(group["label"])
            offset += n
        assert offset == len(scores)
        result[aggregation] = ranking(np.array(labels), np.array(values))
    return result



