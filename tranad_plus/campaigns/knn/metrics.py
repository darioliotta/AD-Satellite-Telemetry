"""Detection metrics with explicit units; PA only as a labelled paper-comparison quantity."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef


def runs(labels, breaks=()):
    """Contiguous positive intervals, never crossing a track boundary."""
    y = np.asarray(labels, bool)
    cut = np.zeros(len(y) + 1, bool)
    cut[list(breaks)] = True
    out, start = [], None
    for i in range(len(y) + 1):
        if start is not None and (i == len(y) or not y[i] or cut[i]):
            out.append((start, i)); start = None
        if i < len(y) and y[i] and start is None:
            start = i
    return out


def thresholded(y, pred):
    y = np.asarray(y, bool); p = np.asarray(pred, bool)
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    return {'precision': tp / (tp + fp) if tp + fp else 0.0, 'recall': tp / (tp + fn) if tp + fn else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if tp else 0.0, 'mcc': float(matthews_corrcoef(y, p)),
            'nominal_false_alarm_rate': fp / max(int((~y).sum()), 1), 'TP': tp, 'FP': fp, 'FN': fn}


def ranking(y, score):
    return {'auc_roc': float(roc_auc_score(y, score)), 'auc_pr': float(average_precision_score(y, score)),
            'prevalence': float(np.mean(y))}


def event_coverage(pred, events):
    return float(np.mean([pred[a:b].any() for a, b in events])) if events else float('nan')


def bf200_point_adjusted_f1(score, y, events):
    """Authors' oracle protocol: 200 thresholds in [min, mean+3std], point adjustment, best test F1."""
    y = np.asarray(y, bool); total = int(y.sum()); best = 0.0
    for th in np.linspace(score.min(), score.mean() + 3 * score.std(), 200):
        p = score > th
        tp = sum(b - a for a, b in events if p[a:b].any())
        fp = int((p & ~y).sum())
        if tp:
            best = max(best, 2 * tp / (2 * tp + fp + total - tp))
    return best
