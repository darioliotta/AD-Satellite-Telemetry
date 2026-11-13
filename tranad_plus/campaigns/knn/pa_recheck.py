# Moved from the session scratchpad (2026-11-13). Run from the repository root with PYTHONPATH=.
"""Independent re-check: why random scores beat KNN under BF200 + PA (SMAP/MSL)."""
import sys, numpy as np
sys.path.insert(0, 'tranad_plus')
from TranADPlus.src.omnianomaly import bf_search
from tranad_plus.campaigns.knn.calibration import load_tracks, split_index, real_mask, CAL_FRACTION
from tranad_plus.campaigns.knn.features import window_features, standardiser
from tranad_plus.campaigns.knn.run import knn_scores, NATIVE_K, NATIVE_WINDOW
from tranad_plus.campaigns.knn.metrics import runs, bf200_point_adjusted_f1

def pa_curve(s, y, events):
    """Exact PA-F1 for every threshold: an event is detected iff its max score exceeds the threshold."""
    emax = np.array([s[a:b].max() for a, b in events]); elen = np.array([b - a for a, b in events])
    nom = np.sort(s[~y]); P = y.sum()
    th = np.unique(np.r_[emax - 1e-12, np.quantile(nom, np.linspace(0, 1, 4001))])
    tp = np.array([elen[emax > t].sum() for t in th]); fp = len(nom) - np.searchsorted(nom, th, side='right')
    f1 = 2 * tp / np.maximum(2 * tp + fp + (P - tp), 1)
    i = f1.argmax()
    return f1[i], fp[i] / len(nom), (emax > th[i]).mean()

for ds in ('SMAP', 'MSL'):
    tracks, _ = load_tracks(ds); w = NATIVE_WINDOW[ds]
    y = np.concatenate([t['labels'] for t in tracks])
    ev = runs(y, np.cumsum([0] + [len(t['test']) for t in tracks])[:-1])
    L = np.array([b - a for a, b in ev])
    feats = []
    for t in tracks:
        f = window_features(t['train'], w); m, _ = split_index(len(f), w, CAL_FRACTION)
        feats.append((f[:m], window_features(t['test'], w)))
    med, std = standardiser(np.concatenate([f[0] for f in feats]))
    knn = np.concatenate([knn_scores((a - med) / std, (b - med) / std, NATIVE_K) for a, b in feats])
    z = np.load(f'results/v13/architectures/{ds}/reference/reference/seed_2026/scores.npz')
    tran = z['test_scores'][10][real_mask([len(t['test']) for t in tracks], w)]
    rnd = np.random.default_rng(0).random(len(y))
    print(f'\n{ds}: {len(ev)} events, anomalous samples {y.sum()}, nominal {(~y).sum()}, '
          f'event length median {np.median(L):.0f}, min {L.min()}, max {L.max()}')
    # analytic expectation for random scores at false-alarm rate f: every event of length L hit with prob 1-(1-f)^L
    for f in (0.001, 0.005, 0.01):
        tp = (L * (1 - (1 - f) ** L)).sum(); fp = f * (~y).sum()
        print(f'  random, analytic, alarm rate {f:.3f}: PA-F1 {2*tp/(2*tp+fp+y.sum()-tp):.3f}')
    for name, s in (('random', rnd), ('KNN', knn), ('TranAD+ ref', tran)):
        auth = bf_search(s, y.astype(int), s.min(), s.mean() + 3 * s.std(), 200, verbose=False)
        f1d, far, cov = pa_curve(s, y, ev)
        emax = np.array([s[a:b].max() for a, b in ev]); q = np.quantile(s[~y], .99)
        print(f'  {name:12s} BF200 ours {bf200_point_adjusted_f1(s, y, ev):.3f} | authors bf_search {auth["bf_f1"]:.3f} | '
              f'best over ALL thresholds {f1d:.3f} (alarm rate {far:.4f}, events hit {cov:.2f}) | '
              f'events whose max score < nominal q99: {(emax < q).mean():.2f}')
