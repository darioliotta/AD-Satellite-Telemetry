"""V27 diagnosis: are KNN/TranAD+ scores comparable across tracks? (SMAP/MSL)

Hypothesis from the MSL examples (results/v27/msl_diagnosis/msl_examples.png): KNN scores rise
inside anomalies of 'quiet' tracks but stay below a threshold shared by all tracks.
Test: label-free per-track normalization, score / q99 of that track's held-out nominal
calibration windows (V22 split: last 20% of each training track). TranAD+ gets the same
treatment on the same last 20% of its (seen) training scores. Metrics as in V25.
"""
import json
import numpy as np, pandas as pd
from .calibration import load_tracks, split_index, real_mask, CAL_FRACTION
from .features import window_features, standardiser
from .run import knn_scores, ROOT, NATIVE_K, NATIVE_WINDOW
from .metrics import ranking, runs
from .comparison import at_equal_far, FARS

OUT = ROOT / 'results' / 'v27' / 'msl_diagnosis'


def per_track(ds):
    tracks, _ = load_tracks(ds); w = NATIVE_WINDOW[ds]
    y = np.concatenate([t['labels'] for t in tracks])
    events = runs(y, np.cumsum([0] + [len(t['test']) for t in tracks])[:-1])
    feats = []
    for t in tracks:
        f = window_features(t['train'], w); m, c0 = split_index(len(f), w, CAL_FRACTION)
        feats.append((f[:m], f[c0:], window_features(t['test'], w)))
    med, std = standardiser(np.concatenate([f[0] for f in feats]))
    raw, norm = [], []
    for a, c, te in feats:
        za = (a - med) / std
        s = knn_scores(za, (te - med) / std, NATIVE_K); raw.append(s)
        ref = knn_scores(za, (c - med) / std, NATIVE_K) if len(c) else knn_scores(za, za, NATIVE_K, leave_one_out=True)
        norm.append(s / max(np.quantile(ref, .99), 1e-12))
    out = {'KNN, one threshold for all tracks': np.concatenate(raw), 'KNN, per-track normalized': np.concatenate(norm)}
    mask_te = real_mask([len(t['test']) for t in tracks], w); mask_tr = real_mask([len(t['train']) for t in tracks], w)
    for seed in range(2026, 2031):
        z = np.load(ROOT / f'results/v13/architectures/{ds}/reference/reference/seed_{seed}/scores.npz')
        te, tr = z['test_scores'][10][mask_te].astype(float), z['train_scores'][10][mask_tr].astype(float)
        tn, i, j = [], 0, 0
        for t in tracks:
            n, nt = len(t['train']), len(t['test']); _, c0 = split_index(n, w, CAL_FRACTION)
            tn.append(te[j:j + nt] / max(np.quantile(tr[i + c0:i + n], .99), 1e-12)); i += n; j += nt
        out.setdefault('TranAD+ reference, one threshold for all tracks', []).append(te)
        out.setdefault('TranAD+ reference, per-track normalized', []).append(np.concatenate(tn))
    rows = []
    for name, s in out.items():
        for k, sc in enumerate(s if isinstance(s, list) else [s]):
            base = {'dataset': ds, 'detector': name, 'seed': 2026 + k if isinstance(s, list) else None, **ranking(y, sc)}
            rows += [{**base, **at_equal_far(y, sc, far, events)} for far in FARS]
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(per_track('MSL') + per_track('SMAP'))
    t.to_csv(OUT / 'per_track_normalization.csv', index=False)
    pd.set_option('display.width', 250)
    print(t.groupby(['dataset', 'detector', 'far_target'], sort=False)[['auc_roc', 'recall', 'precision', 'event_coverage']]
          .mean().round(3).to_string())


if __name__ == '__main__':
    main()


def where_alarms_go(ds='MSL', far=0.05):
    """Within-track AUC and the share of false alarms from tracks whose nominal test leaves the training range."""
    tracks, _ = load_tracks(ds); w = NATIVE_WINDOW[ds]
    y = np.concatenate([t['labels'] for t in tracks]); off = np.cumsum([0] + [len(t['test']) for t in tracks])
    feats = []
    for t in tracks:
        f = window_features(t['train'], w); m, _ = split_index(len(f), w, CAL_FRACTION)
        feats.append((f[:m], window_features(t['test'], w)))
    med, std = standardiser(np.concatenate([f[0] for f in feats]))
    knn = np.concatenate([knn_scores((a - med) / std, (b - med) / std, NATIVE_K) for a, b in feats])
    mask = real_mask([len(t['test']) for t in tracks], w)
    tran = np.mean([np.load(ROOT / f'results/v13/architectures/{ds}/reference/reference/seed_{s}/scores.npz')
                    ['test_scores'][10][mask] for s in range(2026, 2031)], 0)
    rows = []
    for k, t in enumerate(tracks):
        lab = t['labels']; nom = t['test'][~lab]
        shifted = bool(nom.size and (nom.min() < t['train'].min() or nom.max() > t['train'].max()))
        r = {'track': t['name'], 'nominal_test_outside_training_range': shifted, 'test_nominal': int((~lab).sum()),
             'test_anomalous': int(lab.sum())}
        for name, s in (('KNN', knn), ('TranAD+', tran)):
            sc = s[off[k]:off[k + 1]]; th = np.quantile(s[~y], 1 - far)
            r[f'{name}_false_alarms'] = int(((sc > th) & ~lab).sum()); r[f'{name}_true_alarms'] = int(((sc > th) & lab).sum())
            r[f'{name}_within_track_auc'] = ranking(lab, sc)['auc_roc'] if 0 < lab.sum() < len(lab) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

