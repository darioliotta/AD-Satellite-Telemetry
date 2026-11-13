"""V26: KNN and MLP on window statistics of all input coordinates (SMAP/MSL).

Protocol: tranad_plus/campaigns/knn/README.md, section V26 (fixed before running).
Outputs in results/v26/multivariate/.
"""
from pathlib import Path
import glob, json, platform, sys, time
import numpy as np
import pandas as pd
import torch

from .features import window_features, standardiser
from .metrics import ranking, runs, bf200_point_adjusted_f1
from .run import knn_scores, sha256, ROOT, NATIVE_K, NATIVE_WINDOW
from .calibration import split_index, CAL_FRACTION
from .mlp import AE, fit, score, SEEDS, EPOCHS, LR
from .comparison import at_equal_far, FARS

OUT = ROOT / 'results' / 'v26' / 'multivariate'


def load(ds):
    base = ROOT / 'data' / 'processed' / 'tranad_plus' / ds
    names = [Path(f).name for f in sorted(glob.glob(str(base / 'Test' / 'Tracks' / '*.npy')))]
    out = []
    for n in names:
        lab = np.load(base / 'Test' / 'Labels' / n); lab = lab[:, 0] if lab.ndim > 1 else lab
        out.append((np.load(base / 'Train' / 'Tracks' / n), np.load(base / 'Test' / 'Tracks' / n), lab.astype(bool)))
    return out


def multi_features(x, w, keep):
    return np.hstack([window_features(x[:, j], w) for j in keep])


def run(ds):
    tracks = load(ds); w = NATIVE_WINDOW[ds]
    keep = np.flatnonzero(np.concatenate([t[0] for t in tracks]).std(0) > 0)
    y = np.concatenate([t[2] for t in tracks])
    events = runs(y, np.cumsum([0] + [len(t[1]) for t in tracks])[:-1])
    feats = []
    for tr, te, _ in tracks:
        f = multi_features(tr, w, keep); m, _ = split_index(len(f), w, CAL_FRACTION)
        feats.append((f[:m], multi_features(te, w, keep)))
    med, std = standardiser(np.concatenate([f[0] for f in feats]))
    feats = [((a - med) / std, (b - med) / std) for a, b in feats]
    d = feats[0][0].shape[1]
    print(f'  {ds}: kept {len(keep)} coordinates, dimension {d}', flush=True)
    detectors = [('KNN statistics (all coordinates)', None,
                  np.concatenate([knn_scores(a, b, NATIVE_K) for a, b in feats]))]
    for seed in SEEDS:
        test = []
        for a, b in feats:
            torch.manual_seed(seed)
            test.append(score(fit(AE(d), a, EPOCHS['per_track'], LR, seed), b))
        detectors.append(('MLP autoencoder per track (all coordinates)', seed, np.concatenate(test)))
        print(f'  {ds} MLP seed {seed} done', flush=True)
    rows = []
    for name, seed, s in detectors:
        base = {'dataset': ds, 'unit': 'sample', 'detector': name, 'seed': seed, 'input_dimension': d,
                'coordinates_kept': len(keep), **ranking(y, s), 'bf200_PA_f1_ORACLE': bf200_point_adjusted_f1(s, y, events)}
        rows += [{**base, **at_equal_far(y, s, far, events)} for far in FARS]
    return rows, {'kept_coordinates': keep.tolist(), 'dimension': d}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time(); rows = []; meta = {}
    for ds in ('SMAP', 'MSL'):
        print(f'{ds} ...', flush=True)
        r, meta[ds] = run(ds); rows += r
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'all_rows.csv', index=False)
    cols = ['auc_roc', 'auc_pr', 'bf200_PA_f1_ORACLE', 'recall', 'precision', 'f1', 'event_coverage', 'far_observed']
    t.groupby(['dataset', 'detector', 'far_target'], sort=False)[cols].agg(['mean', 'std']).to_csv(OUT / 'summary_mean_sd.csv')
    json.dump({'campaign': 'v26_multivariate', 'meta': meta, 'seeds': SEEDS, 'python': sys.version.split()[0],
               'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(),
               'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
               'seconds': time.time() - started}, open(OUT / 'manifest.json', 'w'), indent=1)
    pd.set_option('display.width', 250)
    print(t.groupby(['dataset', 'detector', 'far_target'], sort=False)[cols].mean().round(3).to_string())


if __name__ == '__main__':
    main()
