"""V25: direct comparison of TranAD+, KNN, MLP and random scores with three rulers.

Protocol: tranad_plus/campaigns/knn/README.md, section V25 (fixed before running).
Read-only inputs; outputs in results/v25/comparison/.
"""
from pathlib import Path
import json, platform, sys, time
import numpy as np
import pandas as pd
import torch

from .features import window_features, standardiser
from .metrics import ranking, runs, event_coverage, bf200_point_adjusted_f1
from .run import knn_scores, by_channel, sha256, ROOT, NATIVE_K, NATIVE_WINDOW
from .calibration import load_tracks, split_index, real_mask, CAL_FRACTION
from .mlp import AE, fit, score, SEEDS, EPOCHS, LR
from .extensions import opssat_frame, tranad_fragment_scores

OUT = ROOT / 'results' / 'v25' / 'comparison'
FARS = (0.01, 0.05, 0.10)
VARIANTS = ('reference', 'encoder_only')


def at_equal_far(y, s, far, events=None):
    th = np.quantile(s[~y], 1 - far)
    p = s > th
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    out = {'far_target': far, 'far_observed': fp / max(int((~y).sum()), 1), 'recall': tp / max(tp + fn, 1),
           'precision': tp / (tp + fp) if tp + fp else 0.0, 'f1': 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
    if events is not None:
        out['event_coverage'] = event_coverage(p, events)
    return out


def native(ds):
    tracks, _ = load_tracks(ds)
    w = NATIVE_WINDOW[ds]
    y = np.concatenate([t['labels'] for t in tracks])
    events = runs(y, np.cumsum([0] + [len(t['test']) for t in tracks])[:-1])
    feats = []
    for t in tracks:
        f = window_features(t['train'], w); m, _ = split_index(len(f), w, CAL_FRACTION)
        feats.append((f[:m], window_features(t['test'], w)))
    med, std = standardiser(np.concatenate([f[0] for f in feats]))
    feats = [((a - med) / std, (b - med) / std) for a, b in feats]
    detectors = [('KNN statistics', None, np.concatenate([knn_scores(a, b, NATIVE_K) for a, b in feats]))]
    for seed in SEEDS:
        test = []
        for a, b in feats:
            torch.manual_seed(seed)
            test.append(score(fit(AE(a.shape[1]), a, EPOCHS['per_track'], LR, seed), b))
        detectors.append(('MLP autoencoder (per track)', seed, np.concatenate(test)))
        print(f'  {ds} MLP seed {seed} done', flush=True)
    mask = real_mask([len(t['test']) for t in tracks], w)
    sources = []
    for v in VARIANTS:
        for seed in SEEDS:
            f = ROOT / f'results/v13/architectures/{ds}/{v}/reference/seed_{seed}/scores.npz'
            z = np.load(f); a = 10 if z['alpha_applicable'][0] else 0
            assert np.array_equal(z['test_labels'][mask].astype(bool), y), 'label alignment'
            detectors.append((f'TranAD+ {v}', seed, z['test_scores'][a][mask].astype(np.float64)))
            sources.append([str(f.relative_to(ROOT)), sha256(f)])
    for seed in SEEDS:
        detectors.append(('uniform random', seed, np.random.default_rng(seed).random(len(y))))
    rows = []
    for name, seed, s in detectors:
        base = {'dataset': ds, 'unit': 'sample', 'detector': name, 'seed': seed, **ranking(y, s),
                'bf200_PA_f1_ORACLE': bf200_point_adjusted_f1(s, y, events)}
        rows += [{**base, **at_equal_far(y, s, far, events)} for far in FARS]
        print(f'  {ds} {name} {seed}: AUC {base["auc_roc"]:.3f} PA-F1 {base["bf200_PA_f1_ORACLE"]:.3f}', flush=True)
    return rows, sources


def opssat():
    info, F = opssat_frame()
    part, ch = info.part.to_numpy(), info.channel.to_numpy()
    z = by_channel(F, ch, part == 'train')
    te = part == 'test'; y = info.label.to_numpy()[te]
    k = json.load(open(ROOT / 'results/v21/knn/OPS-SAT/results.json'))['rows'][0]['selected_k']
    detectors = [('KNN statistics', None, knn_scores(z[part == 'train'], z[te], k))]
    for seed in SEEDS:
        torch.manual_seed(seed)
        detectors.append(('MLP autoencoder (universal)', seed,
                          score(fit(AE(z.shape[1]), z[part == 'train'], EPOCHS['OPS-SAT'], LR, seed), z[te])))
    for v in VARIANTS:
        for seed in SEEDS:
            detectors.append((f'TranAD+ {v}', seed, tranad_fragment_scores(info, v, seed)[0][te]))
    for seed in SEEDS:
        detectors.append(('uniform random', seed, np.random.default_rng(seed).random(len(y))))
    rows = []
    for name, seed, s in detectors:
        base = {'dataset': 'OPS-SAT', 'unit': 'fragment', 'detector': name, 'seed': seed, **ranking(y, s)}
        rows += [{**base, **at_equal_far(y, s, far)} for far in FARS]
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print('OPS-SAT ...', flush=True)
    rows = opssat(); sources = {}
    for ds in ('SMAP', 'MSL'):
        print(f'{ds} ...', flush=True)
        r, sources[ds] = native(ds); rows += r
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'all_rows.csv', index=False)
    cols = ['auc_roc', 'auc_pr', 'bf200_PA_f1_ORACLE', 'recall', 'precision', 'f1', 'event_coverage', 'far_observed']
    s = t.groupby(['dataset', 'detector', 'far_target'], sort=False)[cols].agg(['mean', 'std'])
    s.to_csv(OUT / 'summary_mean_sd.csv')
    json.dump({'campaign': 'v25_comparison', 'fars': FARS, 'seeds': SEEDS, 'tranad_sources': sources,
               'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
               'platform': platform.platform(),
               'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
               'seconds': time.time() - started}, open(OUT / 'manifest.json', 'w'), indent=1)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    print(t.groupby(['dataset', 'detector', 'far_target'], sort=False)[cols].mean().round(3).to_string())


if __name__ == '__main__':
    main()
