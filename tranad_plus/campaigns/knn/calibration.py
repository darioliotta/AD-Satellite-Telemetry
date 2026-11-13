"""V22: label-free threshold calibration on held-out nominal data (SMAP/MSL).

Revision of the V21 threshold rule, declared before running: the V21 leave-one-out
rule scored KNN training windows against overlapping near-copies of themselves and
gave too low a threshold (observed nominal false-alarm rate 12-16% instead of 1%).

Rule (same for every detector):
- each training track (all nominal) is split in time: the first 80% builds the
  detector, the last 20% is held out for calibration; the calibration windows start
  after the end of the memory part, so they share no sample with it;
- thresholds are computed on the pooled calibration scores, for a target nominal
  false-alarm rate alpha in {0.01, 0.001}, with two rules:
  empirical quantile (1 - alpha) and POT (GPD on excesses over the 0.98 quantile,
  same parameters for every dataset and detector);
- test labels are used only to measure the result, once.

TranAD+ (V13 reference and encoder-only, seeds 2026-2030, alpha_mix 0.5): no
retraining. Its calibration scores are the last 20% of each training track, which
the network did see during training; V19 found train/validation nominal error
differences of about 1%, so this is declared as an approximation. The authors'
practice (calibration on the whole training) is also reported for comparison.

OPS-SAT is unchanged: V21 already calibrates on held-out validation-nominal fragments.
Outputs: results/v22/threshold_calibration/.
"""
from pathlib import Path
import glob, json, platform, sys, time
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import genpareto

from .features import window_features, standardiser
from .metrics import ranking, thresholded, runs, event_coverage
from .run import knn_scores, pca_scores, sha256, NATIVE_K, NATIVE_WINDOW, TRANAD_SEEDS, TRANAD_VARIANTS, RANDOM_SEEDS

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'v22' / 'threshold_calibration'
CAL_FRACTION = 0.2
CAL_FRACTION_SENSITIVITY = (0.1, 0.3)       # reported, not used for selection
ALPHAS = (0.01, 0.001)
POT_LEVEL = 0.98


def pot_threshold(cal, alpha, level=POT_LEVEL):
    """Static peaks-over-threshold: GPD fit on excesses over the `level` quantile."""
    cal = np.asarray(cal, float)
    t = float(np.quantile(cal, level)); exc = cal[cal > t] - t
    if len(exc) < 10:
        return float('nan')
    gamma, _, sigma = genpareto.fit(exc, floc=0)
    r = alpha * len(cal) / len(exc)
    return t - sigma * np.log(r) if abs(gamma) < 1e-8 else t + sigma / gamma * (r ** -gamma - 1)


def thresholds(cal):
    out = {}
    for a in ALPHAS:
        out[('quantile', a)] = float(np.quantile(cal, 1 - a))
        out[('POT', a)] = pot_threshold(cal, a)
    return out


def load_tracks(ds):
    base = ROOT / 'data' / 'processed' / 'tranad_plus' / ds
    files = sorted(glob.glob(str(base / 'Test' / 'Tracks' / '*.npy')))
    tracks = []
    for f in files:
        name = Path(f).name
        lab = np.load(base / 'Test' / 'Labels' / name); lab = lab[:, 0] if lab.ndim > 1 else lab
        tracks.append({'name': name, 'train': np.load(base / 'Train' / 'Tracks' / name)[:, 0],
                       'test': np.load(f)[:, 0], 'labels': lab.astype(bool)})
    return tracks, files


def split_index(n, window, fraction):
    """Memory = windows ending before m; calibration = windows starting at m or later."""
    m = int(round(n * (1 - fraction)))
    return m, m + window - 1


def knn_and_pca(tracks, window, k, fraction, with_pca=True):
    feats = []
    for t in tracks:
        tr = window_features(t['train'], window); m, c0 = split_index(len(tr), window, fraction)
        feats.append((tr[:m], tr[c0:], window_features(t['test'], window)))
    med, std = standardiser(np.concatenate([f[0] for f in feats]))
    cal, test = [], []
    for mem, c, te in feats:
        zm = (mem - med) / std
        if len(c):                      # very short tracks may leave no window after the gap
            cal.append(knn_scores(zm, (c - med) / std, k))
        test.append(knn_scores(zm, (te - med) / std, k))
    out = {'KNN statistics': (np.concatenate(cal), np.concatenate(test))}
    if with_pca:
        zm = (np.concatenate([f[0] for f in feats]) - med) / std
        out['PCA statistics (linear control)'] = (pca_scores(zm, (np.concatenate([f[1] for f in feats if len(f[1])]) - med) / std)[0],
                                                  pca_scores(zm, (np.concatenate([f[2] for f in feats]) - med) / std)[0])
    return out


def real_mask(lengths, window):
    """V13 historical layout: each track preceded by window-1 padded targets, first block dropped."""
    return np.concatenate([np.r_[np.zeros(window - 1, bool), np.ones(n, bool)] for n in lengths])[window - 1:]


def tranad_scores(ds, tracks, variant, seed, fraction):
    f = ROOT / f'results/v13/architectures/{ds}/{variant}/reference/seed_{seed}/scores.npz'
    z = np.load(f); a = 10 if z['alpha_applicable'][0] else 0
    w = NATIVE_WINDOW[ds]
    tr = z['train_scores'][a][real_mask([len(t['train']) for t in tracks], w)]
    te = z['test_scores'][a][real_mask([len(t['test']) for t in tracks], w)]
    tail, start = [], 0
    for t in tracks:
        n = len(t['train']); _, c0 = split_index(n, w, fraction)
        tail.append(tr[start + c0:start + n]); start += n
    assert start == len(tr) and len(te) == sum(len(t['test']) for t in tracks)
    return {'tail': np.concatenate(tail), 'whole': tr, 'test': te, 'source': (str(f.relative_to(ROOT)), sha256(f))}


def evaluate(name, cal, test, y, events, extra, calibration):
    rows = []
    rank = ranking(y, test)
    for (rule, a), th in thresholds(cal).items():
        pred = test > th if np.isfinite(th) else np.zeros(len(test), bool)
        rows.append({'detector': name, **extra, 'calibration': calibration, 'rule': rule, 'target_alpha': a,
                     'threshold': th, **rank, **thresholded(y, pred), 'event_coverage': event_coverage(pred, events)})
    return rows


def dataset_rows(ds):
    tracks, files = load_tracks(ds)
    y = np.concatenate([t['labels'] for t in tracks])
    events = runs(y, np.cumsum([0] + [len(t['test']) for t in tracks])[:-1])
    w, rows, sources = NATIVE_WINDOW[ds], [], []
    for name, (cal, test) in knn_and_pca(tracks, w, NATIVE_K, CAL_FRACTION).items():
        rows += evaluate(name, cal, test, y, events, {'window': w, 'k': NATIVE_K}, 'held-out tail 20%')
    for fr in CAL_FRACTION_SENSITIVITY:
        cal, test = knn_and_pca(tracks, w, NATIVE_K, fr, with_pca=False)['KNN statistics']
        rows += evaluate('KNN statistics', cal, test, y, events, {'window': w, 'k': NATIVE_K},
                         f'held-out tail {int(fr * 100)}% (sensitivity)')
    n_cal = len(cal)
    for s in RANDOM_SEEDS:
        r = np.random.default_rng(s)
        rows += evaluate('uniform random', r.random(n_cal), r.random(len(y)), y, events, {'seed': s}, 'held-out tail 20%')
    for v in TRANAD_VARIANTS:
        for seed in TRANAD_SEEDS:
            sc = tranad_scores(ds, tracks, v, seed, CAL_FRACTION); sources.append(sc['source'])
            rows += evaluate(f'TranAD+ {v}', sc['tail'], sc['test'], y, events, {'seed': seed},
                             'training tail 20% (seen in training)')
            rows += evaluate(f'TranAD+ {v}', sc['whole'], sc['test'], y, events, {'seed': seed},
                             'whole training (authors\' practice)')
    meta = {'test_points': int(len(y)), 'events': len(events), 'tranad_sources': sources,
            'track_files_sha256': {Path(f).name: sha256(f) for f in files}}
    return rows, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time(); rows = []; metas = {}
    for ds in ('SMAP', 'MSL'):
        print(f'{ds} ...', flush=True)
        r, metas[ds] = dataset_rows(ds)
        rows += [{'dataset': ds, **x} for x in r]
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'all_rows.csv', index=False)
    cols = ['auc_roc', 'auc_pr', 'precision', 'recall', 'f1', 'mcc', 'nominal_false_alarm_rate', 'event_coverage']
    summary = t.groupby(['dataset', 'detector', 'calibration', 'rule', 'target_alpha'], sort=False)[cols].agg(['mean', 'std'])
    summary.to_csv(OUT / 'summary_mean_sd.csv')
    json.dump({'campaign': 'v22_threshold_calibration', 'cal_fraction': CAL_FRACTION, 'alphas': ALPHAS,
               'pot_level': POT_LEVEL, 'python': sys.version.split()[0], 'numpy': np.__version__,
               'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform(),
               'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
               'inputs': metas, 'seconds': time.time() - started}, open(OUT / 'manifest.json', 'w'), indent=1)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    print(t.groupby(['dataset', 'detector', 'calibration', 'rule', 'target_alpha'], sort=False)[cols].mean().round(3).to_string())


if __name__ == '__main__':
    main()
