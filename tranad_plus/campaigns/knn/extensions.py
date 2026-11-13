"""V24 extensions: per-track MLP quantization (SMAP/MSL), OPS-SAT per channel, operation counts.

Protocol: tranad_plus/campaigns/knn/README.md, section V24 (fixed before running).
Read-only inputs; outputs in results/v24/. Quantization simulated (no integer kernel).
"""
from pathlib import Path
import json, math, platform, sys, time
import numpy as np
import pandas as pd
import torch

from .features import fragment_features, window_features, standardiser
from .metrics import ranking, thresholded, runs, event_coverage
from .run import knn_scores, by_channel, sha256, prepare_data, ROOT, SPLIT_SEED, NATIVE_K, NATIVE_WINDOW
from .calibration import load_tracks, split_index, pot_threshold, CAL_FRACTION
from .mlp import AE, fit, score, train_formats, SEEDS, EPOCHS, LR, HIDDEN, CODE

OUT = ROOT / 'results' / 'v24'
LITERATURE_CHANNELS = ('CADC0872', 'CADC0873', 'CADC0874', 'CADC0888', 'CADC0892', 'CADC0894')
SPECIALIZED_K = 5
OPSSAT_MEAN_LENGTH = None      # filled at run time for the operation counts


# ---------------------------------------------------------------- A. per-track MLP quantization
def per_track_quantization(ds):
    tracks, _ = load_tracks(ds)
    w = NATIVE_WINDOW[ds]
    y = np.concatenate([t['labels'] for t in tracks])
    events = runs(y, np.cumsum([0] + [len(t['test']) for t in tracks])[:-1])
    feats = []
    for t in tracks:
        f = window_features(t['train'], w); m, c0 = split_index(len(f), w, CAL_FRACTION)
        feats.append((f[:m], f[c0:], window_features(t['test'], w)))
    med, std = standardiser(np.concatenate([f[0] for f in feats]))
    feats = [tuple((a - med) / std for a in f) for f in feats]
    rows = []
    for seed in SEEDS:
        t0 = time.time(); cal, test = {}, {}
        for a, c, te in feats:
            for key, m in train_formats(a, seed, EPOCHS['per_track']).items():
                if len(c):
                    cal.setdefault(key, []).append(score(m, c))
                test.setdefault(key, []).append(score(m, te))
        for (mode, fmt) in test:
            sc, st = np.concatenate(cal[(mode, fmt)]), np.concatenate(test[(mode, fmt)])
            for rule, th in (('quantile', float(np.quantile(sc, 0.99))), ('POT', pot_threshold(sc, 0.01))):
                pred = st > th
                rows.append({'dataset': ds, 'detector': 'MLP autoencoder (one network per track)', 'seed': seed,
                             'training': mode, 'format': fmt, 'rule': rule, 'threshold': th, **ranking(y, st),
                             **thresholded(y, pred), 'event_coverage': event_coverage(pred, events)})
        print(f'  {ds} per-track quantization seed {seed} done ({time.time() - t0:.0f}s)', flush=True)
    return rows, {'tracks': len(tracks), 'networks': len(tracks)}


# ---------------------------------------------------------------- B. OPS-SAT per channel
def opssat_frame():
    csv = ROOT / 'data' / 'raw' / 'OPSSAT' / 'segments.csv'
    train, test, _, _, validation = prepare_data(
        {'segments_csv': str(csv), 'split_seed': SPLIT_SEED, 'validation_fraction': 0.2}, return_validation=True)
    groups = [(g, 'train') for g in train] + [(g, 'validation') for g in validation] + [(g, 'test') for g in test]
    info = pd.DataFrame({'id': [g['id'] for g, _ in groups], 'part': [p for _, p in groups],
                         'channel': [str(g['channel']) for g, _ in groups], 'label': [bool(g['label']) for g, _ in groups],
                         'length': [len(g['values']) for g, _ in groups]})
    F = np.array([fragment_features(g['values']) for g, _ in groups])
    return info, F


def tranad_fragment_scores(info, variant, seed):
    """V21 direct score: 95th percentile of point errors, robust per-channel calibration on training fragments."""
    f = ROOT / f'results/v13/architectures/OPS-SAT/{variant}/reference/seed_{seed}/scores.npz'
    npz = np.load(f); a = 10 if npz['alpha_applicable'][0] else 0
    p95 = {}
    for split in ('train', 'validation', 'test'):
        fr = pd.DataFrame({'seg': npz[f'{split}_segment_ids'], 'e': npz[f'{split}_scores'][a]})
        p95.update(np.log10(fr.groupby('seg').e.quantile(.95) + 1e-12).to_dict())
    s = info.id.map(lambda i: p95[int(i)]).to_numpy(float)
    z = np.empty_like(s)
    for c in info.channel.unique():
        m = (info.channel == c).to_numpy(); ref = s[m & (info.part == 'train').to_numpy()]
        z[m] = (s[m] - np.median(ref)) / max(np.subtract(*np.quantile(ref, [.75, .25])), 1e-12)
    return z, str(f.relative_to(ROOT))


def per_channel_metrics(info, s, detector, extra, specialized):
    """Metrics per test channel; threshold from validation-nominal scores (per channel if specialized)."""
    rows = []
    val_nom = ((info.part == 'validation') & ~info.label).to_numpy()
    global_th = float(np.quantile(s[val_nom], 0.95))
    for c in sorted(info.channel.unique()):
        m = (info.channel == c).to_numpy(); te = m & (info.part == 'test').to_numpy()
        yt = info.label.to_numpy()[te]
        th = float(np.quantile(s[m & val_nom], 0.95)) if specialized else global_th
        row = {'detector': detector, **extra, 'channel': c, 'test_nominal': int((~yt).sum()),
               'test_anomalous': int(yt.sum()), 'threshold': th, **thresholded(yt, s[te] > th)}
        if 0 < yt.sum() < len(yt):
            row.update(ranking(yt, s[te]))
        rows.append(row)
    return rows


def opssat_per_channel():
    info, F = opssat_frame()
    global OPSSAT_MEAN_LENGTH
    OPSSAT_MEAN_LENGTH = float(info.length.mean())
    part, ch = info.part.to_numpy(), info.channel.to_numpy()
    z = by_channel(F, ch, part == 'train')
    rows, sources = [], []
    tr = part == 'train'
    # universal models (one for all channels), sliced by channel
    rows += per_channel_metrics(info, knn_scores(z[tr], z, SPECIALIZED_K), 'KNN statistics (universal)', {}, False)
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = fit(AE(z.shape[1]), z[tr], EPOCHS['OPS-SAT'], LR, seed)
        rows += per_channel_metrics(info, score(m, z), 'MLP autoencoder (universal)', {'seed': seed}, False)
    # specialized models (one per channel)
    s_knn = np.full(len(info), np.nan); s_mlp = {seed: np.full(len(info), np.nan) for seed in SEEDS}
    for c in np.unique(ch):
        m = ch == c
        s_knn[m] = knn_scores(z[m & tr], z[m], min(SPECIALIZED_K, int((m & tr).sum())))   # tiny channels
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = fit(AE(z.shape[1]), z[m & tr], EPOCHS['OPS-SAT'], LR, seed)
            s_mlp[seed][m] = score(net, z[m])
        print(f'  OPS-SAT channel {c} done', flush=True)
    rows += per_channel_metrics(info, s_knn, 'KNN statistics (specialized)', {}, True)
    for seed in SEEDS:
        rows += per_channel_metrics(info, s_mlp[seed], 'MLP autoencoder (specialized)', {'seed': seed}, True)
    for variant in ('reference', 'encoder_only'):
        for seed in SEEDS:
            s, src = tranad_fragment_scores(info, variant, seed); sources.append(src)
            rows += per_channel_metrics(info, s, f'TranAD+ {variant} (universal, direct p95 error)', {'seed': seed}, False)
    counts = info.groupby(['channel', 'part', 'label']).size().unstack(fill_value=0)
    return rows, counts, sources


# ---------------------------------------------------------------- C. operation counts
def operation_counts(meta_v23):
    """Elementary operations per decision (estimates). Naive recomputation; see README V24-C."""
    def window_stats(w):
        # mean w, std 2w, min w, max w, diffs w-1, last diff 0, diff std 2(w-1), diff absmax 2(w-1), zero frac 2(w-1)
        return 5 * w + 7 * (w - 1)

    def fragment_stats(L):
        # std 2L, range 2L, two quantiles via one sort L log2 L, kurtosis 4L, diffs L-1, diff std 2L, absmax 2L, zero frac 2L
        return int(14 * L + L * math.log2(max(L, 2)))

    mlp = lambda d: 2 * (d * HIDDEN + HIDDEN * CODE + CODE * HIDDEN + HIDDEN * d) + 2 * HIDDEN + 3 * d
    rows = []
    for ds, d, stats in (('OPS-SAT', 8, fragment_stats(int(round(OPSSAT_MEAN_LENGTH)))),
                         ('SMAP', 9, window_stats(NATIVE_WINDOW['SMAP'])), ('MSL', 9, window_stats(NATIVE_WINDOW['MSL']))):
        n_ref = meta_v23[ds]['n_reference']
        unit = 'fragment (mean length)' if ds == 'OPS-SAT' else 'sample'
        rows.append({'dataset': ds, 'decision_unit': unit, 'statistics_ops': stats, 'standardization_ops': 2 * d,
                     'MLP_ops': mlp(d), 'KNN_ops_universal_reference': 3 * n_ref * d,
                     'note': 'MLP ops = 2 per multiply-accumulate + ReLU + score; KNN = subtract, square, add per stored value'})
    return pd.DataFrame(rows)


def main():
    started = time.time()
    (OUT / 'per_track_quantization').mkdir(parents=True, exist_ok=True)
    (OUT / 'opssat_per_channel').mkdir(parents=True, exist_ok=True)
    print('OPS-SAT per channel ...', flush=True)
    rows, counts, sources = opssat_per_channel()
    pc = pd.DataFrame(rows)
    pc.to_csv(OUT / 'opssat_per_channel' / 'all_rows.csv', index=False)
    counts.to_csv(OUT / 'opssat_per_channel' / 'fragment_counts.csv')
    lit = pc[pc.channel.isin(LITERATURE_CHANNELS)]
    per_seed = lit.groupby(['detector', 'seed'] if 'seed' in lit else ['detector'], dropna=False)[
        ['auc_roc', 'auc_pr', 'f1', 'mcc', 'precision', 'recall']].mean().reset_index()
    agg = per_seed.groupby('detector')[['auc_roc', 'auc_pr', 'f1', 'mcc', 'precision', 'recall']].agg(['mean', 'std'])
    agg.to_csv(OUT / 'opssat_per_channel' / 'mean_over_6_literature_channels.csv')
    pc.groupby(['detector', 'channel'])[['auc_roc', 'auc_pr', 'f1', 'mcc']].agg(['mean', 'std']).to_csv(
        OUT / 'opssat_per_channel' / 'by_channel_mean_sd.csv')
    q = []
    for ds in ('SMAP', 'MSL'):
        print(f'{ds} per-track quantization ...', flush=True)
        r, _ = per_track_quantization(ds); q += r
    q = pd.DataFrame(q)
    q.to_csv(OUT / 'per_track_quantization' / 'all_rows.csv', index=False)
    q.groupby(['dataset', 'training', 'format', 'rule'], sort=False)[
        ['auc_roc', 'auc_pr', 'precision', 'recall', 'f1', 'mcc', 'nominal_false_alarm_rate', 'event_coverage']
    ].agg(['mean', 'std']).to_csv(OUT / 'per_track_quantization' / 'summary_mean_sd.csv')
    meta_v23 = json.load(open(ROOT / 'results/v23/lightweight/manifest.json'))['meta']
    ops = operation_counts(meta_v23)
    ops.to_csv(OUT / 'operation_counts.csv', index=False)
    json.dump({'campaign': 'v24_extensions', 'seeds': SEEDS, 'literature_channels': LITERATURE_CHANNELS,
               'specialized_k': SPECIALIZED_K, 'opssat_mean_fragment_length': OPSSAT_MEAN_LENGTH,
               'tranad_sources': sources, 'python': sys.version.split()[0], 'torch': torch.__version__,
               'numpy': np.__version__, 'platform': platform.platform(),
               'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
               'seconds': time.time() - started}, open(OUT / 'manifest.json', 'w'), indent=1)
    pd.set_option('display.width', 250)
    print(agg.round(3).to_string())
    print(q[q.rule == 'quantile'].groupby(['dataset', 'training', 'format'], sort=False)[
        ['auc_roc', 'f1', 'nominal_false_alarm_rate']].mean().round(3).to_string())
    print(ops.to_string())


if __name__ == '__main__':
    main()
