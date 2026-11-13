"""V23: MLP autoencoder on elementary statistics, PTQ/QAT, and KNN with quantized memory.

Protocol: tranad_plus/campaigns/knn/README.md, section V23 (fixed before running).
Read-only inputs; outputs in results/v23/lightweight/. Quantization is simulated
(fake quantization in floating point); no integer kernel is used or claimed.
"""
from pathlib import Path
import json, platform, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import fragment_features, window_features, standardiser
from .metrics import ranking, thresholded, runs, event_coverage, bf200_point_adjusted_f1
from .run import knn_scores, by_channel, sha256, prepare_data, ROOT, SPLIT_SEED, OPSSAT_K, NATIVE_K, NATIVE_WINDOW
from .calibration import load_tracks, split_index, pot_threshold, CAL_FRACTION

OUT = ROOT / 'results' / 'v23' / 'lightweight'
SEEDS = (2026, 2027, 2028, 2029, 2030)
HIDDEN, CODE, LR, BATCH = 16, 3, 1e-3, 64
EPOCHS = {'OPS-SAT': 300, 'pooled': 30, 'per_track': 100}
QAT_FRACTION, QAT_LR = 0.2, 1e-4
FORMATS = {'fp32': (None, None), 'W8': (8, None), 'W4': (4, None), 'W8A8': (8, 8), 'W4A4': (4, 4)}
KNN_BITS = (None, 8, 4)
torch.set_num_threads(4)


# ---------------------------------------------------------------- quantizers
def ste(x, q):
    return x + (q - x).detach()


def quant_weight(w, bits):
    qmax = 2 ** (bits - 1) - 1
    s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
    return ste(w, torch.clamp(torch.round(w / s), -qmax, qmax) * s)


def quant_act(x, scale, bits, signed):
    qmin, qmax = (-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)
    s = max(scale, 1e-12) / qmax
    return ste(x, torch.clamp(torch.round(x / s), qmin, qmax) * s)


class AE(nn.Module):
    SIGNED = (True, False, True, False)        # inputs of the 4 linear layers: x, relu, code, relu

    def __init__(self, d):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, HIDDEN), nn.Linear(HIDDEN, CODE),
                                     nn.Linear(CODE, HIDDEN), nn.Linear(HIDDEN, d)])
        self.wbits = self.abits = None
        self.register_buffer('scales', torch.ones(4))

    def forward(self, x, record=False):
        inputs = []
        for i, layer in enumerate(self.layers):
            if record:
                inputs.append(x.abs().max().item())
            if self.abits:
                x = quant_act(x, self.scales[i].item(), self.abits, self.SIGNED[i])
            w = quant_weight(layer.weight, self.wbits) if self.wbits else layer.weight
            x = nn.functional.linear(x, w, layer.bias)
            if i in (0, 2):
                x = torch.relu(x)
        return (x, inputs) if record else x

    def calibrate(self, data):
        with torch.no_grad():
            self.scales = torch.tensor(self.forward(torch.as_tensor(data, dtype=torch.float32), record=True)[1])


def fit(model, data, epochs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.as_tensor(data, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(BATCH):
            loss = ((model(X[idx]) - X[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def score(model, data):
    with torch.no_grad():
        X = torch.as_tensor(data, dtype=torch.float32)
        return ((model(X) - X) ** 2).sum(1).numpy().astype(np.float64)


def train_formats(train, seed, epochs):
    """FP32 network, its PTQ versions and QAT versions of W8A8/W4A4. Returns {name: model}."""
    torch.manual_seed(seed)
    base = fit(AE(train.shape[1]), train, epochs, LR, seed)
    base.calibrate(train)
    models = {}
    for name, (wb, ab) in FORMATS.items():
        m = AE(train.shape[1]); m.load_state_dict(base.state_dict()); m.wbits, m.abits = wb, ab
        models[('PTQ' if wb else 'FP32', name)] = m
        if ab:
            q = AE(train.shape[1]); q.load_state_dict(base.state_dict()); q.wbits, q.abits = wb, ab
            models[('QAT', name)] = fit(q, train, max(1, int(QAT_FRACTION * epochs)), QAT_LR, seed + 1)
    return models


def mlp_resources(d):
    params = sum(a * b + b for a, b in ((d, HIDDEN), (HIDDEN, CODE), (CODE, HIDDEN), (HIDDEN, d)))
    weights = params - (HIDDEN + CODE + HIDDEN + d)
    macs = d * HIDDEN + HIDDEN * CODE + CODE * HIDDEN + HIDDEN * d
    return {'parameters': params, 'bytes_fp32': 4 * params,
            'bytes_W8': weights + 4 * (params - weights), 'bytes_W4': weights / 2 + 4 * (params - weights),
            'macs_per_decision': macs}


# ---------------------------------------------------------------- KNN with quantized memory
def quantize_features(ref, queries, bits):
    if bits is None:
        return ref, queries
    qmax = 2 ** (bits - 1) - 1
    s = np.maximum(np.abs(ref).max(0), 1e-12) / qmax
    q = lambda a: np.clip(np.round(a / s), -qmax, qmax)
    return q(ref), [q(x) for x in queries]


# ---------------------------------------------------------------- OPS-SAT
def opssat_data():
    csv = ROOT / 'data' / 'raw' / 'OPSSAT' / 'segments.csv'
    train, test, _, _, validation = prepare_data(
        {'segments_csv': str(csv), 'split_seed': SPLIT_SEED, 'validation_fraction': 0.2}, return_validation=True)
    groups = [(g, 'train') for g in train] + [(g, 'validation') for g in validation] + [(g, 'test') for g in test]
    part = np.array([p for _, p in groups]); ch = np.array([g['channel'] for g, _ in groups])
    y = np.array([bool(g['label']) for g, _ in groups])
    z = by_channel(np.array([fragment_features(g['values']) for g, _ in groups]), ch, part == 'train')
    return z, y, part


def opssat_rows():
    z, y, part = opssat_data()
    tr, va, te = z[part == 'train'], z[part == 'validation'], z[part == 'test']
    yv, yt = y[part == 'validation'], y[part == 'test']
    rows = []

    def evaluate(detector, extra, s_val, s_test):
        th = float(np.quantile(s_val[~yv], 0.95))
        rows.append({'dataset': 'OPS-SAT', 'detector': detector, **extra, 'threshold': th,
                     **ranking(yt, s_test), **thresholded(yt, s_test > th)})

    for seed in SEEDS:
        for (mode, fmt), m in train_formats(tr, seed, EPOCHS['OPS-SAT']).items():
            evaluate('MLP autoencoder', {'seed': seed, 'training': mode, 'format': fmt}, score(m, va), score(m, te))
        print(f'  OPS-SAT MLP seed {seed} done', flush=True)
    auc = {k: ranking(yv, knn_scores(tr, va, k))['auc_roc'] for k in OPSSAT_K}
    k = max(auc, key=lambda c: (auc[c], -c))
    for bits in KNN_BITS:
        r, (v, t) = quantize_features(tr, [va, te], bits)
        evaluate('KNN statistics', {'k': k, 'format': 'fp32' if bits is None else f'INT{bits}'},
                 knn_scores(r, v, k), knn_scores(r, t, k))
    return rows, {'n_reference': int(len(tr)), 'd': int(z.shape[1]), 'knn_k': k}


# ---------------------------------------------------------------- SMAP / MSL
def native_rows(ds):
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
    mem = np.concatenate([f[0] for f in feats]); cal = np.concatenate([f[1] for f in feats if len(f[1])])
    test = np.concatenate([f[2] for f in feats])
    rows = []

    def evaluate(detector, extra, s_cal, s_test, oracle=True):
        for rule, th in (('quantile', float(np.quantile(s_cal, 0.99))), ('POT', pot_threshold(s_cal, 0.01))):
            pred = s_test > th
            rows.append({'dataset': ds, 'detector': detector, **extra, 'rule': rule, 'threshold': th,
                         **ranking(y, s_test), **thresholded(y, pred), 'event_coverage': event_coverage(pred, events),
                         'bf200_point_adjusted_f1_ORACLE': bf200_point_adjusted_f1(s_test, y, events)
                         if oracle and rule == 'quantile' else np.nan})

    for seed in SEEDS:
        t0 = time.time()
        for (mode, fmt), m in train_formats(mem, seed, EPOCHS['pooled']).items():
            evaluate('MLP autoencoder (one network)', {'seed': seed, 'training': mode, 'format': fmt},
                     score(m, cal), score(m, test))
        cal_s, test_s = [], []
        for a, c, te in feats:                       # sensitivity: one FP32 network per track
            torch.manual_seed(seed)
            m = fit(AE(a.shape[1]), a, EPOCHS['per_track'], LR, seed)
            if len(c):
                cal_s.append(score(m, c))
            test_s.append(score(m, te))
        evaluate('MLP autoencoder (one network per track, sensitivity)', {'seed': seed, 'training': 'FP32', 'format': 'fp32'},
                 np.concatenate(cal_s), np.concatenate(test_s), oracle=False)
        print(f'  {ds} MLP seed {seed} done ({time.time() - t0:.0f}s)', flush=True)
    for bits in KNN_BITS:
        cal_s, test_s = [], []
        for a, c, te in feats:
            r, (cq, tq) = quantize_features(a, [c, te], bits)
            if len(c):
                cal_s.append(knn_scores(r, cq, NATIVE_K))
            test_s.append(knn_scores(r, tq, NATIVE_K))
        evaluate('KNN statistics', {'k': NATIVE_K, 'window': w, 'format': 'fp32' if bits is None else f'INT{bits}'},
                 np.concatenate(cal_s), np.concatenate(test_s))
    return rows, {'n_reference': int(len(mem)), 'd': int(mem.shape[1]), 'window': w, 'events': len(events)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time(); rows = []; meta = {}
    print('OPS-SAT ...', flush=True)
    r, meta['OPS-SAT'] = opssat_rows(); rows += r
    for ds in ('SMAP', 'MSL'):
        print(f'{ds} ...', flush=True)
        r, meta[ds] = native_rows(ds); rows += r
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'all_rows.csv', index=False)
    keys = ['dataset', 'detector', 'training', 'format', 'rule']
    for k in keys:
        if k not in t:
            t[k] = np.nan
    t[keys] = t[keys].fillna('-')
    cols = ['auc_roc', 'auc_pr', 'precision', 'recall', 'f1', 'mcc', 'nominal_false_alarm_rate', 'event_coverage',
            'bf200_point_adjusted_f1_ORACLE']
    t.groupby(keys, sort=False)[cols].agg(['mean', 'std']).to_csv(OUT / 'summary_mean_sd.csv')
    res = []
    for ds, m in meta.items():
        res.append({'dataset': ds, 'detector': 'MLP autoencoder', **mlp_resources(m['d'])})
        n = m['n_reference'] * m['d']
        res.append({'dataset': ds, 'detector': 'KNN statistics', 'stored_values': n, 'bytes_fp32': 4 * n,
                    'bytes_W8': n, 'bytes_W4': n / 2, 'macs_per_decision': n})
    pd.DataFrame(res).to_csv(OUT / 'resources.csv', index=False)
    json.dump({'campaign': 'v23_lightweight', 'meta': meta, 'seeds': SEEDS, 'epochs': EPOCHS, 'formats': list(FORMATS),
               'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
               'platform': platform.platform(),
               'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
               'seconds': time.time() - started}, open(OUT / 'manifest.json', 'w'), indent=1, default=str)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    show = t[(t.rule != 'POT')].groupby(keys, sort=False)
    print(show[['auc_roc', 'auc_pr', 'f1', 'mcc', 'nominal_false_alarm_rate', 'event_coverage',
                'bf200_point_adjusted_f1_ORACLE']].mean().round(3).to_string())
    print(show['auc_roc'].std().round(3).to_string())


if __name__ == '__main__':
    main()
