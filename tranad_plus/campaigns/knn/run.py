"""V21: KNN on elementary statistics versus TranAD+ on OPS-SAT, SMAP and MSL.

Read-only inputs: raw/processed datasets, V13 score arrays, V20 tables.
Outputs: results/v21/knn/. No training of neural networks.
"""
from pathlib import Path
import glob, hashlib, json, platform, sys, time
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # opssat_pipeline uses flat imports
from opssat_pipeline import prepare_data
from .features import fragment_features, window_features, standardiser, FRAGMENT_FEATURES, WINDOW_FEATURES
from .metrics import ranking, thresholded, runs, event_coverage, bf200_point_adjusted_f1

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'v21' / 'knn'
SPLIT_SEED = 2026
OPSSAT_K = (1, 3, 5, 10, 20)
OPSSAT_ALARM_QUANTILE = 0.95          # of validation-nominal fragment scores
NATIVE_K, NATIVE_WINDOW = 5, {'SMAP': 10, 'MSL': 50}
NATIVE_ALARM_QUANTILE = 0.99          # of leave-one-out training scores
SENSITIVITY = {'window': (10, 25, 50, 100), 'k': (1, 5, 20)}
TRANAD_SEEDS = (2026, 2027, 2028, 2029, 2030)
TRANAD_VARIANTS = ('reference', 'encoder_only')
RANDOM_SEEDS = range(5)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def knn_scores(reference, queries, k, leave_one_out=False):
    nn = NearestNeighbors(n_neighbors=k + int(leave_one_out)).fit(reference)
    dist = nn.kneighbors(queries)[0]
    return (dist[:, 1:] if leave_one_out else dist).mean(1)


def pca_scores(reference, queries):
    pca = PCA(n_components=0.90, svd_solver='full').fit(reference)
    rec = pca.inverse_transform(pca.transform(queries))
    return ((queries - rec) ** 2).sum(1), int(pca.n_components_)


def by_channel(features, channels, reference_mask):
    """Standardise each statistic with nominal reference fragments of the same channel."""
    z = np.empty_like(features)
    for ch in np.unique(channels):
        m = channels == ch
        med, std = standardiser(features[m & reference_mask])
        z[m] = (features[m] - med) / std
    return z


def fragment_block(name, z, y, ch, part, select_k=True):
    """KNN with k chosen on validation AUC-ROC; alarm at validation-nominal quantile; test once."""
    ref = z[part == 'train']
    val, test = part == 'validation', part == 'test'
    candidates = {}
    for k in (OPSSAT_K if select_k else (5,)):
        s = knn_scores(ref, z[val], k)
        candidates[k] = ranking(y[val], s)['auc_roc']
    k = max(candidates, key=lambda c: (candidates[c], -c))
    s_val, s_test = knn_scores(ref, z[val], k), knn_scores(ref, z[test], k)
    threshold = float(np.quantile(s_val[~y[val]], OPSSAT_ALARM_QUANTILE))
    return {'detector': name, 'selected_k': k, 'validation_auc_by_k': candidates, 'threshold': threshold,
            **ranking(y[test], s_test), **thresholded(y[test], s_test > threshold)}, s_test


def opssat():
    csv = ROOT / 'data' / 'raw' / 'OPSSAT' / 'segments.csv'
    train, test, _, inventory, validation = prepare_data(
        {'segments_csv': str(csv), 'split_seed': SPLIT_SEED, 'validation_fraction': 0.2}, return_validation=True)
    groups = [(g, 'train') for g in train] + [(g, 'validation') for g in validation] + [(g, 'test') for g in test]
    ids = np.array([g['id'] for g, _ in groups]); part = np.array([p for _, p in groups])
    ch = np.array([g['channel'] for g, _ in groups]); y = np.array([bool(g['label']) for g, _ in groups])
    F = np.array([fragment_features(g['values']) for g, _ in groups])
    z = by_channel(F, ch, part == 'train')
    rows, per_fragment = [], {'segment': ids[part == 'test'], 'label': y[part == 'test']}

    res, s = fragment_block('KNN statistics', z, y, ch, part); rows.append(res); per_fragment['knn'] = s
    ref = z[part == 'train']
    s_val, ncomp = pca_scores(ref, z[part == 'validation']); s_test, _ = pca_scores(ref, z[part == 'test'])
    th = float(np.quantile(s_val[~y[part == 'validation']], OPSSAT_ALARM_QUANTILE))
    rows.append({'detector': 'PCA statistics (linear control)', 'components': ncomp, 'threshold': th,
                 **ranking(y[part == 'test'], s_test), **thresholded(y[part == 'test'], s_test > th)})
    per_fragment['pca'] = s_test
    yt = y[part == 'test']
    rows.append({'detector': 'all-positive', 'auc_roc': 0.5, 'auc_pr': float(yt.mean()), 'prevalence': float(yt.mean()),
                 **thresholded(yt, np.ones(len(yt), bool))})
    rnd = []
    for seed in RANDOM_SEEDS:
        r = np.random.default_rng(seed); sv, st = r.random((part == 'validation').sum()), r.random(len(yt))
        rnd.append({**ranking(yt, st), **thresholded(yt, st > np.quantile(sv[~y[part == 'validation']], OPSSAT_ALARM_QUANTILE))})
    rows.append({'detector': 'uniform random (mean of 5)', **{k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]}})

    # TranAD+ under the same fragment protocol, from existing V13 point scores (read-only).
    sources = []
    for variant in TRANAD_VARIANTS:
        for seed in TRANAD_SEEDS:
            f = ROOT / f'results/v13/architectures/OPS-SAT/{variant}/reference/seed_{seed}/scores.npz'
            if not f.exists():
                continue
            npz = np.load(f); sources.append({'file': str(f.relative_to(ROOT)), 'sha256': sha256(f)})
            a = 10 if npz['alpha_applicable'][0] else 0       # alpha = 0.5 for two-output models
            stats = {}
            for split in ('train', 'validation', 'test'):
                frame = pd.DataFrame({'seg': npz[f'{split}_segment_ids'], 'e': npz[f'{split}_scores'][a]})
                frame['log'] = np.log10(frame.e + 1e-12)
                g = frame.groupby('seg', sort=False)
                stats.update({int(i): v for i, v in zip(g.size().index, np.column_stack([
                    g.log.mean(), g.e.quantile(.95).pipe(lambda q: np.log10(q + 1e-12)), g.log.max(), g.log.std().fillna(0)]))})
            E = np.array([stats[int(i)] for i in ids])
            p95 = E[:, 1]
            # direct score: 95th percentile of point errors, robustly calibrated per channel
            zc = np.empty_like(p95)
            for c in np.unique(ch):
                m = ch == c; ref_c = p95[m & (part == 'train')]
                iqr = max(np.subtract(*np.quantile(ref_c, [.75, .25])), 1e-12)
                zc[m] = (p95[m] - np.median(ref_c)) / iqr
            th = float(np.quantile(zc[(part == 'validation') & ~y], OPSSAT_ALARM_QUANTILE))
            rows.append({'detector': f'TranAD+ {variant} direct p95 error', 'seed': seed, 'threshold': th,
                         **ranking(yt, zc[part == 'test']), **thresholded(yt, zc[part == 'test'] > th)})
            res, _ = fragment_block(f'TranAD+ {variant} error statistics + KNN', by_channel(E, ch, part == 'train'), y, ch, part)
            rows.append({**res, 'seed': seed})
    meta = {'inventory_split': {k: inventory[k] for k in ('source_sha256', 'n_train_fragments', 'n_validation_fragments',
                                                          'n_validation_anomalous_fragments', 'n_test_fragments', 'split_seed')},
            'features': FRAGMENT_FEATURES, 'tranad_sources': sources}
    return rows, pd.DataFrame(per_fragment), meta


def native(dataset):
    base = ROOT / 'data' / 'processed' / 'tranad_plus' / dataset
    files = sorted(glob.glob(str(base / 'Test' / 'Tracks' / '*.npy')))
    tracks = []
    for f in files:
        name = Path(f).name
        train = np.load(base / 'Train' / 'Tracks' / name)[:, 0]
        test = np.load(f)[:, 0]
        lab = np.load(base / 'Test' / 'Labels' / name); lab = lab[:, 0] if lab.ndim > 1 else lab
        tracks.append((name, train, test, lab.astype(bool)))
    y = np.concatenate([t[3] for t in tracks])
    breaks = np.cumsum([0] + [len(t[2]) for t in tracks])[:-1]
    events = runs(y, breaks)

    def evaluate(window, k, with_pca=False):
        tr_feats = [window_features(t[1], window) for t in tracks]
        te_feats = [window_features(t[2], window) for t in tracks]
        med, std = standardiser(np.concatenate(tr_feats))           # pooled over training tracks
        loo, test, pca_tr, pca_te = [], [], [], []
        for a, b in zip(tr_feats, te_feats):
            za, zb = (a - med) / std, (b - med) / std
            loo.append(knn_scores(za, za, k, leave_one_out=True)); test.append(knn_scores(za, zb, k))
        loo, test = np.concatenate(loo), np.concatenate(test)
        out = {'knn': (loo, test)}
        if with_pca:
            ztr = (np.concatenate(tr_feats) - med) / std; zte = (np.concatenate(te_feats) - med) / std
            s_tr, ncomp = pca_scores(ztr, ztr); s_te, _ = pca_scores(ztr, zte)
            out['pca'] = (s_tr, s_te, ncomp)
        return out

    def report(name, train_scores, test_scores, extra=None):
        th = float(np.quantile(train_scores, NATIVE_ALARM_QUANTILE)); pred = test_scores > th
        return {'detector': name, **(extra or {}), 'threshold': th, **ranking(y, test_scores), **thresholded(y, pred),
                'event_coverage': event_coverage(pred, events),
                'bf200_point_adjusted_f1_ORACLE': bf200_point_adjusted_f1(test_scores, y, events)}

    w = NATIVE_WINDOW[dataset]
    main = evaluate(w, NATIVE_K, with_pca=True)
    rows = [report('KNN statistics', *main['knn'], {'window': w, 'k': NATIVE_K}),
            report('PCA statistics (linear control)', main['pca'][0], main['pca'][1], {'window': w, 'components': main['pca'][2]})]
    rows.append({'detector': 'all-positive', 'prevalence': float(y.mean()), **thresholded(y, np.ones(len(y), bool)),
                 'auc_roc': 0.5, 'auc_pr': float(y.mean())})
    rnd = [report('random', np.random.default_rng(s).random(10000), np.random.default_rng(s + 100).random(len(y)))
           for s in RANDOM_SEEDS]
    rows.append({'detector': 'uniform random (mean of 5)', **{k: float(np.mean([r[k] for r in rnd]))
                                                               for k in rnd[0] if k != 'detector'}})
    sens = []
    for win in SENSITIVITY['window']:
        for k in SENSITIVITY['k']:
            r = evaluate(win, k)['knn'][1]
            sens.append({'window': win, 'k': k, **ranking(y, r)})
            print(f'  {dataset} sensitivity w={win} k={k}: AUC-ROC {sens[-1]["auc_roc"]:.3f}', flush=True)
    meta = {'tracks': len(tracks), 'test_points': int(len(y)), 'events': len(events), 'features': WINDOW_FEATURES,
            'track_files_sha256': {Path(f).name: sha256(f) for f in files}}
    return rows, pd.DataFrame(sens), meta


def tranad_native_from_v20(dataset):
    t = pd.read_csv(ROOT / 'results/v20/mechanisms/tables/mean_std.csv')
    t = t[(t.stage == 'analysis') & (t.family == 'historical_real_samples') & (t.dataset == dataset) &
          (t.configuration == 'reference') & (t.format == 'fp32') & t.variant.isin(TRANAD_VARIANTS)]
    rows = []
    for v in TRANAD_VARIANTS:
        fixed = t[(t.variant == v) & (t.endpoint == 'fixed_nominal_quantile') & (t.view == 'raw')].iloc[0]
        pa = t[(t.variant == v) & (t.endpoint == 'BF200_PA_oracle_fixed_alpha') & (t.view == 'PA')].iloc[0]
        rows.append({'detector': f'TranAD+ {v} (V20, mean of {int(fixed.cases)} seeds)', 'auc_roc': fixed.AUC_score_mean,
                     'auc_pr': fixed.AP_score_mean, 'precision': fixed.precision_mean, 'recall': fixed.recall_mean,
                     'f1': fixed.f1_mean, 'nominal_false_alarm_rate': fixed.nominal_false_positive_rate_mean,
                     'event_coverage': fixed.event_coverage_mean, 'bf200_point_adjusted_f1_ORACLE': pa.f1_mean})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time(); summary = []; manifest = {'campaign': 'v21_knn', 'started_unix': started,
        'python': sys.version.split()[0], 'numpy': np.__version__, 'sklearn': sklearn.__version__,
        'platform': platform.platform(), 'code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    print('OPS-SAT ...', flush=True)
    rows, frag, meta = opssat()
    (OUT / 'OPS-SAT').mkdir(exist_ok=True)
    frag.to_csv(OUT / 'OPS-SAT' / 'test_fragment_scores.csv', index=False)
    json.dump({'rows': rows, 'meta': meta}, open(OUT / 'OPS-SAT' / 'results.json', 'w'), indent=1, default=float)
    summary += [{'dataset': 'OPS-SAT', 'unit': 'fragment', **r} for r in rows]
    for ds in ('SMAP', 'MSL'):
        print(f'{ds} ...', flush=True)
        rows, sens, meta = native(ds)
        rows += tranad_native_from_v20(ds)
        (OUT / ds).mkdir(exist_ok=True)
        sens.to_csv(OUT / ds / 'sensitivity_window_k.csv', index=False)
        json.dump({'rows': rows, 'meta': meta}, open(OUT / ds / 'results.json', 'w'), indent=1, default=float)
        summary += [{'dataset': ds, 'unit': 'sample', **r} for r in rows]
    table = pd.DataFrame(summary)
    table.to_csv(OUT / 'summary_all_rows.csv', index=False)
    keep = ['auc_roc', 'auc_pr', 'precision', 'recall', 'f1', 'mcc', 'nominal_false_alarm_rate', 'event_coverage',
            'bf200_point_adjusted_f1_ORACLE']
    table['detector'] = table.detector.str.replace(r' \(V20.*', ' (V20)', regex=True)
    compact = table.groupby(['dataset', 'unit', 'detector'], sort=False)[[c for c in keep if c in table]].agg(['mean', 'std'])
    compact.to_csv(OUT / 'summary_mean_sd.csv')
    manifest['seconds'] = time.time() - started
    json.dump(manifest, open(OUT / 'manifest.json', 'w'), indent=1)
    pd.set_option('display.width', 250)
    print(table.groupby(['dataset', 'detector'], sort=False)[[c for c in keep if c in table]].mean().round(3).to_string())


if __name__ == '__main__':
    main()
