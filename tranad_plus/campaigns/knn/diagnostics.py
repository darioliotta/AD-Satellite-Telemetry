"""V21 diagnostics: why PA/BF200 metrics move when TranAD+ is simplified or quantized.

1. random_control: uniform random scores under the authors' BF200 + point adjustment.
2. msl_out_of_range: test values outside the training min-max range (sigmoid output is bounded).
3. bf200_grid: BF200 threshold spacing versus nominal score quantiles for every V13 parent,
   with and without the out-of-range points (scores capped at 0.05).
4. precision_scores: FP32 / INT8 / INT4 nominal and anomalous score quantiles and AUC from V20.
Read-only inputs; outputs in results/v21/knn/diagnostics/.
"""
from pathlib import Path
import glob, json
import numpy as np
import pandas as pd
from .metrics import runs, bf200_point_adjusted_f1

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'v21' / 'knn' / 'diagnostics'


def native_labels(ds):
    files = sorted(glob.glob(str(ROOT / f'data/processed/tranad_plus/{ds}/Test/Labels/*.npy')))
    lab = [np.load(f) for f in files]
    lab = [l[:, 0] if l.ndim > 1 else l for l in lab]
    return np.concatenate(lab).astype(bool), np.cumsum([0] + [len(l) for l in lab])[:-1], files


def opssat_labels():
    d = pd.read_csv(ROOT / 'data/raw/OPSSAT/segments.csv', usecols=['segment', 'anomaly', 'train'])
    d = d[d.train == 0]
    sizes = d.groupby('segment', sort=False).size().values
    return d.anomaly.values.astype(bool), np.cumsum(np.r_[0, sizes])[:-1]


def random_control():
    rows = []
    for ds in ('OPS-SAT', 'SMAP', 'MSL'):
        y, breaks = opssat_labels() if ds == 'OPS-SAT' else native_labels(ds)[:2]
        ev = runs(y, breaks); p = y.mean()
        f1 = [bf200_point_adjusted_f1(np.random.default_rng(s).random(len(y)), y, ev) for s in range(5)]
        rows.append({'dataset': ds, 'points': len(y), 'prevalence': p, 'events': len(ev),
                     'median_event_length': float(np.median([b - a for a, b in ev])),
                     'all_positive_f1_without_PA': 2 * p / (1 + p),
                     'random_bf200_PA_f1_mean': float(np.mean(f1)), 'random_bf200_PA_f1_sd': float(np.std(f1, ddof=1))})
    return pd.DataFrame(rows)


def out_of_range(ds):
    base = ROOT / f'data/processed/tranad_plus/{ds}'
    X = np.concatenate([np.load(f) for f in sorted(glob.glob(str(base / 'Train/Tracks/*.npy')))])
    lo, hi = X.min(0), X.max(0); span = np.where(hi > lo, hi - lo, 1)
    y, _, files = native_labels(ds); rows = []
    for f in files:
        z = (np.load(f.replace('/Labels/', '/Tracks/')) - lo) / span
        excess = (np.maximum(z - 1, 0) + np.maximum(-z, 0)).max(1)
        lab = np.load(f); lab = (lab[:, 0] if lab.ndim > 1 else lab).astype(bool)
        if (excess > 0.5).any():
            rows.append({'dataset': ds, 'track': Path(f).name, 'points_outside_by_more_than_half_range': int((excess > .5).sum()),
                         'inside_annotated_anomaly': int(((excess > .5) & lab).sum()), 'max_excess_in_ranges': float(excess.max()),
                         'coordinates': sorted({int(c) for c in np.argwhere((np.maximum(z - 1, 0) + np.maximum(-z, 0)) > .5)[:, 1]})})
    return rows


def bf200_grid(ds):
    rows = []
    for f in sorted(glob.glob(str(ROOT / f'results/v13/architectures/{ds}/*/*/seed_*/scores.npz'))):
        variant, config, seed = Path(f).parts[-4:-1]
        z = np.load(f); y = z['test_labels'].astype(bool); ev = runs(y)
        s = z['test_scores'][10] if z['alpha_applicable'][0] else z['test_scores'][0]
        cap = np.minimum(s, 0.05)
        grid = np.linspace(s.min(), s.mean() + 3 * s.std(), 200)
        rows.append({'dataset': ds, 'variant': variant, 'configuration': config, 'seed': seed, 'alpha': 0.5,
                     'score_max': float(s.max()), 'bf200_grid_step': float(grid[1] - grid[0]),
                     'nominal_q99': float(np.quantile(s[~y], .99)),
                     'grid_thresholds_below_nominal_q99': int((grid < np.quantile(s[~y], .99)).sum()),
                     'bf200_PA_f1': bf200_point_adjusted_f1(s, y, ev),
                     'bf200_PA_f1_scores_capped_at_0.05': bf200_point_adjusted_f1(cap, y, ev)})
    return pd.DataFrame(rows)


def precision_scores():
    rows = []
    for f in glob.glob(str(ROOT / 'results/v20/mechanisms/analysis/*/*/*/seed_*/*/result.json')):
        d = json.load(open(f)); i = d['identity']; sep = d['metrics']['separation']
        rows.append({**{k: i[k] for k in ('dataset', 'variant', 'configuration', 'seed', 'format')},
                     'nominal_median': sep['nominal']['quantiles_05_25_50_75_95_99'][2],
                     'nominal_q99': sep['nominal']['quantiles_05_25_50_75_95_99'][5],
                     'anomalous_median': sep['anomalous']['quantiles_05_25_50_75_95_99'][2]})
    t = pd.DataFrame(rows)
    auc = pd.read_csv(ROOT / 'results/v20/mechanisms/tables/per_case.csv')
    auc = auc[(auc.stage == 'analysis') & (auc.endpoint == 'fixed_nominal_quantile') & (auc.view == 'raw')]
    auc = auc[['dataset', 'variant', 'configuration', 'seed', 'format', 'AUC_score', 'AP_score']]
    t = t.merge(auc, on=['dataset', 'variant', 'configuration', 'seed', 'format'], how='left')
    return t.groupby(['dataset', 'variant', 'configuration', 'format'])[
        ['nominal_median', 'nominal_q99', 'anomalous_median', 'AUC_score', 'AP_score']].mean().reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random_control().to_csv(OUT / 'random_score_bf200_pa_control.csv', index=False)
    json.dump(out_of_range('MSL') + out_of_range('SMAP'), open(OUT / 'out_of_training_range_points.json', 'w'), indent=1)
    grid = pd.concat([bf200_grid('MSL'), bf200_grid('SMAP')])
    grid.to_csv(OUT / 'bf200_grid_per_parent.csv', index=False)
    grid.groupby(['dataset', 'variant', 'configuration'])[['nominal_q99', 'bf200_grid_step', 'bf200_PA_f1',
        'bf200_PA_f1_scores_capped_at_0.05']].agg(['mean', 'std']).to_csv(OUT / 'bf200_grid_summary.csv')
    precision_scores().to_csv(OUT / 'precision_score_distributions.csv', index=False)
    print(open(OUT / 'random_score_bf200_pa_control.csv').read())


if __name__ == '__main__':
    main()
