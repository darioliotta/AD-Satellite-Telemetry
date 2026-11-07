"""Small, explicit BF200/POT tables; seed means and sample standard deviations."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from tranad_variants.common import atomic_destination, hash_file

METRICS = {'f1_pct': 'f1', 'precision_pct': 'precision', 'recall_pct': 'recall', 'AUC_PA_pct': 'ROC/AUC'}


def write_csv(path, rows):
    if not rows:
        return
    with atomic_destination(path) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    root = Path(root)
    rows = []
    for status_path in sorted(root.rglob('status.json')):
        status = json.loads(status_path.read_text())
        if status['status'] != 'complete':
            continue
        path = status_path.parent/'metrics.json'
        if hash_file(path) != status['artifacts']['metrics.json']:
            raise RuntimeError(f'Changed metrics: {path}')
        report = json.loads(path.read_text())
        p = report['profile']
        if 'conditions' in report:
            evaluations = [(c['condition'], c['methods']) for c in report['conditions']]
            stage, train_condition, protocol = 'frozen', 'original', 'locked_parent_alpha_threshold'
        else:
            evaluations = [(key, report[key]) for key in ('matched', 'ordered_test') if key in report]
            stage = 'smd' if p['dataset']=='SMD' else 'train'
            train_condition, protocol = p['input_condition'], 'BF200_POT_oracle'
        for test_condition, evaluation in evaluations:
            for method in ('bf', 'pot'):
                item = evaluation[method]
                row = dict(stage=stage, dataset=p['dataset'], variant=p['variant'], window=p['window_sz'],
                           ff=p['dim_feedforward'], epochs=p['epochs'], training_condition=train_condition,
                           test_condition=test_condition, method=method, protocol=protocol, seed=p['seed'],
                           alpha=item.get('alpha'), threshold=item.get(method+'_threshold'))
                for column, suffix in METRICS.items():
                    value = item.get(method+'_'+suffix)
                    row[column] = None if value is None else 100*value
                rows.append(row)
    write_csv(root/'tables/per_seed.csv', rows)
    grouped = {}
    keys = ('stage', 'dataset', 'variant', 'window', 'ff', 'epochs', 'training_condition', 'test_condition', 'method', 'protocol')
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        if len({r['seed'] for r in values}) != len(values):
            raise RuntimeError('Duplicate model seed in one summary group')
        result = dict(zip(keys, key))
        result['model_seeds'] = len(values)
        for column in METRICS:
            valid = [r[column] for r in values if r[column] is not None]
            result[column+'_available_seeds'] = len(valid)
            result[column+'_mean'] = statistics.mean(valid) if valid else None
            result[column+'_sample_std_pp'] = statistics.stdev(valid) if len(valid)>1 else None
        summary.append(result)
    write_csv(root/'tables/mean_std.csv', summary)
    return {'metric_rows': len(rows), 'summary_groups': len(summary)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[1]/'results/v15')
    print(summarize(parser.parse_args().output_root))
