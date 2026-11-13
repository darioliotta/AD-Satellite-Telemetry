"""Compact, reproducible tables; deterministic baselines are not extra seeds."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from tranad_plus.core.common import atomic_destination


def summarize(root):
    root = Path(root)
    rows = []
    for path in sorted(root.rglob('result.json')):
        done = path.with_name('status.json')
        if not done.exists() or json.loads(done.read_text()).get('status') != 'complete':
            continue
        result = json.loads(path.read_text())
        meta, report = result['identity'], result['metrics']
        for name in ('fixed_nominal_quantile', 'POT_fixed_alpha', 'BF200_raw_oracle_fixed_alpha', 'BF200_PA_oracle_fixed_alpha'):
            endpoint = report[name]
            for view in ('raw', 'PA'):
                measured = endpoint.get(view, {})
                rows.append({**meta, 'endpoint': name, 'view': view,
                             'f1': measured.get('f1'), 'precision': measured.get('precision'), 'recall': measured.get('recall'),
                             'auc_binary': measured.get('auc_binary'), 'AUC_score': report['separation']['AUC_score'],
                             'AP_score': report['separation']['AP_score'],
                             'nominal_false_positive_rate': endpoint.get('nominal_false_positive_rate'),
                             'event_coverage': endpoint.get('event_coverage'),
                             'mean_fraction_directly_detected_within_events': endpoint.get('mean_fraction_directly_detected_within_events'),
                             'weight_representation_bytes': result.get('resources', {}).get('weight_representation_bytes'),
                             'model_tensor_bytes': result.get('resources', {}).get('model_tensor_bytes')})
        fragment = report['fragment_classification']
        if 'test' in fragment:
            m = fragment['test']
            rows.append({**meta, 'endpoint': 'validation_BF200_fragment_P95', 'view': 'fragment_no_PA',
                         'f1': m['f1'], 'precision': m['precision'], 'recall': m['recall'], 'auc_binary': m['auc_binary'],
                         'AUC_score': fragment['separation']['AUC_score'], 'AP_score': fragment['separation']['AP_score'],
                         'nominal_false_positive_rate': m['FP']/(m['FP']+m['TN']) if m['FP']+m['TN'] else None,
                         'event_coverage': None, 'mean_fraction_directly_detected_within_events': None,
                         'weight_representation_bytes': result.get('resources', {}).get('weight_representation_bytes'),
                         'model_tensor_bytes': result.get('resources', {}).get('model_tensor_bytes')})
    def save(name, values):
        if not values:
            return
        with atomic_destination(root/'tables'/name) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    save('per_case.csv', rows)
    keys = ('stage', 'family', 'dataset', 'variant', 'configuration', 'condition', 'format', 'endpoint', 'view')
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[k]) for k in keys)].append(row)
    aggregate = []
    metrics = ('f1', 'precision', 'recall', 'AUC_score', 'AP_score', 'nominal_false_positive_rate', 'event_coverage')
    for key, group in sorted(groups.items()):
        record = dict(zip(keys, key)); record['cases'] = len(group)
        for field in metrics:
            values = [r[field] for r in group if r[field] is not None]
            record[field+'_available'] = len(values)
            record[field+'_mean'] = statistics.mean(values) if values else None
            record[field+'_sample_std'] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(record)
    save('mean_std.csv', aggregate)
    paired = []
    index = {tuple(str(r[k]) for k in ('stage', 'dataset', 'variant', 'configuration', 'condition', 'format', 'seed', 'endpoint', 'view')): r for r in rows}
    for key, row in sorted(index.items()):
        comparisons = []
        if row['stage'] == 'analysis' and row['format'] != 'fp32':
            other = list(key); other[5] = 'fp32'; comparisons.append(('precision_vs_FP32', tuple(other)))
        if row['stage'] == 'train' and row['condition'] == 'hidden_current':
            other = list(key); other[4] = 'observed_current'; comparisons.append(('hidden_minus_observed', tuple(other)))
        if row['stage'] in ('analysis', 'train') and row['variant'] != 'reference':
            other = list(key); other[2] = 'reference'; comparisons.append(('variant_minus_reference', tuple(other)))
        if row['stage'] == 'analysis' and row['configuration'] != 'reference':
            other = list(key); other[3] = 'reference'; comparisons.append(('W_FF_profile_minus_reference', tuple(other)))
        for kind, other in comparisons:
            if other not in index:
                continue
            reference = index[other]
            value = {**row, 'comparison': kind}
            for field in metrics:
                value[field+'_difference'] = row[field]-reference[field] if row[field] is not None and reference[field] is not None else None
            paired.append(value)
    save('paired_comparisons.csv', paired)
    simple = []
    baselines = [r for r in rows if r['stage'] == 'baselines']
    for row in rows:
        if row['stage'] != 'train':
            continue
        for baseline in baselines:
            if any(row[k] != baseline[k] for k in ('dataset', 'configuration', 'endpoint', 'view')):
                continue
            value = {**row, 'baseline': baseline['variant'], 'baseline_is_deterministic': True}
            for field in metrics:
                value[field+'_minus_baseline'] = row[field]-baseline[field] if row[field] is not None and baseline[field] is not None else None
            simple.append(value)
    save('simple_detector_comparisons.csv', simple)
