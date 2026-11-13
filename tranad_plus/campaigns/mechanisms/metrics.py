"""Threshold-free separation, fixed calibration and unit-bounded event metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from tranad_plus.campaigns.thresholds.metrics import point_metrics, bf200


def separation(scores, labels, bins=50):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    if not len(s) or len(s) != len(y) or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Finite nonempty aligned scores and binary labels required')
    result = {}
    width = max(abs(float(s.min()))*1e-6, 1e-12)
    edges = np.linspace(float(s.min()), float(s.max()), bins+1) if s.min() != s.max() else np.linspace(float(s.min())-width, float(s.max())+width, bins+1)
    for name, mask in (('nominal', y == 0), ('anomalous', y == 1)):
        values = s[mask]
        result[name] = {'points': int(mask.sum()), 'mean': float(values.mean()) if len(values) else None,
                        'quantiles_05_25_50_75_95_99': np.quantile(values, [.05, .25, .5, .75, .95, .99]).tolist() if len(values) else None,
                        'histogram_counts': np.histogram(values, edges)[0].tolist()}
    result['histogram_edges'] = edges.tolist()
    result['AUC_score'] = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None
    result['AP_score'] = float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, s)
        precision, recall, _ = precision_recall_curve(y, s)
        def sampled(a, b):
            index = np.unique(np.linspace(0, len(a)-1, min(101, len(a)), dtype=int))
            return {'x': a[index].tolist(), 'y': b[index].tolist()}
        result['roc_plot'] = sampled(fpr, tpr)
        result['pr_plot'] = sampled(recall, precision)
    result['curve_note'] = 'AUC_score/AP_score computed from all scores; only plotting coordinates subsampled. AP is average precision, not trapezoidal PR area.'
    return result


def events(labels, units):
    """Contiguous positive runs, split at raw track/fragment boundaries."""
    y, ids = np.asarray(labels).astype(bool), np.asarray(units)
    if len(y) != len(ids):
        raise ValueError('Unit/label alignment differs')
    starts = np.flatnonzero(y & np.r_[True, (~y[:-1]) | (ids[1:] != ids[:-1])])
    ends = np.flatnonzero(y & np.r_[(~y[1:]) | (ids[1:] != ids[:-1]), True])+1
    return list(zip(starts.tolist(), ends.tolist()))


def lifted_scores(scores, labels, units):
    """Equivalent to PA at every threshold, without propagating across units."""
    lifted = np.asarray(scores).copy()
    for start, stop in events(labels, units):
        lifted[start:stop] = np.max(lifted[start:stop])
    return lifted


def endpoint(scores, labels, units, threshold):
    s, y = np.asarray(scores), np.asarray(labels)
    direct = s > threshold
    adjusted = lifted_scores(s, y, units) > threshold
    spans = events(y, units)
    hits, positions, fractions = 0, [], []
    for start, stop in spans:
        found = np.flatnonzero(direct[start:stop])
        fractions.append(float(len(found)/(stop-start)))
        if len(found):
            hits += 1
            positions.append(float(found[0]/(stop-start)))
    raw, pa = point_metrics(y, direct), point_metrics(y, adjusted)
    return {'threshold': float(threshold), 'raw': raw, 'PA': pa,
            'nominal_false_positive_rate': float(direct[y == 0].mean()) if np.any(y == 0) else None,
            'event_count': len(spans), 'events_detected': hits,
            'event_coverage': hits/len(spans) if spans else None,
            'mean_fraction_directly_detected_within_events': float(np.mean(fractions)) if fractions else None,
            'mean_first_alarm_fraction_in_detected_event': float(np.mean(positions)) if positions else None,
            'event_note': 'OPS-SAT events are weak labelled fragments; first-alarm position is not a physical fault-onset latency.',
            'PA_definition': 'Positive contiguous annotation run, bounded by each raw unit; same direct decisions before/after PA'}


def bf200_pa(scores, labels, units):
    # Use the same raw-score grid as BF200. PA only changes positive-score counts.
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    grid = np.linspace(float(s.min()), float(s.mean()+3*s.std()), 200)
    lifted = lifted_scores(s, y, units)
    tp = int(y.sum())-np.searchsorted(np.sort(lifted[y]), grid, side='right')
    fp = int((~y).sum())-np.searchsorted(np.sort(s[~y]), grid, side='right')
    den = int(y.sum())+tp+fp
    f1 = np.divide(2*tp, den, out=np.zeros(200), where=den > 0)
    i = int(np.argmax(f1))
    return {'threshold': float(grid[i]), 'threshold_index': i, 'threshold_steps': 200}


def segment_scores(scores, layout, q):
    ids, values, labels = [], [], []
    for unit in np.unique(layout.units):
        mask = layout.units == unit
        y = np.unique(layout.labels[mask])
        if len(y) != 1:
            raise ValueError('Segment classification requires one weak label per fragment')
        ids.append(int(unit)); values.append(float(np.quantile(scores[mask], q))); labels.append(int(y[0]))
    return np.array(ids), np.array(values), np.array(labels, dtype=np.int8)


def evaluate_scores(scores_by_split, layouts, dataset, settings, calibration_source):
    """No test labels in primary calibration; test oracles are separate diagnostics."""
    arrays = {k: np.asarray(v)[layouts[k].real] for k, v in scores_by_split.items()}
    meta = {k: layouts[k].selected() for k in arrays}
    calibration_split = 'validation' if 'validation' in arrays else 'train'
    normal = arrays[calibration_split][meta[calibration_split].labels == 0]
    if not len(normal):
        raise ValueError('No nominal calibration scores')
    threshold = float(np.quantile(normal, settings['nominal_calibration_quantile']))
    test, labels, ids = arrays['test'], meta['test'].labels, meta['test'].units
    report = {'calibration': {'source': calibration_source, 'split': calibration_split,
                             'quantile': settings['nominal_calibration_quantile'], 'threshold': threshold,
                             'actual_nominal_alarm_fraction': float(np.mean(normal > threshold)),
                             'selection_uses_test_labels': False, 'strict_comparison': 'score > threshold'},
              'separation': separation(test, labels, settings['histogram_bins']),
              'fixed_nominal_quantile': endpoint(test, labels, ids, threshold),
              'all_positive_baseline': point_metrics(labels, np.ones(len(labels), dtype=np.int8)),
              'split_score_summaries': {k: separation(v, meta[k].labels, settings['histogram_bins']) for k, v in arrays.items()},
              'sample_counts': {k: {'original': len(scores_by_split[k]), 'real': len(v),
                                   'excluded_padded_targets': int((~layouts[k].real).sum())} for k, v in arrays.items()},
              'AUC_names': {'AUC_score': 'ROC-AUC on continuous scores', 'raw.auc_binary': 'AUC-bin', 'PA.auc_binary': 'AUC-PA'},
              'fixed_alpha_note': 'Diagnostic control; no alpha selected from test labels. Not a reproduction of the 21-alpha paper oracle.'}
    # Retain both BF200 objectives explicitly, never a dense best-F1 sweep.
    raw_bf = bf200(test, labels)
    pa_bf = bf200_pa(test, labels, ids)
    report['BF200_raw_oracle_fixed_alpha'] = endpoint(test, labels, ids, raw_bf['threshold'])
    report['BF200_PA_oracle_fixed_alpha'] = endpoint(test, labels, ids, pa_bf['threshold'])
    report['oracle_note'] = 'These two BF200 thresholds use test labels and are supplementary, fixed-alpha oracle diagnostics.'
    # Tail fit sees only nominal optimization/reference scores and nominal stream.
    from opssat_pipeline import _pot_threshold
    train_normal = arrays['train'][meta['train'].labels == 0]
    pot, error = _pot_threshold(train_normal, normal, settings.get('pot_q', .01))
    report['POT_fixed_alpha'] = (endpoint(test, labels, ids, pot) if pot is not None else {'status': 'not_available', 'reason': error})
    segment_arrays = None
    if dataset == 'OPS-SAT':
        if 'validation' not in arrays:
            raise ValueError('OPS-SAT segment calibration requires separate validation')
        vi, vs, vy = segment_scores(arrays['validation'], meta['validation'], settings['segment_quantile'])
        ti, ts, ty = segment_scores(test, meta['test'], settings['segment_quantile'])
        if len(np.unique(vy)) != 2:
            raise ValueError('OPS-SAT segment validation requires both classes')
        chosen = bf200(vs, vy)
        segment_threshold = chosen['threshold']
        report['fragment_classification'] = {
            'aggregation': f"fixed_percentile_{100*settings['segment_quantile']:g}",
            'threshold': segment_threshold, 'selection': 'BF200 F1 on validation fragments; alpha/aggregation fixed in advance',
            'selection_uses_test_labels': False, 'point_adjustment': False,
            'test': point_metrics(ty, ts > segment_threshold),
            'separation': separation(ts, ty, settings['histogram_bins'])}
        segment_arrays = {'validation_ids': vi, 'validation_scores': vs, 'validation_labels': vy,
                          'test_ids': ti, 'test_scores': ts, 'test_labels': ty}
    else:
        report['fragment_classification'] = {'status': 'not_applicable', 'reason': 'Native point/event labels; do not relabel whole tracks as anomalous'}
    return report, segment_arrays
