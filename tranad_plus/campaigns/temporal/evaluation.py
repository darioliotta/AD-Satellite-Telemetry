"""BF200/POT with author AUC-PA; no dense threshold search."""
from pathlib import Path
import numpy as np

from tranad_plus.core.common import save_npz, write_json
from tranad_plus.core.evaluation.metrics import fixed_metrics, bounded_pot
from tranad_plus.core.evaluation.scoring import collect_outputs, reconstruction_scores
from tranad_plus.campaigns.exploration.metrics import real_point_mask
from .controls import Windows
from .progress import bar, record, alpha_search
from paper_metrics import _stats, paper_best_f1
from opssat_pipeline import _pot_threshold, point_metrics


def bf200(scores, labels, groups=None):
    if groups is None:
        return paper_best_f1(scores, labels, steps=200)
    thresholds = np.linspace(float(scores.min()), float(scores.mean()+3*scores.std()), 200)
    boundaries = np.cumsum([len(g['values']) for g in groups])[:-1]
    statistics = _stats(scores, labels, thresholds, boundaries=boundaries,
                        preserve_upstream_index_zero=False)
    threshold = float(thresholds[int(np.argmax(statistics['f1']))])
    return fixed_metrics(scores, labels, threshold, 'bf', groups)


def oracle(train_grid, test_grid, labels, alphas, q, groups, folder, title):
    best = {'bf': None, 'pot': None}
    records = []
    with alpha_search(folder, len(alphas), title) as advance:
        for index, alpha in enumerate(alphas):
            tr, te = train_grid[index], test_grid[index]
            if not np.isfinite(tr).all() or not np.isfinite(te).all():
                raise FloatingPointError('Nonfinite scores: do not mask or silently drop samples')
            bf = bf200(te, labels, groups)
            if groups is None:
                pot = bounded_pot(tr, te, labels, q, max_attempts=35)
            else:
                threshold, error = _pot_threshold(tr, te, q)
                pot = ({**fixed_metrics(te, labels, threshold, 'pot', groups), 'status': 'measured'}
                       if threshold is not None else {'status': 'not_available', 'pot_f1': None, 'reason': error})
            for key, item in (('bf', bf), ('pot', pot)):
                item.update(alpha=alpha, alpha_applicable=alpha is not None,
                            beta=None if alpha is None else 1-alpha,
                            selection_uses_test_labels=True,
                            point_adjustment='fragment_aware' if groups else 'upstream_global')
                value = item.get(key+'_f1')
                old = best[key]
                if old is None or (value is not None and (old.get(key+'_f1') is None or value > old[key+'_f1'])):
                    best[key] = item
            records.append({'alpha': alpha, 'bf_f1': bf['bf_f1'], 'pot_f1': pot.get('pot_f1'),
                            'pot_threshold': pot.get('pot_threshold'), 'bf_threshold': bf['bf_threshold']})
            advance(index, alpha, bf['bf_f1'], pot.get('pot_f1'))
    return {**best, 'alpha_grid': records, 'auc_name': 'AUC-PA',
            'auc_definition': 'ROC-AUC of binary predictions after point adjustment (balanced accuracy)',
            'bf_definition': '200 thresholds, min(score) to mean(score)+3*std(score)',
            'selection': 'Test-label oracle, matching the benchmark; not a deployable calibration estimate'}


def errors_at(outputs, alpha):
    values = outputs['outputs']
    reconstructed = values[0] if alpha is None else alpha*values[0]+(1-alpha)*values[1]
    return (reconstructed-outputs['target'])**2


def error_summary(errors, labels):
    result = {}
    for name, mask in (('nominal', labels == 0), ('anomalous', labels == 1)):
        subset = errors[mask]
        result[name] = {'points': int(mask.sum()),
                        'mean_by_feature': subset.mean(axis=0).astype(float).tolist() if len(subset) else None,
                        'mse_quantiles': np.quantile(subset.mean(axis=1), [0, .25, .5, .75, .95, .99, 1]).tolist()
                        if len(subset) else None}
    return result


def metadata(bundle, profile, data_root):
    arrays = {f'{key}_labels': value for key, value in bundle.labels.items()}
    arrays['test_real_point_mask'] = real_point_mask(profile, data_root, len(bundle.labels['test']))
    for key, groups in bundle.groups.items():
        arrays[f'{key}_segment_ids'] = np.concatenate([np.repeat(g['id'], len(g['values'])) for g in groups])
        arrays[f'{key}_channels'] = np.concatenate([np.repeat(g['channel'], len(g['values'])) for g in groups])
    return arrays


def save_outputs(path, outputs, arrays=None):
    arrays = dict(arrays or {})
    for split, value in outputs.items():
        arrays[split+'_target'] = value['target']
        for i, output in enumerate(value['outputs'], 1):
            arrays[f'{split}_output_{i}'] = output
    save_npz(path, **arrays)


def trained_evaluation(model, bundle, folder, profile, device, settings, data_root):
    """Evaluate matched inputs, then ordered test inputs for trained interventions."""
    from .controls import changed_bundle
    condition = profile['input_condition']
    matched = changed_bundle(bundle, condition)
    batch = settings['score_batch_size_by_dataset'][profile['dataset']]
    outputs = {key: collect_outputs(model, ds, device, batch, f"V15 {key} {condition}",
                                   folder / 'numerical_diagnostics.json') for key, ds in matched.datasets.items()}
    alphas = np.linspace(0, 1, 21).tolist() if model.output_count == 2 else [None]
    grids = {key: np.stack([reconstruction_scores(value, a) for a in alphas]) for key, value in outputs.items()}
    info = metadata(bundle, profile, data_root)
    save_outputs(folder / 'outputs_matched.npz', outputs, info)
    save_npz(folder / 'scores_matched.npz', **info, **{k+'_scores': v for k, v in grids.items()},
             alpha_values=np.array([a for a in alphas if a is not None]))
    groups = bundle.groups.get('test')
    report = {'matched': oracle(grids['train'], grids['test'], bundle.labels['test'], alphas,
                               profile['pot_q'], groups, folder, 'V15 BF200/POT | matched')}
    test_outputs = {'matched': outputs['test']}
    if condition != 'original':
        ordered = collect_outputs(model, bundle.datasets['test'], device, batch, 'V15 test original',
                                  folder / 'numerical_diagnostics.json')
        ordered_grid = np.stack([reconstruction_scores(ordered, a) for a in alphas])
        save_outputs(folder / 'outputs_ordered_test.npz', {'test': ordered})
        save_npz(folder / 'scores_ordered_test.npz', test_scores=ordered_grid,
                 test_labels=bundle.labels['test'], alpha_values=np.array([a for a in alphas if a is not None]))
        report['ordered_test'] = oracle(grids['train'], ordered_grid, bundle.labels['test'], alphas,
                                        profile['pot_q'], groups, folder, 'V15 BF200/POT | ordered test')
        test_outputs['ordered_test'] = ordered
    for test_condition, values in test_outputs.items():
        for method in ('bf', 'pot'):
            best = report[test_condition][method]
            if best.get(method+'_f1') is not None:
                errors = errors_at(values, best['alpha'])
                best['reconstruction_error_by_class'] = error_summary(errors, bundle.labels['test'])
                best['raw_without_PA'] = point_metrics(bundle.labels['test'],
                                                        errors.mean(axis=1) > best[method+'_threshold'], method)
    report['note'] = 'Train inputs match the declared intervention; ordered_test changes test inputs only. All targets unchanged.'
    return report


def frozen_evaluation(model, bundle, parent, parent_report, folder, profile, device, spec, data_root):
    """Parent-selected alpha and thresholds remain locked for all interventions."""
    settings, conditions = spec['evaluation'], spec['frozen_conditions']
    batch = settings['score_batch_size_by_dataset'][profile['dataset']]
    labels, groups = bundle.labels['test'], bundle.groups.get('test')
    save_npz(folder / 'metadata.npz', **metadata(bundle, profile, data_root))
    results = []
    clean_outputs = collect_outputs(model, bundle.datasets['test'], device, batch, 'V15 frozen original',
                                    folder / 'numerical_diagnostics.json')
    with np.load(parent / 'scores.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['test_labels'], labels)
        for key in ('bf', 'pot'):
            endpoint = parent_report['paper_oracle'][key]
            if endpoint is None or endpoint.get(key+'_f1') is None:
                raise ValueError(f'Parent {key} endpoint unavailable')
            alpha = endpoint['alpha']
            row = int(np.argmin(np.abs(saved['alpha_values']-alpha))) if alpha is not None else 0
            np.testing.assert_allclose(reconstruction_scores(clean_outputs, alpha), saved['test_scores'][row],
                                       rtol=5e-4, atol=1e-7)
    for i, condition in enumerate(bar(conditions, total=len(conditions), position=1,
                                    desc='V15 frozen | history conditions', unit='condition', leave=False), 1):
        record(folder, 'frozen_inference', condition=condition, condition_index=i, conditions=len(conditions),
               identifier=profile['identifier'])
        output_path = folder / f'{condition}.npz'
        # Inference is deterministic in eval mode. A completed condition is reused
        # after restart only after its file hash has been checked.
        from tranad_plus.core.common import hash_file
        condition_report = folder / f'{condition}.json'
        if condition_report.exists():
            import json
            saved_condition = json.loads(condition_report.read_text())
            if hash_file(output_path) != saved_condition['scores_sha256']:
                raise RuntimeError(f'Changed condition scores: {output_path}')
            results.append(saved_condition)
            continue
        outputs = clean_outputs if condition == 'original' else collect_outputs(
            model, Windows(bundle.datasets['test'], condition), device, batch, f'V15 frozen {condition}',
            folder / 'numerical_diagnostics.json')
        arrays, measured = {}, {}
        for method in ('bf', 'pot'):
            original = parent_report['paper_oracle'][method]
            alpha, threshold = original['alpha'], original[method+'_threshold']
            error_key = 'squared_errors_single_output' if alpha is None else f'squared_errors_alpha_{round(alpha*20):02d}'
            if error_key not in arrays:
                arrays[error_key] = errors_at(outputs, alpha)
            errors = arrays[error_key]
            scores = errors.mean(axis=1)
            arrays[method+'_scores'] = scores
            arrays[method+'_direct_alarms'] = (scores > threshold).astype(np.int8)
            measured[method] = {**fixed_metrics(scores, labels, threshold, method, groups),
                                'alpha': alpha, 'alpha_and_threshold_locked_to_parent': True,
                                'squared_errors_by_feature_array': error_key,
                                'raw_without_PA': point_metrics(labels, scores > threshold, method),
                                'reconstruction_error_by_class': error_summary(errors, labels)}
            if condition == 'original':
                # Counts are more sensitive than allclose(score): verify the
                # actual baseline conclusions before interpreting interventions.
                for stat in ('TP', 'TN', 'FP', 'FN'):
                    if measured[method][method+'_'+stat] != original[method+'_'+stat]:
                        raise RuntimeError(f'Original {method} prediction parity failed: {stat}')
        save_npz(output_path, **arrays)
        result = {'condition': condition, 'methods': measured, 'scores_sha256': hash_file(output_path),
                  'inference': outputs['performance']}
        write_json(condition_report, result)
        results.append(result)
    return {'conditions': results, 'auc_name': 'AUC-PA', 'thresholds': 'locked BF200/POT parent endpoints',
            'interpretation': 'Frozen interventions measure sensitivity, including distribution shift; training controls test adaptation.',
            'permutation_seeds_are_not_training_replicates': True}
