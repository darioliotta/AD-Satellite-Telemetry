"""Compact endpoint evaluation: validation calibration and declared test oracle."""
from pathlib import Path
import time
import warnings
import numpy as np

from tranad_plus.core.common import write_json
from tranad_plus.core.evaluation.scoring import collect_outputs, reconstruction_scores
from tranad_plus.core.evaluation.metrics import fixed_metrics, continuous_metrics
from tranad_plus.campaigns.temporal.evaluation import oracle
from tranad_plus.campaigns.thresholds.metrics import point_metrics
from tranad_plus.campaigns.exploration.progress import scoring_bars


def evaluate(model, bundle, profile, folder, device, spec):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    alphas = np.linspace(0, 1, 21).tolist() if model.output_count == 2 else [None]
    grids, performance = {}, {}
    for split in ('train', 'validation', 'test'):
        with scoring_bars():
            value = collect_outputs(model, bundle.datasets[split], device,
                                    spec['evaluation'].get('batch_size_by_dataset', {}).get(profile['dataset'],
                                    spec['evaluation']['batch_size']), f'V19 SCORE {split}', folder/'nonfinite.json')
        grids[split] = np.stack([reconstruction_scores(value, a) for a in alphas])
        performance[split] = value['performance']
        del value
    # POT is fitted on nominal training scores and evaluated on validation here.
    # Neither checkpoint selection nor these alpha/threshold choices see test labels.
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter('always')
        if len(np.unique(bundle.labels['validation'])) == 2:
            calibration = oracle(grids['train'], grids['validation'], bundle.labels['validation'],
                                 alphas, profile['pot_q'], bundle.groups.get('validation'), folder,
                                 'V19 VALIDATION calibration | BF200/POT')
            calibration['selection'] = 'Alpha/threshold selected on validation; test labels unused'
            for method in ('bf', 'pot'):
                calibration[method]['selection_uses_test_labels'] = False
        else:
            # With no validation anomalies, do not pretend that F1 or alpha can
            # be selected by supervised detection. Fix alpha before evaluation.
            from opssat_pipeline import _pot_threshold
            alpha = spec['evaluation'].get('native_fixed_alpha', .5) if model.output_count == 2 else None
            row = alphas.index(alpha)
            threshold, error = _pot_threshold(grids['train'][row], grids['validation'][row], profile['pot_q'])
            calibration = {'bf': {'status': 'not_available', 'reason': 'Nominal-only validation cannot select detection F1'},
                           'pot': {'status': 'measured' if threshold is not None else 'not_available',
                                   'alpha': alpha, 'pot_threshold': threshold, 'reason': error,
                                   'selection_uses_test_labels': False},
                           'selection': 'Predeclared alpha; SPOT on training with nominal validation stream; no label optimization'}
        # Retain author-compatible metrics, explicitly separated from the honest
        # held-out calibrated endpoints above. No dense threshold optimization.
        paper = oracle(grids['train'], grids['test'], bundle.labels['test'], alphas,
                       profile['pot_q'], bundle.groups.get('test'), folder,
                       'V19 TEST oracle | BF200/POT')
    calibrated = {}
    for method in ('bf', 'pot'):
        endpoint = calibration[method]
        if endpoint.get(method+'_threshold') is None:
            calibrated[method] = {'status': 'not_available', 'reason': endpoint.get('reason')}
        else:
            scores = grids['test'][alphas.index(endpoint['alpha'])]
            threshold = endpoint[method+'_threshold']
            calibrated[method] = {
                **fixed_metrics(scores, bundle.labels['test'], threshold, method, bundle.groups.get('test')),
                'alpha': endpoint['alpha'], 'threshold': threshold,
                'selection_uses_test_labels': False,
                'raw_without_PA': point_metrics(bundle.labels['test'], scores > threshold),
                **continuous_metrics(scores, bundle.labels['test'])}
        endpoint = paper[method]
        if endpoint.get(method+'_f1') is not None:
            scores = grids['test'][alphas.index(endpoint['alpha'])]
            endpoint['raw_without_PA'] = point_metrics(bundle.labels['test'], scores > endpoint[method+'_threshold'])
            endpoint.update(continuous_metrics(scores, bundle.labels['test']))
    result = {'validation_calibration': calibration, 'test_validation_calibrated': calibrated,
              'test_paper_oracle': paper, 'inference_by_split': performance,
              'evaluation_seconds': time.perf_counter()-started,
              'fit_warnings': sorted({str(w.message) for w in warning_records}),
              'archive_policy': 'Scalar metrics/counts/alpha/threshold only; regenerate arrays from checkpoint',
              'label_granularity': 'weak fragment labels expanded to points' if bundle.groups.get('test') else 'native point labels',
              'AUC_PA': 'ROC-AUC on PA-adjusted binary predictions, author convention',
              'AUC_score': 'ROC-AUC on continuous reconstruction scores, explicitly separate',
              'raw_without_PA': 'Same selected alpha and threshold; no independent raw-F1 optimization'}
    write_json(folder/'result.json', result)
    return result
