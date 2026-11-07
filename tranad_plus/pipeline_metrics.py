"""Automatic, in-project measurement after each TranAD+ run.

Quantities from real FPGA synthesis/on-board instruments must be imported with
provenance. CUDA measures are PyTorch allocator and model-forward only, not device
power or onboard latency. The author's paper evaluation is kept separate.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from config import NATIVE_PACKAGE, RESULTS_DIR, PROCESSED_DATA_DIR
from observability.measure_experiment import (
    architecture_statistics, benchmark, build_from_checkpoint,
    hash_file, hardware_status, load_hardware_report, utc_now,
    write_json_atomic,
)
from observability.export_anomaly_scores import pair_paths

SCHEMA_VERSION = 'thesis-integrated-v3'


def measurement_key(settings, device_name):
    # Avoid confusing results from two cuda:0 GPUs on DIFFERENT hosts or model-code revisions.
    device = torch.device(device_name)
    identity = {
        'device_name': device_name, 'hostname': platform.node(),
        'processor': platform.processor(), 'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
    }
    root = Path(__file__).resolve().parent
    source_files = [root / 'pipeline_metrics.py', root / 'observability' / 'measure_experiment.py',
                    root / 'observability' / 'export_anomaly_scores.py',
                    root / 'TranADPlus' / 'src' / 'models.py', root / 'ablation.py',
                    root / 'opssat_pipeline.py', root / 'opssat_scoring.py']
    h = hashlib.sha256()
    h.update(json.dumps({'settings': settings, 'identity': identity}, sort_keys=True).encode())
    for file in source_files:
        h.update(file.read_bytes())
    return h.hexdigest()[:10], identity


def epoch_history(dataset, identifier):
    path = RESULTS_DIR / dataset / f'{identifier}_epochs.jsonl'
    if not path.exists():
        return {'status': 'not_available', 'records': [], 'reason': 'Run predates v3 per-epoch telemetry or no training was run.'}
    records = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        index = int(entry['epoch_one_based'])
        if index in records:
            # Resume may replay an epoch. The last complete durable record wins.
            pass
        records[index] = entry
    rows = [records[k] for k in sorted(records)]
    return {'status': 'measured', 'records': rows, 'epochs_logged': len(rows),
            'sum_epoch_seconds': sum(float(e['elapsed_seconds']) for e in rows)}


def _raw_quality(scores, labels, threshold):
    if len(scores) != len(labels) or len(scores) == 0:
        raise ValueError('Scores and labels are empty/misaligned')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite scores')
    pred = np.asarray(scores > threshold, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    valid_auc = len(np.unique(labels)) == 2
    return {
        'status': 'measured', 'protocol': 'unadjusted_pointwise_scores_fixed_train_quantile_threshold',
        'n_points': int(len(labels)), 'n_anomalies': int(labels.sum()),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'precision': float(precision_score(labels, pred, zero_division=0)),
        'recall': float(recall_score(labels, pred, zero_division=0)),
        'f1': float(f1_score(labels, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if tn + fp else None,
        'accuracy': float(accuracy_score(labels, pred)),
        'roc_auc_continuous_scores': float(roc_auc_score(labels, scores)) if valid_auc else None,
        'average_precision_continuous_scores': float(average_precision_score(labels, scores)) if valid_auc else None,
        'auc_note': None if valid_auc else 'Undefined: evaluation labels have only one class.',
        'point_adjustment': False,
        'threshold': float(threshold),
        'threshold_comparison': 'score > threshold',
    }


def _scores(model, dataset, device, batch_size, alpha):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    from gpu_batches import maybe_cuda_batches
    resident = maybe_cuda_batches(dataset,batch_size=batch_size,shuffle=False,device=device)
    batches = resident if resident is not None else loader
    consumed=0
    chunk_scores = []
    chunk_labels = []
    with torch.inference_mode():
        for batch in batches:
            if resident is not None:
                windows=batch
                begin=consumed + dataset.window_size - 1 if dataset.padding else consumed
                labels=dataset.big_anom[begin:begin+len(windows)].unsqueeze(1)
            else:
                windows,labels=batch
            consumed+=len(windows)
            windows = windows.to(device, dtype=torch.float32).permute(1, 0, 2)
            target = windows[-1:, :, :]
            first, second = model(windows, target)
            combined = alpha * first + (1-alpha) * second
            loss = ((combined - target) ** 2).mean(dim=-1)[0]
            chunk_scores.append(loss.cpu().numpy())
            label = labels[:, -1, :] if dataset.padding else labels[:, 0, :]
            chunk_labels.append((label.sum(dim=1) > 0).numpy().astype(np.int8))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    count = sum(len(chunk) for chunk in chunk_scores)
    perf = {
        'elapsed_seconds_including_input_transfer_and_scoring': float(elapsed),
        'windows_scored': int(count),
        'windows_per_second': float(count / elapsed) if elapsed else None,
        'batch_size': int(batch_size),
        'gpu_peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None,
        'gpu_peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else None,
    }
    return np.concatenate(chunk_scores).astype(np.float32), np.concatenate(chunk_labels).astype(np.int8), perf


def compute_raw_scores(model, hp, dataset_name, device, settings, report_dir, identifier, checkpoint_digest):
    from TranADPlus.src.datasets import TSDataset_tracks
    if hp.get('downsample') is not None:
        return {'status': 'not_available', 'reason': 'Native downsample protocol not validated for independent score exporter.'}
    if hp.get('scaling') not in (True, 'True', None):
        return {'status': 'not_available', 'reason': 'Native non-MinMax scaling not validated for independent score exporter.'}
    base = PROCESSED_DATA_DIR / dataset_name
    train_t, train_l, train_ids = pair_paths(base, 'Train')
    test_t, test_l, test_ids = pair_paths(base, 'Test')
    ws = int(hp['window_sz'])
    train_set = TSDataset_tracks(track_paths=train_t, anom_paths=train_l, window_size=ws,
                                 scaling=True, padding=bool(hp.get('padding', True)))
    test_set = TSDataset_tracks(track_paths=test_t, anom_paths=test_l, window_size=ws,
                                scaling=True, scaler=train_set.scaler, padding=bool(hp.get('padding', True)))
    if int(train_set.big_track.shape[-1]) != int(hp['features']) or int(test_set.big_track.shape[-1]) != int(hp['features']):
        raise ValueError('Dataset features differ from trained checkpoint')
    alpha = float(settings['alpha'])
    environment_token, _ = measurement_key(settings, str(device))
    train_scores, _, train_performance = _scores(model, train_set, device, int(settings['score_batch_size']), alpha)
    test_scores, test_labels, test_performance = _scores(model, test_set, device, int(settings['score_batch_size']), alpha)
    quantile = float(settings['threshold_train_quantile'])
    threshold = float(np.quantile(train_scores, quantile))
    metrics = _raw_quality(test_scores, test_labels, threshold)
    metrics.update({
        'alpha_fixed_pre_test': alpha,
        'beta_fixed_pre_test': 1.0-alpha,
        'threshold_source': f'training_reconstruction_score_quantile_{quantile}',
        'threshold_quantile': quantile,
        'scoring_performance': {'train': train_performance, 'test': test_performance},
        'training_scores': {'count': int(len(train_scores)), 'mean': float(train_scores.mean()),
                            'std': float(train_scores.std()), 'max': float(train_scores.max())},
        'score_formula': 'mean((alpha*decoder1+(1-alpha)*decoder2-target)^2, axis=features)',
        'train_track_ids': train_ids, 'test_track_ids': test_ids,
        'label_protocol': 'window target label (any feature)',
        'limitations': [
            'Training-score-quantile is a fixed, unsupervised heuristic; calibrate with separate validation before scientific conclusions.',
            'Upstream TSDataset_tracks merges padded tracks; this reproduces its boundary behavior, not an improved one.',
            'Original paper POT/BF metrics may use point adjustment and test-label-dependent threshold or alpha search.',
        ],
    })
    if settings['save_raw_scores']:
        filename = f'{identifier}_scores_{checkpoint_digest[:12]}_{environment_token}_alpha{alpha:.3f}.npz'
        file = report_dir / filename
        if file.exists():
            with np.load(file, allow_pickle=False) as old:
                if not (np.array_equal(old['scores'], test_scores) and np.array_equal(old['labels'], test_labels)):
                    raise RuntimeError(f'Existing raw-score artifact differs: {file}')
        else:
            fd, tmp = tempfile.mkstemp(dir=report_dir, prefix='.raw_', suffix='.npz')
            os.close(fd)
            try:
                np.savez_compressed(tmp, scores=test_scores, labels=test_labels,
                                    train_scores=train_scores,
                                    fixed_threshold=np.array([threshold], dtype=np.float64))
                os.replace(tmp, file)
            finally:
                Path(tmp).unlink(missing_ok=True)
        metrics['scores_file'] = file.name
        metrics['scores_sha256'] = hash_file(file)
    else:
        metrics['scores_file'] = None
    return metrics


def _cuda_device_details(device):
    if device.type != 'cuda':
        return {'status': 'not_applicable'}
    p = torch.cuda.get_device_properties(device)
    return {'status': 'measured', 'name': p.name, 'total_memory_bytes': int(p.total_memory),
            'torch_cuda_version': torch.version.cuda, 'capability': list(torch.cuda.get_device_capability(device))}


def latest_matching_evaluation(report_dir, checkpoint_hash):
    for path in sorted(report_dir.glob('*_evaluation_*.json'), reverse=True):
        try:
            val = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if val.get('checkpoint', {}).get('sha256') == checkpoint_hash and isinstance(val.get('results'), dict):
            return {'source': path.name, 'metrics': val['results'],
                    'caution': 'Original POT/Best F1 metrics include point adjustment; Best F1 searches thresholds on test labels.'}
    return {'status': 'not_available', 'reason': 'Matching completed native evaluation not archived.'}


def measure_profile(profile, settings, device_name, *, code_digest=None):
    dataset, identifier = profile['dataset'], profile['identifier']
    report_dir = RESULTS_DIR / dataset
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = report_dir / f'{identifier}.ckpt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Final checkpoint not archived: {checkpoint}')
    device = torch.device(device_name)
    model, meta, ckpt_type = build_from_checkpoint(
        checkpoint, NATIVE_PACKAGE / 'src' / 'models.py', {})
    if ckpt_type != 'final':
        raise RuntimeError('Integrated final report requires a final checkpoint')
    checkpoint_digest = hash_file(checkpoint)
    model_stats = architecture_statistics(model)
    from gpu_settings import configure
    configure()
    model = model.to(device).eval()
    rng = np.random.default_rng(841)
    window = torch.tensor(rng.random((meta['window_sz'], 1, meta['features'])), dtype=torch.float32)
    # Synthetic inputs are only for reproducible model-only latency measurements.
    inference = benchmark(model, device, window, int(settings['warmup']), int(settings['iterations']))
    try:
        raw = compute_raw_scores(model, meta['hyperparameters'], dataset, device,
                                 settings, report_dir, identifier, checkpoint_digest)
    finally:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    artifact = None
    if settings.get('hardware_json_dir'):
        hardware_path = Path(settings['hardware_json_dir']) / f'{identifier}.json'
        if hardware_path.is_file():
            artifact = hardware_path
    hardware = load_hardware_report(artifact)
    settings_digest, environment_identity = measurement_key(settings, device_name)
    from ablation import parse_ablation
    stage1 = (
        __import__('ablation').evaluate_native_stage1(model,meta,dataset,device,settings,report_dir,identifier)
        if settings.get('inference_ablation',False) else
        {'status':'not_requested'}
    )
    report = {
        'ablation_design':parse_ablation(profile), 'ablation_inference':stage1,
        'measurement_settings': settings, 'settings_digest': settings_digest,
        'execution_identity': environment_identity,
        'schema_version': SCHEMA_VERSION, 'created_at': utc_now(),
        'dataset': dataset, 'identifier': identifier, 'profile': profile,
        'checkpoint': {'file': checkpoint.name, 'sha256': checkpoint_digest,
                       'size_bytes': checkpoint.stat().st_size},
        'source': {'models_py': 'TranADPlus/src/models.py',
                   'models_sha256': hash_file(NATIVE_PACKAGE / 'src' / 'models.py'),
                   'runner_sha256': code_digest},
        'architecture': meta, 'model_statistics': model_stats,
        'inference': inference,
        'training': epoch_history(dataset, identifier),
        'original_paper_evaluation': latest_matching_evaluation(report_dir, checkpoint_digest),
        'raw_pointwise_evaluation': raw,
        'hardware_acceleration': hardware,
        'environment': {'device': device_name, 'torch': torch.__version__,
                        'cuda': _cuda_device_details(device)},
        'methodology': {
            'inference': 'batch-1 sequential model.forward() after warmup, no I/O/transfer/postprocessing',
            'quality': 'Raw point-wise scores without adjustment; training-quantile heuristic fixed without test labels',
            'hardware': 'FPGA utilization requires actual synthesis, not inference on a CUDA GPU',
        },
    }
    file = report_dir / f'{identifier}_integrated_{checkpoint_digest[:12]}_{device_name.replace(":", "-")}_{settings_digest}.json'
    write_json_atomic(file, report)
    update_summary_csv(RESULTS_DIR / 'all_experiments.csv')
    return file


def update_summary_csv(path):
    """Lossless detailed records stay in per-run JSON; this is the comparable wide index."""
    rows = []
    for file in sorted(RESULTS_DIR.glob('*/*_integrated_*.json')):
        d = json.loads(file.read_text(encoding='utf-8'))
        raw = d.get('raw_pointwise_evaluation') or {}
        infer = d.get('inference') or {}
        gpu = infer.get('gpu_memory_bytes') or {}
        orig = (d.get('original_paper_evaluation') or {}).get('metrics') or {}
        h = d.get('hardware_acceleration') or {}
        hdata = h.get('user_supplied_hardware_report') or h
        synth = hdata.get('synthesis') or {}
        quant = hdata.get('quantization') or {}
        onboard = hdata.get('onboard_measurement') or hdata.get('onboard_measured') or {}
        train = d.get('training') or {}
        scoring = raw.get('scoring_performance') or {}
        conf = raw.get('confusion_matrix') or {}
        rows.append({
            'dataset': d['dataset'], 'identifier': d['identifier'],
            'device': d['environment']['device'],
            'gpu_device_name': (d['environment'].get('cuda') or {}).get('name'),
            'checkpoint_sha256': d['checkpoint']['sha256'],
            'parameters': d['model_statistics']['parameters_total'],
            'estimated_macs_per_window': ((d['model_statistics'].get('compute_estimate') or {}).get('macs_per_window_batch1')),
            'estimated_matmul_flops_per_window': ((d['model_statistics'].get('compute_estimate') or {}).get('matmul_flops_per_window_batch1')),
            'trainable_weights_bytes': d['model_statistics']['trainable_weight_bytes'],
            'model_state_bytes': d['model_statistics']['model_state_including_buffers_bytes'],
            'checkpoint_bytes': d['checkpoint']['size_bytes'],
            'sparsity': d['model_statistics']['unstructured_sparsity_fraction'],
            'training_epochs_recorded': train.get('epochs_logged'),
            'training_epoch_seconds_recorded': train.get('sum_epoch_seconds'),
            'max_cpu_rss_training_bytes': max((r['process_peak_rss_bytes'] for r in train.get('records', [])
                if r.get('process_peak_rss_bytes') is not None), default=None),
            'max_gpu_allocated_training_bytes': max((r['gpu_peak_allocated_bytes'] for r in train.get('records', [])
                if r.get('gpu_peak_allocated_bytes') is not None), default=None),
            'latency_mean_ms': infer.get('latency_ms', {}).get('mean'),
            'latency_p50_ms': infer.get('latency_ms', {}).get('p50'),
            'latency_p95_ms': infer.get('latency_ms', {}).get('p95'),
            'latency_p99_ms': infer.get('latency_ms', {}).get('p99'),
            'throughput_model_forward_windows_s': infer.get('throughput_windows_per_second_from_mean_latency'),
            'gpu_peak_forward_allocated_bytes': gpu.get('peak_allocated'),
            'gpu_peak_forward_reserved_bytes': gpu.get('peak_reserved'),
            'score_test_elapsed_seconds': (scoring.get('test') or {}).get('elapsed_seconds_including_input_transfer_and_scoring'),
            'score_test_throughput_windows_s': (scoring.get('test') or {}).get('windows_per_second'),
            'gpu_peak_scoring_allocated_bytes': (scoring.get('test') or {}).get('gpu_peak_allocated_bytes'),
            'raw_threshold': raw.get('threshold'),
            'raw_threshold_source': raw.get('threshold_source'),
            'raw_fixed_alpha': raw.get('alpha_fixed_pre_test'),
            'raw_tp': conf.get('tp'), 'raw_fp': conf.get('fp'),
            'raw_tn': conf.get('tn'), 'raw_fn': conf.get('fn'),
            'f1_raw': raw.get('f1'), 'precision_raw': raw.get('precision'),
            'recall_raw': raw.get('recall'),
            'roc_auc_raw': raw.get('roc_auc_continuous_scores'),
            'average_precision_raw': raw.get('average_precision_continuous_scores'),
            'pot_f1_original': (orig.get('pot') or {}).get('pot_f1'),
            'best_f1_original': (orig.get('bf') or {}).get('bf_f1'),
            'best_f1_dense_oracle': (orig.get('bf') or {}).get('bf_dense_f1'),
            'bf_alpha_original': (orig.get('bf') or {}).get('alpha'),
            'pot_alpha_original': (orig.get('pot') or {}).get('alpha'),
            'bf_dense_alpha_oracle': (orig.get('bf') or {}).get('bf_dense_alpha'),
            'hardware_status': hdata.get('status'),
            'lut': synth.get('lut'), 'ff': synth.get('ff'),
            'dsp': synth.get('dsp'), 'bram': synth.get('bram'),
            'clock_period_ns': synth.get('clock_period_ns') or synth.get('clock_ns'),
            'latency_cycles_min': synth.get('latency_min_cycles') or synth.get('latency_cycles'),
            'latency_cycles_max': synth.get('latency_max_cycles'),
            'initiation_interval_cycles': synth.get('initiation_interval_cycles'),
            'ebops': quant.get('ebops'),
            'onboard_latency_ms_p50': onboard.get('inference_latency_ms_p50') or onboard.get('latency_ms'),
            'onboard_power_w': onboard.get('power_w'),
            'onboard_energy_mj_per_window': onboard.get('energy_mj_per_window'),
            'settings_digest':d.get('settings_digest'),
            'training_epochs_target':(d.get('profile') or {}).get('epochs'),
            'seed':(d.get('ablation_design') or {}).get('seed'),
            'ablation_family':(d.get('ablation_design') or {}).get('family'),
            'ablation_factor':(d.get('ablation_design') or {}).get('factor'),
            'ablation_level':(d.get('ablation_design') or {}).get('level'),
            'ablation_reference':(d.get('ablation_design') or {}).get('reference'),
            'first_stage_only_f1_raw':((d.get('ablation_inference') or {}).get('quality') or {}).get('f1'),
            'first_stage_only_latency_p50_ms':(((d.get('ablation_inference') or {}).get('inference') or {}).get('latency_ms') or {}).get('p50'),
            'first_stage_only_parameters':(((d.get('ablation_inference') or {}).get('model_statistics') or {}).get('parameters_total')),
            'opssat_fragment_f1': (d.get('fragment_level_evaluation') or {}).get('segment_f1'),
            'opssat_fragment_roc_auc': (d.get('fragment_level_evaluation') or {}).get('roc_auc_continuous_scores'),
            'opssat_fragment_pr_auc': (d.get('fragment_level_evaluation') or {}).get('average_precision_continuous_scores'),
            'opssat_point_adjusted_f1_fixed_threshold': (d.get('point_adjusted_pointwise_evaluation') or {}).get('point_adjusted_f1'),
            'report_file': str(file.relative_to(RESULTS_DIR)),
        })
    columns = list(rows[0]) if rows else ['dataset', 'identifier', 'device']
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.metrics_', suffix='.csv')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    # Factorial analysis is handled by reports/make_plots.py after all seeds finish.
