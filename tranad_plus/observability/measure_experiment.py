#!/usr/bin/env python3
"""Non-invasive post-hoc measurement of native TranAD+ checkpoints.

Does not edit the original TranAD+ implementation, training runner or checkpoint.
Measurements of GPU/CPU *model-only forward* are not embedded-device measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCHEMA = "thesis-observability-v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.obs_', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False, default=json_default)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_model_class(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location('tranadplus_original_models_for_observability', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import original model source: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TranAD


def feedforward_width(features, value):
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    s = str(value).strip()
    if s.endswith('x'):
        return int(features * float(s[:-1]))
    return int(s)


def build_from_checkpoint(path, model_source, partial_overrides):
    """Support native final and resumable intermediate checkpoints."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise RuntimeError('Could not safely read the checkpoint. Use a checkpoint you created; '
                           'do not disable weights_only casually.') from exc
    if not isinstance(ckpt, dict):
        raise ValueError('Checkpoint is not a dict')
    if 'hp_dict' in ckpt and 'model_states' in ckpt:
        hp = ckpt['hp_dict']
        if not isinstance(hp, dict):
            raise ValueError('Invalid hp_dict')
        state = ckpt['model_states']
        fmt = 'final'
        meta = {
            'features': int(hp['features']),
            'window_sz': int(hp['window_sz']),
            'batch_sz_training': int(hp.get('batch_sz', 1)),
            'dim_feedforward': feedforward_width(int(hp['features']), hp['dim_feedforward']),
            'num_layers': int(hp.get('num_layers', 1)),
            'dataset': hp.get('dataset_str'),
            'identifier': hp.get('identifier'),
            'epochs_configured': hp.get('epochs'),
            'final_loss': hp.get('final_loss'),
            'hyperparameters': hp,
        }
    elif {'epoch','model_state_dict'}.issubset(ckpt):
        if not all(partial_overrides.get(x) is not None for x in ('features','window_sz','dim_feedforward')):
            raise ValueError('Intermediate checkpoint requires --features --window-size --feedforward '
                             '(and optionally --layers). Prefer profiling a final checkpoint.')
        state = ckpt['model_state_dict']
        fmt = 'intermediate'
        meta = {
            'features': int(partial_overrides['features']),
            'window_sz': int(partial_overrides['window_sz']),
            'batch_sz_training': None,
            'dim_feedforward': feedforward_width(int(partial_overrides['features']), partial_overrides['dim_feedforward']),
            'num_layers': int(partial_overrides.get('num_layers') or 1),
            'dataset': partial_overrides.get('dataset'),
            'identifier': partial_overrides.get('identifier'),
            'epochs_completed': int(ckpt['epoch']) + 1,
            'hyperparameters': None,
        }
    else:
        raise ValueError(f'Unknown checkpoint keys: {list(ckpt)}')
    Model = load_model_class(model_source)
    model = Model(n_feats=meta['features'], window_sz=meta['window_sz'],
                  batch_sz=1, dim_feedforward=meta['dim_feedforward'],
                  num_encoder_layers=meta['num_layers'], num_decoder_layers=meta['num_layers'])
    model.load_state_dict(state, strict=True)
    model.eval()
    del ckpt, state
    return model, meta, fmt


def theoretical_matmul_macs(features, window, feedforward, encoder_layers=1, decoder_layers=1):
    """Analytical multiply-accumulates for ONE full two-stage TranAD forward, batch=1.

    Includes attention Q/K/V/output projections, QK and AV products, the
    encoder/decoder FFN linear maps and final FCN linear projections. Excludes
    bias adds, layer nonlinearities, softmax, dropout, elementwise ops,
    memory traffic, preprocessing, and the anomaly score computation.
    """
    d, w, h = 2 * int(features), int(window), int(feedforward)
    ne, nd = int(encoder_layers), int(decoder_layers)
    if min(d, w, h, ne, nd) < 1:
        raise ValueError('Positive model dimensions/layers required')
    # Each stage: W-token encoder self-attention + encoder FFN.
    encoder = ne * (4*w*d*d + 2*w*w*d + 2*w*d*h)
    # Each stage: single-query self- and cross-attention + decoder FFN.
    decoder = nd * (4*d*d + 2*d + (2+2*w)*d*d + 2*w*d + 2*d*h)
    fcn = d * int(features)  # One output token per stage.
    macs = 2 * (encoder + decoder + fcn)
    return {
        'macs_per_window_batch1': int(macs),
        'matmul_flops_per_window_batch1': int(2 * macs),
        'flops_convention': '2 FLOPs per MAC; matmul ops only',
        'scope': 'Two full TranAD stages; QKV/out projections, attention QK/AV, feedforward and FCN linear layers',
        'excluded': 'Biases, nonlinearities, softmax, dropout, elementwise and memory/I-O operations; not a full hardware FLOP count',
        'method': 'analytical_from_model_architecture',
    }


def architecture_statistics(model):
    trainable = list(model.named_parameters())
    total = sum(int(p.numel()) for _, p in trainable)
    nonzero = sum(int(torch.count_nonzero(p).item()) for _, p in trainable)
    weight_bytes = sum(int(p.numel()) * p.element_size() for _, p in trainable)
    state_bytes = sum(int(t.numel()) * t.element_size() for t in model.state_dict().values())
    groups = {}
    for name, p in trainable:
        group = name.split('.')[0]
        record = groups.setdefault(group, {'parameters': 0, 'zeros': 0, 'bytes': 0})
        record['parameters'] += int(p.numel())
        record['zeros'] += int(p.numel()) - int(torch.count_nonzero(p).item())
        record['bytes'] += int(p.numel()) * p.element_size()
    return {
        'compute_estimate': theoretical_matmul_macs(model.n_feats, model.window_sz, model.transformer_encoder.layers[0].linear1.out_features, model.num_encoder_layers, model.num_decoder_layers) if hasattr(model, 'transformer_encoder') else None,
        'parameters_total': total,
        'parameters_nonzero': nonzero,
        'parameters_zero': total - nonzero,
        'unstructured_sparsity_fraction': (total - nonzero) / total if total else None,
        'trainable_weight_bytes': weight_bytes,
        'model_state_including_buffers_bytes': state_bytes,
        'weight_dtype': str(next(model.parameters()).dtype),
        'by_module': groups,
        'notes': 'Zero-weight fraction is not equivalent to hardware savings. Checkpoint files also include optimizer states and metadata.',
    }


def load_window(window_npy, window_size, features, seed):
    if window_npy is None:
        # Inputs are synthetic: valid only for measuring model-only compute, not accuracy.
        rng = np.random.default_rng(seed)
        return torch.tensor(rng.random((window_size, 1, features)), dtype=torch.float32), 'synthetic_uniform_0_1'
    src = np.load(window_npy, allow_pickle=False)
    if src.ndim == 2 and src.shape == (window_size, features):
        src = src[:, None, :]
    elif src.ndim == 3 and src.shape == (1, window_size, features):
        src = np.transpose(src, (1, 0, 2))
    elif src.ndim != 3 or src.shape != (window_size, 1, features):
        raise ValueError(f'Expected input (W,F), (1,W,F), or (W,1,F) with W={window_size}, F={features}; received {src.shape}')
    if not np.isfinite(src).all():
        raise ValueError('Input window contains NaN or Inf')
    return torch.tensor(np.array(src, copy=True), dtype=torch.float32), 'provided_npy'


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def benchmark(model, device, source, warmup, iterations):
    if warmup < 0 or iterations < 2:
        raise ValueError('warmup must be >=0, iterations must be >=2')
    model = model.to(device)
    source = source.to(device)
    target = source[-1:, :, :]
    # For fair comparison, benchmark only the two-stage forward (not dataset loading, POT or transfers).
    # reset_peak_memory_stats after inputs are resident. Baseline includes model+input.
    if device.type == 'cuda':
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_alloc = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
    else:
        baseline_alloc = baseline_reserved = None
    with torch.inference_mode():
        for _ in range(warmup):
            model(source, target)
        synchronize(device)
        times_ms = []
        for _ in range(iterations):
            synchronize(device)
            t0 = time.perf_counter_ns()
            model(source, target)
            synchronize(device)
            times_ms.append((time.perf_counter_ns() - t0) / 1e6)
    x = np.array(times_ms, dtype=np.float64)
    p95 = float(np.percentile(x, 95))
    stats = {
        'mode': 'torch_inference_mode_model_forward_only',
        'batch_size': 1,
        'includes_both_decoders': True,
        'warmup_iterations': warmup,
        'measured_iterations': iterations,
        'latency_ms': {
            'mean': float(x.mean()),
            'std': float(x.std(ddof=1)),
            'min': float(x.min()),
            'p50': float(np.percentile(x, 50)),
            'p95': p95,
            'p99': float(np.percentile(x, 99)),
            'max': float(x.max()),
        },
        'throughput_windows_per_second_from_mean_latency': float(1000.0 / x.mean()),
        'gpu_memory_bytes': None,
        'comment': 'Batch-1 sequential forward throughput is an inference-only estimate; excludes I/O, window generation, scoring, thresholding, data transfer, and real on-board hardware.',
    }
    if device.type == 'cuda':
        stats['gpu_memory_bytes'] = {
            'allocated_before_measurement': int(baseline_alloc),
            'reserved_before_measurement': int(baseline_reserved),
            'peak_allocated': int(torch.cuda.max_memory_allocated(device)),
            'peak_reserved': int(torch.cuda.max_memory_reserved(device)),
            'peak_delta_allocated_vs_baseline': int(max(0,torch.cuda.max_memory_allocated(device) - baseline_alloc)),
            'note': 'Peak delta includes output tensors and transient allocations, not a precise standalone activation-memory inventory.',
        }
    return stats


def hardware_status():
    # These are placeholders, not measured or estimated hardware results.
    return {
        'status': 'not_measured',
        'target_device': None,
        'toolchain': None,
        'quantization': {'status': 'not_measured', 'weight_bits': None, 'activation_bits': None, 'ebops': None},
        'synthesis': {'status': 'not_run', 'lut': None, 'ff': None, 'dsp': None, 'bram': None,
                      'clock_ns': None, 'latency_cycles': None, 'initiation_interval_cycles': None},
        'onboard_measured': {'status': 'not_measured', 'latency_ms': None, 'throughput_windows_s': None, 'power_w': None, 'energy_mj_per_window': None},
    }


def parse_training_log(path):
    if not path:
        return {'status': 'not_provided', 'epochs': []}
    pattern = re.compile(r'^Epoch\s+(\d+)\s+avg\.\s+loss:\s*([0-9eE+.\-]+)')
    epochs = []
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            match = pattern.search(line.strip())
            if match:
                epochs.append({'epoch_zero_based': int(match.group(1)), 'loss': float(match.group(2))})
    return {
        'status': 'parsed', 'epochs': epochs,
        'epoch_durations_seconds': None,
        'note': 'Original log records epoch loss, not per-epoch timestamps. Durations cannot be recovered retrospectively.',
    }


def load_hardware_report(path):
    if not path:
        return hardware_status()
    report = load_report(path)
    if not isinstance(report, dict) or 'status' not in report:
        raise ValueError('Hardware JSON must be an object containing status')
    if report['status'] != 'not_measured' and not report.get('source_artifact'):
        raise ValueError('Hardware values require source_artifact identifying the actual synthesis or measurement report')
    return {'user_supplied_hardware_report': report,
            'provenance': str(path),
            'note': 'Values are externally supplied, not independently verified by this script.'}


def load_report(path):
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def raw_metrics(npz_path, threshold):
    """Evaluate independent continuous anomaly scores, without point adjustment.

    Scores must already be correctly aligned with ground-truth labels. Labels remain unmodified.
    """
    if not npz_path:
        return {'status': 'not_measured', 'reason': 'Requires aligned raw continuous anomaly scores and labels.'}
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
    with np.load(npz_path, allow_pickle=False) as arrays:
        score = np.asarray(arrays['scores'], dtype=float).reshape(-1)
        labels = np.asarray(arrays['labels']).reshape(-1)
    if score.size != labels.size or not score.size or not np.isfinite(score).all():
        raise ValueError('Invalid score/label shape or nonfinite scores')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('labels must be binary 0/1')
    labels = labels.astype(int)
    both_classes = len(np.unique(labels)) == 2
    result = {
        'status': 'measured', 'n_points': int(labels.size), 'n_anomalies': int(labels.sum()),
        'point_adjustment': False,
        'roc_auc_score_based': float(roc_auc_score(labels, score)) if both_classes else None,
        'average_precision': float(average_precision_score(labels, score)) if labels.sum() > 0 else None,
        'threshold': threshold,
        'threshold_origin': 'supplied_by_user_do_not_select_on_test',
        'pointwise_threshold_metrics': None,
        'notes': 'ROC AUC uses continuous scores; undefined ROC AUC for single-class ground truth is stored as null.',
    }
    if threshold is not None:
        pred = (score > threshold).astype(int)
        p, r, f, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0,1]).ravel()
        result['pointwise_threshold_metrics'] = {
            'precision': float(p), 'recall': float(r), 'f1': float(f),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-source', type=Path, default=Path(__file__).resolve().parent / 'TranADPlus/src/models.py')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu', help='cpu, cuda:0 or mps')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--window-npy', type=Path)
    parser.add_argument('--evaluation-json', type=Path)
    parser.add_argument('--training-json', type=Path)
    parser.add_argument('--training-log', type=Path, help='Optional original epoch-loss log; no retrospective epoch timing')
    parser.add_argument('--hardware-json', type=Path, help='Optional real, provenance-labeled synthesis/hardware results')
    parser.add_argument('--scores-npz', type=Path)
    parser.add_argument('--threshold', type=float, help='A threshold fixed before test, not oracle-selected from test labels')
    # For intermediate checkpoints only:
    parser.add_argument('--features', type=int)
    parser.add_argument('--window-size', type=int)
    parser.add_argument('--feedforward')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dataset')
    parser.add_argument('--identifier')
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f'Missing checkpoint: {args.checkpoint}')
    if not args.model_source.is_file():
        parser.error(f'Missing original models.py: {args.model_source}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available; use --device cpu')
    if device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        parser.error('MPS is not available; use --device cpu')
    if args.threshold is not None and args.scores_npz is None:
        parser.error('--threshold requires --scores-npz')
    model, metadata, ckpt_format = build_from_checkpoint(
        args.checkpoint, args.model_source,
        {'features': args.features, 'window_sz': args.window_size,
         'dim_feedforward': args.feedforward, 'num_layers': args.layers,
         'dataset': args.dataset, 'identifier': args.identifier})
    model_metrics = architecture_statistics(model)
    window, input_provenance = load_window(args.window_npy, metadata['window_sz'], metadata['features'], args.seed)
    timings = benchmark(model, device, window, args.warmup, args.iterations)
    ident = metadata.get('identifier') or args.checkpoint.stem
    dataset = metadata.get('dataset') or args.dataset
    details = {
        'schema_version': SCHEMA,
        'created_at': utc_now(),
        'model': 'TranAD+ original',
        'dataset': dataset,
        'identifier': ident,
        'architecture': metadata,
        'checkpoint': {
            'format': ckpt_format, 'source_file': str(args.checkpoint),
            'sha256': hash_file(args.checkpoint), 'size_bytes': args.checkpoint.stat().st_size,
        },
        'source': {'models_py': str(args.model_source), 'sha256': hash_file(args.model_source)},
        'environment': {
            'platform': platform.platform(), 'python': sys.version.split()[0], 'torch': torch.__version__,
            'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or device.type,
            'cuda_runtime': torch.version.cuda, 'torch_threads': torch.get_num_threads(),
            'input_provenance': input_provenance, 'input_shape_W_B_F': list(window.shape), 'dtype': 'float32', 'seed_for_synthetic_input': args.seed if not args.window_npy else None,
        },
        'model_statistics': model_metrics,
        'inference': timings,
        'original_paper_evaluation': load_report(args.evaluation_json),
        'training_record': load_report(args.training_json),
        'training_loss_history': parse_training_log(args.training_log),
        'raw_score_evaluation': raw_metrics(args.scores_npz, args.threshold),
        'hardware_acceleration': load_hardware_report(args.hardware_json),
        'measurement_limitations': [
            'No onboard device latency, power, FPGA logic utilization, or HGQ2 EBOPs measured.',
            'Model-only inference benchmark, batch=1; excludes preprocessing and anomaly decision.',
            'POT/BF metrics imported from native reports may include point-adjustment and test-label-based model selection.',
            'Raw metrics require externally generated continuous scores correctly aligned with labels.',
        ],
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = args.output_dir / f'{ident}_observability_{stamp}.json'
    write_json_atomic(destination, details)
    print('Created:', destination)
    print('Checkpoint:', ckpt_format, '| parameters:', model_metrics['parameters_total'])
    print('Latency batch=1:', round(timings['latency_ms']['p50'], 4), 'ms p50,',
          round(timings['latency_ms']['p95'], 4), 'ms p95, device', str(device))
    print('No GPU/FPGA hardware numbers are inferred from these results.')
    return destination


if __name__ == '__main__':
    main()
