#!/usr/bin/env python3
"""Export continuous, point-aligned TranAD+ anomaly scores without changing native evaluation.

This reproduces the native dataset/window protocol (including its known cross-track
window boundary behavior) and fits its MinMaxScaler on Train data only.
Run separately from the current training campaign; it reads but never edits checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .measure_experiment import build_from_checkpoint, hash_file, write_json_atomic, utc_now
except ImportError:  # direct execution: python observability/export_anomaly_scores.py
    from measure_experiment import build_from_checkpoint, hash_file, write_json_atomic, utc_now


def pair_paths(base, split):
    tracks_dir = base / split / 'Tracks'
    labels_dir = base / split / 'Labels'
    if not tracks_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f'Missing Tracks/Labels directory for {split}: {base}')
    tracks = {p.stem: p for p in tracks_dir.glob('*.npy')}
    labels = {p.stem: p for p in labels_dir.glob('*.npy')}
    if not tracks or not labels or tracks.keys() != labels.keys():
        raise ValueError(f'Unmatched tracks/labels for {split}: only tracks={sorted(tracks.keys()-labels.keys())}, only labels={sorted(labels.keys()-tracks.keys())}')
    stems = sorted(tracks)
    return [str(tracks[s]) for s in stems], [str(labels[s]) for s in stems], stems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--models-py', required=True, type=Path)
    parser.add_argument('--project-dir', required=True, type=Path, help='Directory containing TranADPlus/src/')
    parser.add_argument('--preprocessed-dir', required=True, type=Path, help='.../TranADPlus/Datasets/Preprocessed/DATASET')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--alpha', default=0.5, type=float, help='Pre-specified mixing alpha; beta=1-alpha. Not selected on test labels.')
    args = parser.parse_args(argv)
    if not 0 <= args.alpha <= 1 or args.batch_size < 1:
        parser.error('alpha must be in [0,1], batch-size >= 1')
    import sys
    sys.path.insert(0, str(args.project_dir))
    from TranADPlus.src.datasets import TSDataset_tracks
    model, meta, fmt = build_from_checkpoint(args.checkpoint, args.models_py, {})
    if fmt != 'final':
        parser.error('Only final checkpoints supported by score exporter')
    hp = meta['hyperparameters']
    if hp.get('downsample') is not None:
        parser.error('This script has not validated downsample handling. Refusing to export incompatible scores.')
    train_tracks, train_labels, train_stems = pair_paths(args.preprocessed_dir, 'Train')
    test_tracks, test_labels, test_stems = pair_paths(args.preprocessed_dir, 'Test')
    ws = meta['window_sz']
    # Identical scaler and padding choices to original native code.
    train_dataset = TSDataset_tracks(track_paths=train_tracks, anom_paths=train_labels, window_size=ws,
                                     scaling=True, padding=bool(hp.get('padding', True)), downsample=None)
    test_dataset = TSDataset_tracks(track_paths=test_tracks, anom_paths=test_labels, window_size=ws,
                                    scaling=True, scaler=train_dataset.scaler,
                                    padding=bool(hp.get('padding', True)), downsample=None)
    if int(test_dataset.big_track.shape[-1]) != meta['features']:
        raise ValueError('Test feature width differs from checkpoint')
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    model = model.to(device).eval()
    chunks_score = []
    chunks_label = []
    with torch.inference_mode():
        for windows, anomalies in loader:
            windows = windows.to(device).permute(1,0,2)
            tgt = windows[-1:, :, :]
            out1, out2 = model(windows, tgt)
            mixed = args.alpha * out1 + (1.0-args.alpha) * out2
            scores = torch.mean((mixed - tgt).square(), dim=2)[0]
            chunks_score.append(scores.detach().cpu().numpy())
            label_slice = anomalies[:, -1, :] if test_dataset.padding else anomalies[:, 0, :]
            chunks_label.append((label_slice.sum(dim=1) >= 1).numpy().astype(np.int8))
    scores = np.concatenate(chunks_score).astype(np.float32)
    labels = np.concatenate(chunks_label).astype(np.int8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / f"{meta.get('identifier') or args.checkpoint.stem}_raw_scores_alpha{args.alpha:.3f}.npz"
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite existing scores: {dest}')
    fd, temp = tempfile.mkstemp(prefix='.scores_', suffix='.npz', dir=args.output_dir)
    os.close(fd)
    try:
        np.savez_compressed(temp, scores=scores, labels=labels)
        os.replace(temp, dest)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    meta_dest = dest.with_suffix('.json')
    write_json_atomic(meta_dest, {
        'created_at': utc_now(), 'dataset': meta.get('dataset'), 'identifier': meta.get('identifier'),
        'checkpoint_sha256': hash_file(args.checkpoint), 'alpha': args.alpha, 'beta': 1.0-args.alpha,
        'n_points': len(scores), 'n_anomalies': int(labels.sum()), 'track_stems_train': train_stems,
        'track_stems_test': test_stems, 'scaling': 'minmax_fitted_train', 'padding': test_dataset.padding,
        'labels': 'any_feature_at_window_target', 'score': 'mean((alpha*x1+(1-alpha)*x2-target)^2 over features)',
        'warnings': [
            'Original TSDataset_tracks concatenates padded tracks before windowing; boundary-crossing windows remain possible.',
            'These scores are raw, without point adjustment. Choose any threshold on separate validation/training.',
            'These scores are NOT automatically comparable to test-optimized alphas in the native author evaluation.',
        ],
    })
    print('Raw scores:', dest, 'n_points:', len(scores), 'anomalies:', int(labels.sum()))
    print('Metadata:', meta_dest)

if __name__ == '__main__':
    main()
