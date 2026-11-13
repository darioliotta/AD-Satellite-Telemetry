"""A chronological nominal holdout, without touching the official native test.

Split raw processed tracks before scaling/padding/window construction. Preserve
the authors' global concatenation/padding convention separately inside each
split; no window can cross optimization/validation boundaries. Test retains
every official sample and its existing padding/label alignment convention.
"""
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from tranad_plus.core.data import DataBundle, load_data as original_load


class NativeWindows(torch.utils.data.Dataset):
    def __init__(self, tracks, labels, scaler, window):
        self.window_size, self.padding, self.scaler = window, True, scaler
        # Upstream merge_tracks duplicates W-1 copies of each first sample, then
        # concatenates. Preserve that policy, including cross-track padding rows.
        padded = [np.concatenate([np.repeat(a[:1], window-1, axis=0), a]) for a in tracks]
        flags = [np.concatenate([np.repeat(a[:1], window-1, axis=0), a]) for a in labels]
        raw = np.concatenate(padded).astype(np.float32)
        self.big_track = torch.from_numpy(scaler.transform(raw).astype(np.float32))
        self.big_anom = torch.from_numpy(np.concatenate(flags).astype(np.float32))
        self.point_labels = (self.big_anom[window-1:].sum(dim=1) > 0).numpy().astype(np.int8)
        if not torch.isfinite(self.big_track).all() or not len(self):
            raise ValueError('Empty/nonfinite native split')
    def __len__(self):
        return len(self.big_track)-self.window_size+1
    def __getitem__(self, index):
        return self.big_track[index:index+self.window_size]


def load_data(profile, data_root, files=None, *, validation_fraction=.2):
    if profile['dataset'] == 'OPS-SAT':
        return original_load(profile, data_root, files)
    if profile['dataset'] not in ('SMAP', 'MSL') or not 0 < validation_fraction < .5:
        raise ValueError('Expected SMAP/MSL and nominal holdout fraction in (0, .5)')
    from tranad_plus.core.data import inventory
    files = files or inventory(profile, data_root)
    parent = Path(data_root)/'processed/tranad_plus'/profile['dataset']
    tracks, labels, ranges = {'train': [], 'validation': [], 'test': []}, {'train': [], 'validation': [], 'test': []}, []
    for path in sorted((parent/'Train/Tracks').glob('*.npy')):
        a = np.load(path, allow_pickle=False).astype(np.float32)
        y = np.load(parent/'Train/Labels'/path.name, allow_pickle=False)
        if not np.isfinite(a).all() or not np.isfinite(y).all() or np.any(y != 0):
            raise ValueError('Native optimization/validation source must be finite and nominal: '+str(path))
        cut = int(np.floor(len(a)*(1-validation_fraction)))
        if min(cut, len(a)-cut) < 2:
            raise ValueError('Track too short for chronological holdout: '+path.name)
        for split, start, stop in (('train', 0, cut), ('validation', cut, len(a))):
            tracks[split].append(a[start:stop])
            labels[split].append(y[start:stop])
        ranges.append({'track': path.name, 'raw_points': len(a), 'optimization_range': [0, cut],
                       'validation_range': [cut, len(a)], 'range_convention': 'half_open'})
    for path in sorted((parent/'Test/Tracks').glob('*.npy')):
        a = np.load(path, allow_pickle=False).astype(np.float32)
        y = np.load(parent/'Test/Labels'/path.name, allow_pickle=False)
        if not np.isfinite(a).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('Invalid official test track/labels')
        tracks['test'].append(a)
        labels['test'].append(y)
    scaler = MinMaxScaler().fit(np.concatenate(tracks['train']))
    datasets = {split: NativeWindows(tracks[split], labels[split], scaler, profile['window_sz']) for split in tracks}
    point_labels = {split: ds.point_labels for split, ds in datasets.items()}
    if any(point_labels[s].any() for s in ('train', 'validation')):
        raise AssertionError('Anomalies entered native training/validation')
    info = {**files, 'split': {
        'policy': 'chronological_tail_of_each_official_training_track; official_test_untouched',
        'validation_fraction': validation_fraction, 'ranges': ranges,
        'train_windows': len(datasets['train']), 'validation_windows': len(datasets['validation']),
        'test_windows': len(datasets['test']),
        'normalization': 'per_feature_MinMax_fit_on_raw_optimization_only; no_validation_or_test_fit; no_clipping',
        'scaler_data_min': scaler.data_min_.tolist(), 'scaler_data_max': scaler.data_max_.tolist(),
        'window_policy': 'upstream_global_concatenation_with_per_track_padding_separately_in_each_split',
        'point_adjustment': 'upstream_global_on_official_test',
        'overlap_optimization_validation_raw_points': 0,
        'historical_comparison': 'New 80% training control; not identical to full-training benchmark'}}
    return DataBundle(datasets, point_labels, info)
