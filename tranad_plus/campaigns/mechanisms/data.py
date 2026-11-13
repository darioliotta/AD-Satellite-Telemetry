"""Real-sample historical layouts and boundary-safe matched control datasets."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from tranad_plus.core.data import DataBundle
from opssat_pipeline import prepare_data


@dataclass
class Layout:
    labels: np.ndarray
    units: np.ndarray
    real: np.ndarray

    def selected(self):
        return Layout(self.labels[self.real], self.units[self.real], np.ones(int(self.real.sum()), dtype=bool))


def historical_layout(bundle, profile, data_root):
    """Keep existing inference; remove only padded targets for diagnostics."""
    result = {}
    if profile['dataset'] == 'OPS-SAT':
        for split, groups in bundle.groups.items():
            if split not in bundle.datasets:
                continue
            y = bundle.labels[split]
            ids = np.concatenate([np.repeat(g['id'], len(g['values'])) for g in groups])
            result[split] = Layout(y, ids, np.ones(len(y), dtype=bool))
        return result
    root = Path(data_root)/'processed/tranad_plus'/profile['dataset']
    w = profile['window_sz']
    for split, name in (('train', 'Train'), ('test', 'Test')):
        units, real = [], []
        for i, path in enumerate(sorted((root/name/'Tracks').glob('*.npy'))):
            n = len(np.load(path, mmap_mode='r', allow_pickle=False))
            units.append(np.repeat(i, n+w-1))
            real.append(np.r_[np.zeros(w-1, dtype=bool), np.ones(n, dtype=bool)])
        ids, mask = np.concatenate(units)[w-1:], np.concatenate(real)[w-1:]
        if len(ids) != len(bundle.labels[split]):
            raise RuntimeError('Historical padding/label alignment changed')
        result[split] = Layout(bundle.labels[split], ids, mask)
    return result


class IndexedWindows(torch.utils.data.Dataset):
    """One compact series tensor and target indices, no N×W×F archive/copy."""
    def __init__(self, groups, scaler_by_channel, window):
        series, targets, labels, units, exclusions = [], [], [], [], []
        offset = 0
        for g in groups:
            values = np.asarray(g['values'], dtype=np.float32)
            values = values[:, None] if values.ndim == 1 else values
            normalized = scaler_by_channel[g['channel']].transform(values).astype(np.float32)
            n = len(values)
            if n < window:
                exclusions.append({'id': g['id'], 'points': n, 'reason': 'shorter_than_full_window'})
                continue
            index = np.arange(window-1, n, dtype=np.int64)
            y = g['labels'][index] if 'labels' in g else np.full(len(index), g['label'], dtype=np.int8)
            series.append(torch.from_numpy(normalized))
            targets.append(index+offset)
            labels.append(y)
            units.append(np.repeat(g['id'], len(index)))
            offset += n
        if not targets:
            raise ValueError('No complete windows in split')
        self.series = torch.cat(series).contiguous()
        self.targets = torch.from_numpy(np.concatenate(targets))
        self.window_size = window
        self.point_labels = np.concatenate(labels).astype(np.int8)
        self.unit_ids = np.concatenate(units).astype(np.int64)
        self.exclusions = exclusions
        if not torch.isfinite(self.series).all():
            raise FloatingPointError('Nonfinite normalized control data')
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index):
        target = int(self.targets[index])
        return self.series[target-self.window_size+1:target+1]


def control_data(profile, data_root, files, fraction=.2):
    """Both conditions/baselines see exactly the same eligible target samples."""
    if profile['window_sz'] < 2 or not 0 < fraction < .5:
        raise ValueError('Controls require W>=2 and holdout fraction in (0,.5)')
    if profile['dataset'] == 'OPS-SAT':
        p = {**profile, 'segments_csv': str(Path(data_root)/'raw/OPSSAT/segments.csv')}
        tr, te, scalers, info, va = prepare_data(p, return_validation=True)
        groups = {'train': tr, 'validation': va, 'test': te}
    else:
        root = Path(data_root)/'processed/tranad_plus'/profile['dataset']
        groups, ranges = {'train': [], 'validation': [], 'test': []}, []
        for i, path in enumerate(sorted((root/'Train/Tracks').glob('*.npy'))):
            a = np.load(path, allow_pickle=False).astype(np.float32)
            labels = np.load(root/'Train/Labels'/path.name, allow_pickle=False)
            if np.any(labels != 0) or not np.isfinite(a).all():
                raise ValueError('Native optimization/validation must be finite nominal data')
            cut = int(np.floor(len(a)*(1-fraction)))
            if min(cut, len(a)-cut) < profile['window_sz']:
                raise ValueError('Training track too short for complete holdout windows: '+path.name)
            for key, lo, hi in (('train', 0, cut), ('validation', cut, len(a))):
                groups[key].append({'id': i, 'channel': 'native', 'values': a[lo:hi], 'label': 0})
            ranges.append({'track': path.name, 'optimization': [0, cut], 'validation': [cut, len(a)],
                           'range_convention': 'half_open'})
        for i, path in enumerate(sorted((root/'Test/Tracks').glob('*.npy'))):
            a = np.load(path, allow_pickle=False).astype(np.float32)
            y = np.load(root/'Test/Labels'/path.name, allow_pickle=False)
            if not np.isfinite(a).all() or not np.isin(y, [0, 1]).all():
                raise ValueError('Invalid native test values/labels')
            groups['test'].append({'id': i, 'channel': 'native', 'values': a,
                                   'labels': (y.sum(axis=1)>0).astype(np.int8)})
        scalers = {'native': MinMaxScaler().fit(np.concatenate([g['values'] for g in groups['train']]))}
        info = {'policy': 'chronological_80_20_official_train; raw_official_test_source_retained',
                'ranges': ranges, 'scaler_min': scalers['native'].data_min_.tolist(),
                'scaler_max': scalers['native'].data_max_.tolist()}
    datasets = {k: IndexedWindows(g, scalers, profile['window_sz']) for k, g in groups.items()}
    layouts = {k: Layout(ds.point_labels, ds.unit_ids, np.ones(len(ds), dtype=bool)) for k, ds in datasets.items()}
    if layouts['train'].labels.any():
        raise AssertionError('Anomalies entered optimization data')
    if profile['dataset'] == 'OPS-SAT' and any(len(np.unique(layouts[s].labels)) != 2 for s in ('validation', 'test')):
        raise ValueError('OPS-SAT complete-window validation/test lost one class')
    inventory = {**files, 'split': info, 'policy': 'complete_windows_inside_each_raw_unit; no_padding',
                 'window': profile['window_sz'], 'eligible_points': {k: len(d) for k, d in datasets.items()},
                 'excluded_short_units': {k: d.exclusions for k, d in datasets.items()},
                 'eligibility_note': 'Identical for observed/hidden conditions and deterministic baselines; excludes W-1 warmup points per unit'}
    bundle = DataBundle(datasets, {k: d.labels for k, d in layouts.items()}, inventory,
                        groups if profile['dataset'] == 'OPS-SAT' else {})
    return bundle, layouts


def batches(dataset, batch_size, device):
    """CUDA resident series + on-demand index gather, bounded by one batch."""
    from gpu_batches import training_batches
    if not isinstance(dataset, IndexedWindows):
        return training_batches(dataset, batch_size=batch_size, shuffle=False, device=device, num_workers=0)
    cuda = torch.device(device).type == 'cuda'
    series, indices = dataset.series, dataset.targets
    transfer = 'cpu'
    if cuda:
        required = series.numel()*series.element_size()+indices.numel()*indices.element_size()
        free, _ = torch.cuda.mem_get_info(device)
        if required < min(4*1024**3, int(.2*free)):
            series, indices = series.to(device), indices.to(device)
            transfer = 'cuda_resident_indexed'
        else:
            return training_batches(dataset, batch_size=batch_size, shuffle=False, device=device, num_workers=0)
    class Batches:
        def __len__(self):
            return (len(indices)+batch_size-1)//batch_size
        def __iter__(self):
            offsets = torch.arange(1-dataset.window_size, 1, device=series.device)
            for start in range(0, len(indices), batch_size):
                yield series[indices[start:start+batch_size, None]+offsets]
    return Batches(), transfer
