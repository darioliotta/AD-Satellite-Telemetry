"""Versioned single-file experiment selection with validated overrides.

The JSON file is the source of truth for the campaign. Original upstream
TranADPlus files remain unchanged. Importable by all worker processes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PLAN_ENV = 'TRANADPLUS_CAMPAIGN_PLAN'
OVERRIDES = {'epochs', 'batch_sz', 'window_sz', 'lr', 'dim_feedforward',
             'pot_q', 'num_layers', 'seed', 'shuffle', 'padding', 'num_workers', 'downsample', 'segments_csv', 'channels', 'max_train_fragments_per_channel', 'max_test_fragments_per_channel', 'max_validation_anomalies_per_channel', 'validation_fraction', 'split_seed', 'ablation_family', 'ablation_factor', 'ablation_level', 'ablation_reference'}


def read_plan(path: str | Path):
    file = Path(path).expanduser().resolve()
    data = json.loads(file.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('experiments'), list) or not data['experiments']:
        raise ValueError('Campaign JSON must have a nonempty experiments list')
    settings = data.get('measurement', {})
    if not isinstance(settings, dict):
        raise ValueError('measurement must be an object')
    settings = {
        'enabled': settings.get('enabled', True),
        'warmup': settings.get('warmup', 10),
        'iterations': settings.get('iterations', 50),
        'score_batch_size': settings.get('score_batch_size', 512),
        'alpha': settings.get('alpha', 0.5),
        'threshold_train_quantile': settings.get('threshold_train_quantile', 0.995),
        'save_raw_scores': settings.get('save_raw_scores', True),
        'hardware_json_dir': settings.get('hardware_json_dir'),
        'paper_alpha_search': settings.get('paper_alpha_search',False),
        'inference_ablation': settings.get('inference_ablation',False),
        'export_decoder_arrays': settings.get('export_decoder_arrays',False),
    }
    if (not isinstance(settings['enabled'], bool) or not isinstance(settings['save_raw_scores'], bool)
            or not isinstance(settings['paper_alpha_search'], bool)
            or not isinstance(settings['inference_ablation'], bool)
            or not isinstance(settings['export_decoder_arrays'], bool)):
        raise ValueError('enabled/save_raw_scores/paper_alpha_search/inference_ablation must be boolean')
    if any(not isinstance(settings[k], int) or isinstance(settings[k], bool) or settings[k] < limit
           for k, limit in [('warmup', 0), ('iterations', 2), ('score_batch_size', 1)]):
        raise ValueError('invalid measurement warmup / iterations / score_batch_size')
    if not 0 <= settings['alpha'] <= 1 or not 0 < settings['threshold_train_quantile'] < 1:
        raise ValueError('alpha in [0,1] and quantile in (0,1) required')
    if settings['hardware_json_dir'] is not None:
        settings['hardware_json_dir'] = str((file.parent / settings['hardware_json_dir']).resolve())
    return file, data['experiments'], settings


def register_plan_profiles(profiles: dict, path: str | Path):
    _, entries, _ = read_plan(path)
    original = {name: (dataset, spec) for dataset, specs in profiles.items() for name, spec in specs.items()}
    seen = set()
    names = []
    for entry in entries:
        if isinstance(entry, str):
            name = entry
            if name not in original:
                raise ValueError(f'Unknown catalog profile {name!r}')
        elif isinstance(entry, dict):
            keys = set(entry)
            invalid = keys - OVERRIDES - {'base_profile', 'identifier'}
            if invalid:
                raise ValueError(f'Unsupported profile override(s): {sorted(invalid)}')
            base, name = entry.get('base_profile'), entry.get('identifier')
            if base not in original:
                raise ValueError(f'Unknown base_profile {base!r}')
            if not isinstance(name, str) or not name or not all(c.isalnum() or c in '_-' for c in name):
                raise ValueError('Custom identifier must contain only letters, digits, _ or -')
            dataset, spec = original[base]
            new = {**spec, **{k: v for k, v in entry.items() if k in OVERRIDES}}
            if new.get('ablation_family') is not None:
                if new.get('ablation_factor') not in ('baseline','window','feedforward','combined'):
                    raise ValueError(f'{name}: supported training ablation factors: baseline/window/feedforward/combined')
                if new.get('ablation_reference') is not None and not isinstance(new['ablation_reference'],str):
                    raise ValueError(f'{name}: ablation_reference must be a string')
            if dataset == 'OPS-SAT':
                if 'segments_csv' in entry:
                    raw_path = entry['segments_csv']
                    if isinstance(raw_path, str) and raw_path.startswith('@data/'):
                        # Portable v7 plan: data root comes from project_paths / environment.
                        # Legacy relative and absolute paths retain their old interpretation.
                        from project_paths import DATA_ROOT
                        new['segments_csv'] = str((DATA_ROOT / raw_path[len('@data/'):]).resolve())
                    else:
                        new['segments_csv'] = str((Path(path).expanduser().resolve().parent / raw_path).resolve())
                if new.get('channels') is not None and (not isinstance(new['channels'],list) or not all(isinstance(x,str) for x in new['channels'])):
                    raise ValueError('OPS-SAT channels must be a list of strings')
                if 'split_seed' in new and (type(new['split_seed']) is not int or new['split_seed']<0):
                    raise ValueError(f'{name}: split_seed must be non-negative int')
                if type(new.get('validation_fraction',0.0)) not in (float,int) or not 0 <= new.get('validation_fraction',0) < 0.5:
                    raise ValueError(f'{name}: validation_fraction must be in [0,0.5)')
                for key in ('max_train_fragments_per_channel','max_test_fragments_per_channel','max_validation_anomalies_per_channel'):
                    if new.get(key) is not None and (type(new[key]) is not int or new[key] < 1):
                        raise ValueError(f'{key} must be a positive integer or null')
            if name in original and (dataset != original[name][0] or new != original[name][1]):
                raise ValueError(f'{name!r} is an original profile; use a NEW identifier for overrides')
            for key in ('epochs', 'batch_sz', 'window_sz'):
                if type(new[key]) is not int or new[key] < 1:
                    raise ValueError(f'{name}: {key} must be a positive integer')
            if new.get('seed') is not None and (type(new['seed']) is not int or new['seed'] < 0):
                raise ValueError(f'{name}: seed must be a nonnegative integer')
            if not isinstance(new['lr'], (int, float)) or new['lr'] <= 0:
                raise ValueError(f'{name}: lr must be >0')
            if not 0 < float(new['pot_q']) < 1:
                raise ValueError(f'{name}: pot_q must be in (0,1)')
            if name in profiles[dataset] and profiles[dataset][name] != new:
                raise ValueError(f'Conflicting identifier {name}')
            profiles[dataset][name] = new
        else:
            raise ValueError(f'Invalid experiment entry: {entry!r}')
        if name in seen:
            raise ValueError(f'Duplicate selected profile {name!r}')
        seen.add(name)
        names.append(name)
    return names


def config_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
