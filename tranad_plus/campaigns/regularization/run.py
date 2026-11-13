"""V19: staged, recoverable generalization and compression controls."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'toast-matplotlib'))
from tranad_plus.core.common import digest, hash_file, now, write_json, run_lock, source_manifest


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
    p.add_argument('--output-root', type=Path, default=PROJECT/'results/v19/controlled_study')
    p.add_argument('--data-root', type=Path, default=PROJECT/'data')
    p.add_argument('--parent-root', type=Path, default=PROJECT/'results/v13/architectures')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--seed', type=int, action='append')
    p.add_argument('--dataset', action='append', choices=('OPS-SAT', 'SMAP', 'MSL'))
    p.add_argument('--stage', choices=('diagnostic', 'compression', 'all'), default='diagnostic')
    p.add_argument('--with-integer', action='store_true', help='Also evaluate W8A8/W4A4 after each trajectory')
    p.add_argument('--quantize-only', action='store_true', help='No training; integer evaluation of completed selected jobs')
    p.add_argument('--config', action='append', choices=('reference', 'compact', 'window_only', 'ff_only'))
    p.add_argument('--arm', action='append', choices=('baseline', 'weight_decay'))
    p.add_argument('--all-configs', action='store_true', help='Also isolate W and FF; reuses completed jobs')
    p.add_argument('--only', help='Exact job id from --dry-run')
    p.add_argument('--stop-on-patience', action='store_true',
                   help='Actually halt at patience; default audits full horizon to expose overfitting')
    p.add_argument('--stop-after-epoch', type=int, help='Intentional recovery test; exits 75')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check', action='store_true', help='Read-only data/split/source/storage preflight')
    p.add_argument('--status', action='store_true')
    p.add_argument('--audit-existing', action='store_true',
                   help='Evaluate existing V13 final train/nominal-validation MSE; no training')
    p.add_argument('--no-progress', action='store_true')
    p.add_argument('--fail-fast', action='store_true')
    args = p.parse_args(argv)
    if args.cpu_threads < 1 or (args.stop_after_epoch is not None and args.stop_after_epoch < 1):
        p.error('Positive thread count and stop epoch required')
    if args.all_configs and args.config:
        p.error('--all-configs and --config are mutually exclusive')
    if args.with_integer and args.quantize_only:
        p.error('Choose --with-integer or --quantize-only')
    return args


def build_plan(spec, args):
    if spec.get('schema_version') != 2 or spec.get('name') != 'v19_controlled_generalization':
        raise ValueError('Expected V19 controlled-study schema 2')
    if set(spec['variants']) != {'reference', 'shared_decoder', 'single_pass', 'encoder_only'} or len(spec['variants']) != 4:
        raise ValueError('All four architectures required exactly once')
    if spec['evaluation']['alpha_steps'] != 21 or spec['evaluation']['paper_threshold_steps'] != 200:
        raise ValueError('BF200 and 21-alpha grids are fixed')
    if spec['max_epochs'] < spec['fixed_epoch_checkpoint']:
        raise ValueError('Audit horizon shorter than fixed checkpoint')
    if spec['storage']['save_pointwise_arrays'] or spec['storage']['save_epoch_checkpoints']:
        raise ValueError('Compact storage required')
    configs = args.config or spec['configurations']
    if args.all_configs:
        configs = configs+spec['optional_configurations']
    seeds = args.seed or spec['seeds']
    arms = args.arm or list(spec['arms'])
    for selected, available in ((seeds, spec['seeds']), (arms, spec['arms']),
                                (configs, spec['configurations']+spec['optional_configurations'])):
        if len(selected) != len(set(selected)) or not set(selected).issubset(available):
            raise ValueError('Select distinct predeclared settings')
    baseline = json.loads((PROJECT/'tranad_plus/core/configs/baselines.json').read_text())
    jobs = []
    for ds in args.dataset or spec['datasets']:
        info = baseline['datasets'][ds]
        candidates = []
        # OPS-SAT includes the architecture controls from phase one. Native
        # phase one targets an archived benchmark configuration, phase two
        # uses the existing study grid (different, explicitly labelled family).
        if ds == 'OPS-SAT' or args.stage in ('compression', 'all'):
            for config in configs:
                for variant in spec['variants']:
                    candidates.append({**baseline['training'], **info['training'], **info['configurations'][config],
                                       'features': info['features'], 'dataset': ds, 'variant': variant,
                                       'configuration': config, 'epochs': spec['max_epochs'], 'family': 'study_grid',
                                       'fixed_epoch_checkpoint': spec['fixed_epoch_checkpoint']})
        if ds != 'OPS-SAT' and args.stage in ('diagnostic', 'all'):
            anchor = {k: v for k, v in spec['author_anchors'][ds].items() if k != 'evidence'}
            candidates.append({**baseline['training'], **anchor, 'dataset': ds, 'variant': 'reference',
                               'family': 'author_anchor'})
        for candidate in candidates:
            for seed in seeds:
                for arm in arms:
                    identifier = f"{ds}/{candidate['variant']}/{candidate['configuration']}/seed_{seed}/{arm}"
                    profile = {**candidate, 'seed': seed, 'identifier': identifier.replace('/', '_'),
                               'validation_fraction': (info['training']['validation_fraction'] if ds == 'OPS-SAT'
                                                       else spec['native_validation_fraction'])}
                    if args.only is None or args.only == identifier:
                        jobs.append({'id': identifier, 'profile': profile, 'arm': spec['arms'][arm], 'arm_name': arm})
    if not jobs:
        raise ValueError('No selected jobs')
    return jobs


def output_path(args):
    output = args.output_root.expanduser().resolve()
    protected = [args.data_root.resolve(), args.parent_root.resolve(), PROJECT/'tranad_plus', PROJECT/'tesi', PROJECT/'docs']
    protected += [PROJECT/'results'/f'v{v}' for v in range(10, 19)]
    protected.append(PROJECT/'results/legacy')
    protected.append(PROJECT/'results/v19/regularization')  # Preserve schema-1 audit/results.
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Output overlaps protected sources, data, or prior results')
    return output


def complete(folder, signature):
    path = folder/'status.json'
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    if value['signature'] != signature:
        raise RuntimeError('Changed configuration/source/data/environment: '+str(folder))
    if value['status'] != 'complete':
        return False
    for name, expected in value['artifacts'].items():
        if not (folder/name).is_file() or hash_file(folder/name) != expected:
            raise RuntimeError('Changed/missing completed artifact: '+str(folder/name))
    return True


def summarize(output):
    """Compact rows, paired checkpoint deltas, and descriptive seed statistics."""
    import statistics
    from collections import defaultdict
    from tranad_plus.core.common import atomic_destination
    rows = []
    for path in sorted(output.glob('*/*/*/seed_*/*/training.json')):
        folder = path.parent
        status = folder/'status.json'
        if not status.exists() or json.loads(status.read_text())['status'] != 'complete':
            continue
        profile = json.loads((folder/'config.json').read_text())
        trained = json.loads(path.read_text())
        history = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
        by_epoch = {r['epoch']: r for r in history}
        for checkpoint, epoch in trained['snapshots'].items():
            result_path = folder/'evaluation'/checkpoint/'result.json'
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            for protocol in ('test_validation_calibrated', 'test_paper_oracle'):
                for method in ('bf', 'pot'):
                    endpoint = result[protocol][method]
                    row = {'dataset': profile['dataset'], 'family': profile.get('family'),
                           'variant': profile['variant'], 'configuration': profile['configuration'],
                           'window': profile['window_sz'], 'ff': profile['dim_feedforward'],
                           'seed': profile['seed'], 'arm': folder.name, 'checkpoint': checkpoint,
                           'epoch': epoch, 'protocol': protocol, 'method': method,
                           'early_stop_epoch': trained['policy']['stop_epoch'],
                           'train_mse': by_epoch[epoch]['train_nominal']['last_output_mse'],
                           'validation_mse': by_epoch[epoch]['validation_nominal']['last_output_mse']}
                    for field in ('f1', 'precision', 'recall', 'ROC/AUC', 'TP', 'TN', 'FP', 'FN'):
                        row[field] = endpoint.get(method+'_'+field)
                    row['raw_f1'] = endpoint.get('raw_without_PA', {}).get('f1')
                    row['auc_score'] = endpoint.get('roc_auc_continuous_scores')
                    rows.append(row)
    def save(name, values):
        if values:
            with atomic_destination(output/'tables'/name) as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    save('per_seed.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ('dataset', 'family', 'variant', 'configuration', 'arm', 'checkpoint', 'protocol', 'method'))].append(row)
    aggregated = []
    for key, group in sorted(groups.items()):
        row = dict(zip(('dataset', 'family', 'variant', 'configuration', 'arm', 'checkpoint', 'protocol', 'method'), key))
        row['seeds'] = len(group)
        for field in ('f1', 'precision', 'recall', 'ROC/AUC', 'raw_f1', 'auc_score', 'train_mse', 'validation_mse'):
            values = [r[field] for r in group if r[field] is not None]
            row[field+'_available'] = len(values)
            row[field+'_mean'] = statistics.mean(values) if values else None
            row[field+'_sample_std'] = statistics.stdev(values) if len(values) > 1 else None
        aggregated.append(row)
    save('mean_std.csv', aggregated)
    paired = []
    index = {(r['dataset'], r['variant'], r['configuration'], r['arm'], r['seed'], r['protocol'], r['method'], r['checkpoint']): r for r in rows}
    for key, early in sorted(index.items()):
        if key[-1] != 'early_stopped':
            continue
        for comparator in sorted({r['checkpoint'] for r in rows if r['checkpoint'].startswith('fixed_')})+['final']:
            final = index.get(key[:-1]+(comparator,))
            if final is None:
                continue
            row = {k: early[k] for k in ('dataset', 'variant', 'configuration', 'arm', 'seed', 'protocol', 'method')}
            row.update(comparator=comparator, selected_epoch=early['epoch'], comparator_epoch=final['epoch'])
            for field in ('f1', 'raw_f1', 'train_mse', 'validation_mse'):
                row[field+'_early_minus_comparator'] = early[field]-final[field] if early[field] is not None and final[field] is not None else None
            paired.append(row)
    save('paired_checkpoint_differences.csv', paired)
    arms = []
    for key, regularized in sorted(index.items()):
        if regularized['arm'] != 'weight_decay':
            continue
        baseline_key = list(key)
        baseline_key[3] = 'baseline'
        baseline_row = index.get(tuple(baseline_key))
        if baseline_row is None:
            continue
        row = {k: regularized[k] for k in ('dataset', 'variant', 'configuration', 'seed', 'checkpoint', 'protocol', 'method')}
        for field in ('f1', 'raw_f1', 'train_mse', 'validation_mse'):
            row[field+'_weight_decay_minus_baseline'] = (regularized[field]-baseline_row[field]
                                                       if regularized[field] is not None and baseline_row[field] is not None else None)
        arms.append(row)
    save('paired_regularization_differences.csv', arms)
    integer_rows = []
    for path in sorted(output.glob('*/*/*/seed_*/*/integer/*/int*/result.json')):
        if not path.with_name('status.json').exists():
            continue
        folder = path.parents[3]
        profile = json.loads((folder/'config.json').read_text())
        result = json.loads(path.read_text())
        for protocol in ('test_validation_calibrated', 'test_paper_oracle'):
            for method in ('bf', 'pot'):
                endpoint = result[protocol][method]
                integer_rows.append({'dataset': profile['dataset'], 'variant': profile['variant'],
                                     'configuration': profile['configuration'], 'seed': profile['seed'],
                                     'arm': folder.name, 'checkpoint': path.parents[1].name,
                                     'precision': path.parent.name, 'protocol': protocol, 'method': method,
                                     'f1': endpoint.get(method+'_f1'), 'precision_metric': endpoint.get(method+'_precision'),
                                     'recall': endpoint.get(method+'_recall'), 'AUC_PA': endpoint.get(method+'_ROC/AUC'),
                                     'raw_f1': endpoint.get('raw_without_PA', {}).get('f1'),
                                     'AUC_score': endpoint.get('roc_auc_continuous_scores')})
    save('integer_per_seed.csv', integer_rows)
    # One tidy table for size/operation/performance plots. Integer model buffers
    # are measured on the converted object; FP32 includes positional buffers.
    tradeoffs = []
    for row in rows:
        parent = output/row['dataset']/row['variant']/row['configuration']/f"seed_{row['seed']}"/row['arm']
        resources = json.loads((parent/'resources.json').read_text())
        result = json.loads((parent/'evaluation'/row['checkpoint']/'result.json').read_text())
        tradeoffs.append({**{k: row[k] for k in ('dataset', 'family', 'variant', 'configuration', 'window', 'ff', 'seed', 'arm', 'checkpoint', 'epoch', 'protocol', 'method')},
                          'precision': 'fp32', 'f1': row['f1'], 'precision_metric': row['precision'], 'recall': row['recall'],
                          'AUC_PA': row['ROC/AUC'], 'raw_f1': row['raw_f1'], 'AUC_score': row['auc_score'],
                          'parameters_original_architecture': resources['parameters'],
                          'model_tensor_bytes': resources['state_bytes'],
                          'analytic_MACs_per_window': resources['analytic_operations']['matmul_MACs_per_window'],
                          'test_inference_seconds_including_transfer': result['inference_by_split']['test']['seconds_including_transfer']})
    for row in integer_rows:
        parent = output/row['dataset']/row['variant']/row['configuration']/f"seed_{row['seed']}"/row['arm']
        profile = json.loads((parent/'config.json').read_text())
        trained = json.loads((parent/'training.json').read_text())
        resources = json.loads((parent/'resources.json').read_text())
        target = parent/'integer'/row['checkpoint']/row['precision']
        storage = json.loads((target/'storage.json').read_text())
        result = json.loads((target/'result.json').read_text())
        tradeoffs.append({**{k: row[k] for k in ('dataset', 'variant', 'configuration', 'seed', 'arm', 'checkpoint', 'protocol', 'method', 'precision', 'f1', 'precision_metric', 'recall', 'AUC_PA', 'raw_f1', 'AUC_score')},
                          'family': profile.get('family'), 'window': profile['window_sz'], 'ff': profile['dim_feedforward'],
                          'epoch': trained['snapshots'][row['checkpoint']],
                          'parameters_original_architecture': resources['parameters'],
                          'model_tensor_bytes': storage['resident_buffers_bytes'],
                          'analytic_MACs_per_window': resources['analytic_operations']['matmul_MACs_per_window'],
                          'test_inference_seconds_including_transfer': result['inference_by_split']['test']['seconds_including_transfer']})
    save('compression_tradeoffs.csv', tradeoffs)


def audit_existing(jobs, args, output, device):
    """Final train/validation gap only: cannot reconstruct historical stopping."""
    from tranad_plus.core.data import load_data, inventory
    from .training import configured_model, nominal_loader, evaluate_mse
    import torch
    rows, seen = [], set()
    for job in jobs:
        profile = job['profile']
        if profile['dataset'] != 'OPS-SAT':
            continue  # Existing native models did not hold out this validation.
        key = (profile['variant'], profile['configuration'], profile['seed'])
        if key in seen:
            continue
        seen.add(key)
        parent = args.parent_root/'OPS-SAT'/key[0]/key[1]/f'seed_{key[2]}'
        path = parent/'checkpoints/final.ckpt'
        if not path.exists():
            rows.append({'parent': str(parent), 'status': 'missing_checkpoint'})
            continue
        old = json.loads((parent/'config.json').read_text())
        # A newly invented split must never masquerade as unseen validation for
        # a model that trained on those same fragments.
        if old.get('validation_fraction') != profile['validation_fraction'] or old.get('split_seed') != profile['split_seed']:
            raise RuntimeError('Existing model has a different held-out split')
        bundle = load_data(old, args.data_root)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        old_manifest = json.loads((parent/'manifest.json').read_text())
        if saved.get('profile') != old or saved.get('signature') != old_manifest.get('signature'):
            raise RuntimeError('Existing checkpoint does not match its archived profile/signature')
        current_files = inventory(old, args.data_root)['files']
        recorded = old_manifest.get('data', old_manifest.get('data_inventory', {}))
        # Native V13 manifests store input inventory under data. Verify below
        # against the committed data_inventory.json as well, when present.
        inventory_path = parent/'data_inventory.json'
        if inventory_path.exists():
            recorded = json.loads(inventory_path.read_text())
        if recorded.get('files') != current_files:
            raise RuntimeError('Existing checkpoint input files cannot be verified')
        previous_split = recorded.get('split', {})
        for name in ('train_fragment_ids', 'validation_fragment_ids', 'test_fragment_ids'):
            if previous_split.get(name) != bundle.inventory['split'][name]:
                raise RuntimeError('Existing model fragment split mismatch')
        model = configured_model(old, {'dropout': .1}).to(device)
        model.load_state_dict(saved['model_state'], strict=True)
        metrics = {s: evaluate_mse(model, *nominal_loader(bundle, s, 1024, device), device)
                   for s in ('train', 'validation')}
        rows.append({'parent': str(parent), 'status': 'measured', 'checkpoint_sha256': hash_file(path),
                     'epoch': saved['epoch']+1, **metrics})
    write_json(output/'existing_final_audit.json', {'records': rows,
               'limitation': 'Final nominal train/validation errors only; not a historical overfitting diagnosis'})
    from collections import defaultdict
    import statistics
    from tranad_plus.core.common import atomic_destination
    grouped = defaultdict(list)
    for row in rows:
        if row['status'] == 'measured':
            parts = Path(row['parent']).parts
            grouped[(parts[-3], parts[-2])].append(row)
    summary = []
    for (variant, config), group in sorted(grouped.items()):
        training = statistics.mean(r['train']['last_output_mse'] for r in group)
        validation = statistics.mean(r['validation']['last_output_mse'] for r in group)
        summary.append({'variant': variant, 'configuration': config, 'seeds': len(group),
                        'train_nominal_mse_mean': training, 'validation_nominal_mse_mean': validation,
                        'relative_gap_pct': 100*(validation/training-1) if training else None})
    if summary:
        with atomic_destination(output/'existing_final_audit_summary.csv') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)


def main(argv=None):
    args = arguments(argv)
    spec = json.loads(args.plan.read_text())
    jobs = build_plan(spec, args)
    output = output_path(args)
    print(f"V19 CONTROLLED STUDY | {args.stage} | {len(jobs)} trajectories | "
          f"datasets={','.join(dict.fromkeys(j['profile']['dataset'] for j in jobs))}", flush=True)
    print('Adam wd=0 vs 1e-4 | dropout=0.1 | FP32 | nominal-validation stopping on every dataset.', flush=True)
    print('OPS-SAT also has a secondary checkpoint selected on labelled validation; test never selects epochs.', flush=True)
    print('Output: '+str(output), flush=True)
    if args.status:
        counts = {}
        for job in jobs:
            path = output/job['id']/'status.json'
            state = json.loads(path.read_text())['status'] if path.exists() else 'pending'
            counts[state] = counts.get(state, 0)+1
        print(json.dumps(counts, indent=2))
        if args.quantize_only or args.with_integer:
            integers = {'complete': 0, 'pending': 0, 'missing_parent_trajectories': 0}
            for job in jobs:
                folder = output/job['id']
                trained = folder/'training.json'
                if not trained.exists():
                    integers['missing_parent_trajectories'] += 1
                    continue
                for checkpoint in json.loads(trained.read_text())['snapshots']:
                    if checkpoint not in spec['integer']['checkpoints']:
                        continue
                    for bits in spec['integer']['bits']:
                        path = folder/'integer'/checkpoint/f'int{bits}'/'status.json'
                        state = 'complete' if path.exists() and json.loads(path.read_text())['status'] == 'complete' else 'pending'
                        integers[state] += 1
            print('INTEGER endpoints: '+json.dumps(integers))
        return 0
    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            p = job['profile']
            print(f"{i:03}/{len(jobs):03} {job['id']} | {p['family']} | W{p['window_sz']} FF{p['dim_feedforward']} "
                  f"L{p['num_layers']} BS{p['batch_sz']} fixed={p['fixed_epoch_checkpoint']} max={p['epochs']}")
        return 0
    os.environ.update(TRANADPLUS_CUDA_PRECISION='faithful', TRANADPLUS_ADAM_IMPL='standard',
                      TRANADPLUS_GPU_CACHE='auto', TRANADPLUS_DATA_ROOT=str(args.data_root.resolve()),
                      TOAST_NO_TQDM='1' if args.no_progress else '0',
                      TOAST_FORCE_TQDM='0' if args.no_progress else '1')
    from tranad_plus.core.bootstrap import activate_baseline
    activate_baseline()
    import torch
    from importlib.metadata import version
    from gpu_settings import configure
    from tranad_plus.core.data import inventory
    from tranad_plus.campaigns.exploration.progress import bar, logged
    from tranad_plus.campaigns.precision.resources import analytic_operations
    from tranad_plus.core.training.engine import TrainingInterrupted
    from .data import load_data
    from .training import train, configured_model
    from .evaluation import evaluate
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    configure()
    shared = [PROJECT/'tranad_plus/campaigns'/p for p in (
        'temporal/evaluation.py', 'temporal/progress.py', 'exploration/progress.py',
        'thresholds/metrics.py', 'precision/resources.py')]
    shared.extend((PROJECT/'tranad_plus/campaigns/integer').glob('*.py'))
    sources = {'core': source_manifest(), 'campaign': {p.name: hash_file(p)
                for p in sorted(Path(__file__).parent.glob('*.py'))},
               'shared': {str(p.relative_to(PROJECT)): hash_file(p) for p in sorted(shared)}}
    environment = {'torch': str(torch.__version__), 'python': platform.python_version(),
                   'device': str(device), 'cpu_threads': args.cpu_threads,
                   'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   **{p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'pandas')}}
    files_by_dataset, sizes = {}, []
    for job in jobs:
        profile = job['profile']
        ds = profile['dataset']
        if ds not in files_by_dataset:
            files_by_dataset[ds] = inventory(profile, args.data_root)
            bundle = load_data(profile, args.data_root, files_by_dataset[ds],
                               validation_fraction=spec['native_validation_fraction'])
            print(f"CHECK {ds}: train={len(bundle.datasets['train'])} validation={len(bundle.datasets['validation'])} "
                  f"test={len(bundle.datasets['test'])} windows | validation classes={sorted(set(bundle.labels['validation'].tolist()))}", flush=True)
            del bundle
        model = configured_model(profile, job['arm'])
        sizes.append(sum(v.numel()*v.element_size() for v in model.state_dict().values()))
    storage_estimate = 10*sum(sizes)+sum(512*1024+j['profile']['epochs']*2048 for j in jobs)
    print(f'Conservative archive planning allowance: {storage_estimate/1024**2:.1f} MiB '
          '(models/recovery/scalar reports; no pointwise archives).', flush=True)
    if storage_estimate > spec['storage']['max_planned_gib']*1024**3:
        raise RuntimeError('Selected plan exceeds the compact-storage allowance; run a smaller stage/dataset filter')
    if args.check:
        print('CHECK PASS: all selected data/splits/model profiles/source hashes; no files written.')
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if args.audit_existing:
        with run_lock(output):
            audit_existing(jobs, args, output, device)
        print('Existing OPS-SAT final audit saved; native historical checkpoints were not trained with this holdout.')
        return 0
    if args.with_integer or args.quantize_only:
        from tranad_plus.campaigns.integer.kernel import preflight
        from .integer import evaluate_integer
        integer_preflight = preflight(device)
        write_json(output/'integer_preflight.json', integer_preflight)
    failed, pending = [], []
    with run_lock(output), bar(jobs, total=len(jobs), desc='V19 '+args.stage.upper(), unit='model') as progress:
        for index, job in enumerate(progress, 1):
            folder = output/job['id']
            files = files_by_dataset[job['profile']['dataset']]
            manifest = {'profile': job['profile'], 'arm': job['arm'], 'spec': spec,
                        'sources': sources, 'files': files, 'environment': environment,
                        'stop_on_patience': args.stop_on_patience}
            signature = digest(manifest)
            existing = folder/'status.json'
            if existing.exists() and json.loads(existing.read_text())['signature'] != signature:
                raise RuntimeError('Refusing to overwrite changed-provenance results; use a new output root: '+str(folder))
            try:
                verified = complete(folder, signature)
                if args.quantize_only and not verified:
                    pending.append(job['id'])
                    print('PENDING FP32 parent (no training in quantize-only): '+job['id'], flush=True)
                    continue
                if verified and not (args.with_integer or args.quantize_only):
                    print(f'[{index}/{len(jobs)}] SKIP verified {job["id"]}', flush=True)
                    continue
                with run_lock(folder):
                    bundle = load_data(job['profile'], args.data_root, files,
                                       validation_fraction=spec['native_validation_fraction'])
                    if not verified:
                        write_json(folder/'config.json', job['profile'])
                        write_json(folder/'manifest.json', manifest)
                        write_json(folder/'status.json', {'status': 'running', 'signature': signature, 'updated': now()})
                        with logged(folder/'run.log'):
                            print(f"[{index:03}/{len(jobs):03}] START {job['id']} | TRAIN -> EVALUATE checkpoints", flush=True)
                            write_json(folder/'data_inventory.json', bundle.inventory)
                            trained = train(job['profile'], job['arm'], bundle, folder, signature, spec, device,
                                            stop_on_patience=args.stop_on_patience, stop_after_epoch=args.stop_after_epoch)
                            for checkpoint in trained['snapshots']:
                                target = folder/'evaluation'/checkpoint/'result.json'
                                if target.exists():
                                    continue
                                saved = torch.load(folder/'checkpoints'/f'{checkpoint}.ckpt', map_location='cpu', weights_only=True)
                                if saved['signature'] != signature:
                                    raise RuntimeError('Checkpoint signature mismatch')
                                model = configured_model(job['profile'], job['arm']).to(device)
                                model.load_state_dict(saved['model_state'], strict=True)
                                evaluate(model, bundle, job['profile'], target.parent, device, spec)
                            write_json(folder/'resources.json', {'parameters': sum(p.numel() for p in model.parameters()),
                                       'state_bytes': sum(v.numel()*v.element_size() for v in
                                                          list(model.parameters())+list(model.buffers())),
                                       'state_bytes_scope': 'Unique parameters and buffers; shared decoder aliases counted once',
                                       'analytic_operations': analytic_operations(job['profile']),
                                       'operation_note': 'Analytic inference MACs; no measured training FLOP claim',
                                       'files_bytes_before_status': sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())})
                        artifacts = {str(p.relative_to(folder)): hash_file(p) for p in folder.rglob('*')
                                     if p.is_file() and 'integer' not in p.relative_to(folder).parts
                                     and p.name not in ('status.json', '.lock', 'run.log', 'progress.json', 'error.json')}
                        write_json(folder/'status.json', {'status': 'complete', 'signature': signature,
                                                         'updated': now(), 'artifacts': artifacts})
                        verified = True
                    if args.with_integer or args.quantize_only:
                        evaluate_integer(job, bundle, folder, signature, spec, device)
                summarize(output)
            except TrainingInterrupted:
                write_json(folder/'status.json', {'status': 'interrupted', 'signature': signature, 'updated': now()})
                return 75
            except Exception as exc:
                folder.mkdir(parents=True, exist_ok=True)
                error_name = 'integer_error.json' if verified or args.quantize_only else 'error.json'
                write_json(folder/error_name, {'reason': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
                if not verified and not args.quantize_only:
                    write_json(folder/'status.json', {'status': 'failed', 'signature': signature, 'updated': now()})
                failed.append(job['id'])
                print('FAILED '+job['id']+': '+str(exc), flush=True)
                if args.fail_fast:
                    raise
    summarize(output)
    status = {'updated': now(), 'stage': args.stage, 'selected': len(jobs), 'failed': failed,
              'missing_fp32_parents': pending, 'status': 'failed' if failed else 'pending' if pending else 'complete'}
    write_json(output/('integer_campaign_status.json' if args.quantize_only else 'campaign_status.json'), status)
    return 1 if failed else 2 if pending else 0


if __name__ == '__main__':
    raise SystemExit(main())


