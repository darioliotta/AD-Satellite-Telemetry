#!/usr/bin/env python3
"""V15: paired history controls first; two faithful SMD replications last."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'toast-diagnostics-mpl'))
if __package__ in (None, ''):
    __package__ = 'tranad_v15'

from tranad_variants.common import digest, hash_file, now, run_lock, write_json, source_manifest


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('all', 'frozen', 'train', 'smd'), default='all')
    p.add_argument('--plan', type=Path, default=Path(__file__).parent / 'plan.json')
    p.add_argument('--output-root', type=Path, default=PROJECT / 'results/v15')
    p.add_argument('--parent-root', type=Path, default=PROJECT / 'results/v13/architectures')
    p.add_argument('--data-root', type=Path, default=PROJECT / 'data')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cpu-threads', type=int, default=8,
                   help='Moderate CPU threading avoids overhead for tiny OPS-SAT matrices')
    p.add_argument('--seed', type=int, action='append', help='Filter paired OPS/SMAP/MSL seeds; SMD uses its own predeclared seeds')
    p.add_argument('--only', help='Exact job id printed by --dry-run')
    p.add_argument('--check', action='store_true', help='Read-only source/data/parent/checkpoint preflight; CPU is sufficient')
    p.add_argument('--dry-run', action='store_true', help='Print the full plan without torch or writes')
    p.add_argument('--no-progress', action='store_true')
    p.add_argument('--fail-fast', action='store_true')
    p.add_argument('--stop-after-epoch', type=int, help='Recovery test: stop after saving this epoch, exit 75')
    args = p.parse_args(argv)
    if args.cpu_threads < 1 or (args.stop_after_epoch is not None and args.stop_after_epoch < 1):
        p.error('Thread count and stop epoch must be positive')
    return args


def build_plan(spec, args):
    if spec.get('schema_version') != 1 or spec.get('name') != 'v15_history_controls':
        raise ValueError('Unsupported V15 plan')
    if spec['evaluation']['alpha_steps'] != 21 or spec['evaluation']['paper_threshold_steps'] != 200:
        raise ValueError('V15 fixes the benchmark alpha and BF200 grids')
    expected_variants = {'reference', 'shared_decoder', 'single_pass', 'encoder_only'}
    if set(spec['variants']) != expected_variants or len(spec['variants']) != 4:
        raise ValueError('V15 requires all four architectures exactly once')
    benchmark = json.loads((Path(__file__).parent/'smd_benchmark.json').read_text())
    expected_smd = benchmark['profile']
    if spec['smd']['dataset'] != expected_smd['dataset'] or spec['smd']['features'] != benchmark['checkpoint_evidence']['features'] or spec['smd']['variant'] != 'reference':
        raise ValueError('SMD replicas require the original TranAD+ architecture and feature count')
    for key, expected in expected_smd.items():
        if key not in ('identifier', 'dataset') and spec['smd'].get(key) != expected:
            raise ValueError(f'SMD must match the archived benchmark: {key}={expected!r}')
    seeds = args.seed or spec['seeds']
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(spec['seeds']):
        raise ValueError('Select distinct predeclared paired seeds')
    base = json.loads((PROJECT / 'tranad_variants/configs/baselines.json').read_text())
    jobs = []
    def profile(ds, variant, config, seed):
        info = base['datasets'][ds]
        return {**base['training'], **info['training'], **info['configurations'][config],
                'dataset': ds, 'features': info['features'], 'variant': variant,
                'configuration': config, 'seed': seed}
    for ds in spec['frozen_datasets']:
        for config in spec['configurations']:
            for variant in spec['variants']:
                for seed in seeds:
                    rel = Path(ds) / variant / config / f'seed_{seed}'
                    jobs.append({'stage': 'frozen', 'id': 'frozen/'+str(rel), 'parent': str(rel)})
        if ds == spec['training_dataset']:
            # Cheap one-sample controls first; preserve FF and layer counts.
            for config, condition in [('no_history', 'original')] + [
                    (c, t) for c in spec['configurations'] for t in spec['training_conditions']]:
                for variant in spec['variants']:
                    for seed in seeds:
                        p = profile(ds, variant, 'reference' if config == 'no_history' else config, seed)
                        if config == 'no_history':
                            p['window_sz'] = spec['no_history_window']
                        p.update(configuration=config, input_condition=condition, epochs=spec['training_epochs'])
                        name = f"train/{ds}/{variant}/{config}/{condition}/seed_{seed}"
                        p['identifier'] = name.replace('/', '_')
                        jobs.append({'stage': 'train', 'id': name, 'profile': p})
    for seed in spec['smd_seeds']:
        p = {**spec['smd'], 'seed': seed, 'input_condition': 'original'}
        p['identifier'] = (f"v15_SMD_w{p['window_sz']}_ff{p['dim_feedforward']}_l{p['num_layers']}"
                           f"_bs{p['batch_sz']}_e{p['epochs']}_s{seed}")
        jobs.append({'stage': 'smd', 'id': f"smd/{p['configuration']}/seed_{seed}", 'profile': p})
    result = [j for j in jobs if (args.stage == 'all' or j['stage'] == args.stage)
              and (args.only is None or j['id'] == args.only)]
    if not result:
        raise ValueError('No selected V15 jobs')
    return result


def validate_output(args):
    output = args.output_root.expanduser().resolve()
    protected = [args.data_root.resolve(), args.parent_root.resolve()]
    protected += [PROJECT / name for name in ('tranad_plus', 'tranad_variants', 'tranad_v14', 'tranad_v15', 'tesi', 'docs')]
    protected += [PROJECT / 'results' / name for name in ('legacy', 'v10', 'v12', 'v13', 'v14')]
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('V15 output overlaps existing sources, data or earlier experiments')
    return output


def complete(folder, signature):
    path = folder / 'status.json'
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    if saved['signature'] != signature:
        raise RuntimeError(f'Existing V15 configuration/data/source/environment differs: {folder}')
    if saved['status'] != 'complete':
        return False
    for name, expected in saved['artifacts'].items():
        if hash_file(folder / name) != expected:
            raise RuntimeError(f'Changed V15 artifact: {folder / name}')
    return True


def reconcile_epochs(folder):
    """Remove telemetry ahead of the durable recovery checkpoint after a crash."""
    import torch
    recovery, history = folder / 'checkpoints/resume.ckpt', folder / 'epochs.jsonl'
    if not history.exists() or (folder / 'checkpoints/final.ckpt').exists():
        return
    count = torch.load(recovery, map_location='cpu', weights_only=True)['epoch']+1 if recovery.exists() else 0
    records = {}
    for line in history.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # Only a torn telemetry write; checkpoint remains authoritative.
        if row['epoch_one_based'] <= count:
            records[row['epoch_one_based']] = row
    if set(records) != set(range(1, count+1)):
        raise RuntimeError('Recovery telemetry is missing committed epochs')
    from tranad_variants.common import atomic_destination
    with atomic_destination(history) as stream:
        for index in sorted(records):
            stream.write(json.dumps(records[index])+'\n')


def execution_info(torch, args, device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' and torch.cuda.is_available() else None,
            'cpu_threads': args.cpu_threads, 'precision': 'fp32_tf32_disabled',
            'optimizer': 'Adam_standard', 'gpu_cache': 'auto',
            **{name: version(package) for name, package in (('numpy', 'numpy'), ('pandas', 'pandas'),
               ('sklearn', 'scikit-learn'), ('scipy', 'scipy'))}}


def main(argv=None):
    args = arguments(argv)
    output = validate_output(args)
    spec = json.loads(args.plan.read_text())
    jobs = build_plan(spec, args)
    counts = {stage: sum(j['stage'] == stage for j in jobs) for stage in ('frozen', 'train', 'smd')}
    print(f"V15 | {len(jobs)} jobs | frozen={counts['frozen']} OPS training={counts['train']} SMD training={counts['smd']}", flush=True)
    print(f"Frozen conditions per checkpoint: {len(spec['frozen_conditions'])}. TRAIN -> EVALUATE for each new model.", flush=True)
    print(f'Output: {output}', flush=True)
    for i, job in enumerate(jobs, 1):
        p = job.get('profile')
        detail = f" | W{p['window_sz']} FF{p['dim_feedforward']} L{p['num_layers']} BS{p['batch_sz']} epochs={p['epochs']}" if p else ''
        print(f"{i:03d}/{len(jobs):03d} {job['id']}{detail}", flush=True)
    if args.dry_run:
        return 0
    os.environ.update(TRANADPLUS_CUDA_PRECISION='faithful', TRANADPLUS_ADAM_IMPL='standard',
                      TRANADPLUS_GPU_CACHE='auto', TRANADPLUS_DATA_ROOT=str(args.data_root.resolve()),
                      TOAST_NO_TQDM='1' if args.no_progress else '0', TOAST_FORCE_TQDM='0' if args.no_progress else '1')
    from tranad_variants.bootstrap import activate_baseline
    activate_baseline()
    import torch
    from gpu_settings import configure
    from tranad_variants.data import inventory, load_data
    from tranad_variants.models import build_model
    from tranad_variants.training.engine import train, load_checkpoint, seed_all, TrainingInterrupted
    from tranad_v14.run import read_parent
    from .controls import changed_bundle
    from .evaluation import frozen_evaluation, trained_evaluation
    from .progress import bar, logged, record, scoring_bars, training
    from .summarize import summarize
    sources = source_manifest()
    own = {p.name: hash_file(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}
    own['plan.json'] = hash_file(args.plan)
    own['smd_benchmark.json'] = hash_file(Path(__file__).parent/'smd_benchmark.json')
    # Reused v14 provenance, progress and real-point helpers also enter binding.
    own.update({'v14/'+name: hash_file(PROJECT/'tranad_v14'/name)
                for name in ('run.py', 'progress.py', 'metrics.py')})
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('V15 supports CPU or CUDA')
    if not args.check:
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing an accidental overnight CPU campaign')
        configure()
    execution = execution_info(torch, args, device)
    report = {'name': spec['name'], 'started': now(), 'status': 'running', 'counts': counts,
              'planned_jobs': len(jobs), 'planned_ids': [j['id'] for j in jobs], 'runs': []}
    report_path = output / '_campaign_logs' / ('v15_'+now().replace(':', '-')+'.json')
    cache = {}
    def save_report():
        if not args.check:
            write_json(report_path, report)
            write_json(output / 'campaign_latest.json', report)
    def get_bundle(profile, files):
        key = digest({'profile': {k: v for k, v in profile.items() if k not in
                     ('identifier', 'seed', 'variant', 'configuration', 'input_condition')}, 'files': files})
        if key not in cache:
            cache.clear()  # Retain only the current data geometry; avoid accumulating native windows.
            cache[key] = load_data(profile, args.data_root, files)
        return cache[key]
    save_report()
    with bar(jobs, total=len(jobs), desc='V15 | campaign', position=0, unit='job') as progress:
        # Jobs range from seconds to 100 SMD epochs. A uniform-job ETA would
        # be misleading; epoch/batch bars retain their meaningful local ETAs.
        progress.bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed}{postfix}]'
        for index, job in enumerate(progress, 1):
            folder, started = output / job['id'], time.monotonic()
            item = {'id': job['id'], 'stage': job['stage'], 'started': now()}
            report['current_job'] = job['id']
            save_report()
            try:
                parent, parent_report = None, None
                if job['stage'] == 'frozen':
                    parent = args.parent_root.resolve() / job['parent']
                    profile = json.loads((parent / 'config.json').read_text())
                    expected = (profile['dataset'], profile['variant'], profile['configuration'], f"seed_{profile['seed']}")
                    if tuple(Path(job['parent']).parts) != expected:
                        raise RuntimeError('Parent folder/profile mismatch')
                    files = inventory(profile, args.data_root)
                    parent_report, _ = read_parent(parent, sources, files)
                    saved_parent = load_checkpoint(parent / 'checkpoints/final.ckpt', profile, parent_report['signature'], final=True)
                else:
                    profile = job['profile']
                    files = inventory(profile, args.data_root)
                if args.check:
                    bundle = get_bundle(profile, files)
                    print(f"CHECK OK {job['id']}: " + str({k: len(v) for k, v in bundle.datasets.items()}), flush=True)
                    item['status'] = 'checked'
                    continue
                binding = {'job': job, 'profile': profile, 'input_files': files,
                           'sources': sources, 'v15_sources': own, 'execution': execution, 'plan': spec,
                           'parent_artifacts': parent_report['artifacts'] if parent_report else None,
                           'parent_signature': parent_report['signature'] if parent_report else None}
                signature = digest(binding)
                with run_lock(folder):
                    manifest_path = folder / 'manifest.json'
                    if manifest_path.exists() and json.loads(manifest_path.read_text())['signature'] != signature:
                        raise RuntimeError('Existing V15 job differs; refusing overwrite')
                    if complete(folder, signature):
                        item['status'] = 'skipped_verified_complete'
                        continue
                    if not manifest_path.exists():
                        write_json(manifest_path, {'signature': signature, 'created': now(), 'binding': binding})
                        write_json(folder/'config.json', profile)
                    write_json(folder/'status.json', {'status': 'running', 'signature': signature, 'updated': now()})
                    bundle = get_bundle(profile, files)
                    write_json(folder/'data_inventory.json', bundle.inventory)
                    with logged(folder/'run.log'), scoring_bars():
                        print(f"\n[{index}/{len(jobs)}] START {job['id']}", flush=True)
                        if parent is not None:
                            seed_all(profile['seed'])
                            model = build_model(profile).to(device)
                            model.load_state_dict(saved_parent['model_state'], strict=True)
                            del saved_parent
                            metrics = frozen_evaluation(model, bundle, parent, parent_report, folder,
                                                        profile, device, spec, args.data_root)
                        else:
                            reconcile_epochs(folder)
                            with training(folder, profile):
                                train(profile, changed_bundle(bundle, profile['input_condition']), folder,
                                      signature, device, stop_after_epoch=args.stop_after_epoch)
                            saved = load_checkpoint(folder/'checkpoints/final.ckpt', profile, signature, final=True)
                            model = build_model(profile).to(device)
                            model.load_state_dict(saved['model_state'], strict=True)
                            del saved
                            metrics = trained_evaluation(model, bundle, folder, profile, device,
                                                         spec['evaluation'], args.data_root)
                            history = {r['epoch_one_based']: r for r in map(json.loads, (folder/'epochs.jsonl').read_text().splitlines())}
                            if set(history) != set(range(1, profile['epochs']+1)):
                                raise RuntimeError('Training telemetry does not cover all configured epochs')
                            metrics['training'] = {'epochs': len(history),
                                                   'seconds': sum(r['elapsed_seconds'] for r in history.values()),
                                                   'parameters': sum(p.numel() for p in model.parameters()),
                                                   'history_condition': profile['input_condition']}
                        write_json(folder/'metrics.json', {'status': 'complete', 'profile': profile,
                                                          'signature': signature, **metrics})
                        del model
                    record(folder, 'complete', seconds=time.monotonic()-started)
                    artifacts = {str(p.relative_to(folder)): hash_file(p) for p in sorted(folder.rglob('*'))
                                 if p.is_file() and p.suffix in ('.json', '.npz', '.ckpt', '.jsonl')
                                 and p.name != 'status.json'}
                    write_json(folder/'status.json', {'status': 'complete', 'signature': signature,
                                                      'updated': now(), 'artifacts': artifacts})
                    item['status'] = 'complete'
                    report['tables'] = summarize(output)
            except TrainingInterrupted as exc:
                item.update(status='interrupted_after_checkpoint', reason=str(exc))
                report['status'] = 'interrupted_after_checkpoint'
                return 75
            except KeyboardInterrupt:
                item['status'] = 'interrupted'
                report['status'] = 'interrupted'
                return 130
            except Exception as exc:
                item.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
                if args.fail_fast:
                    report['status'] = 'failed'
                    return 1
            finally:
                item.update(finished=now(), seconds=time.monotonic()-started)
                report['runs'].append(item)
                save_report()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    failures = sum(r['status'] == 'failed' for r in report['runs'])
    if not args.check:
        report['tables'] = summarize(output)
    report.update(status='completed_with_failures' if failures else ('checked' if args.check else 'complete'),
                  finished=now(), failures=failures, current_job=None)
    save_report()
    print(f"V15 {report['status']} | failures={failures} | {report_path}", flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
