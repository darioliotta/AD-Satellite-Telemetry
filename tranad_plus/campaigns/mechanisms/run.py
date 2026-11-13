"""V20: mechanism controls on OPS-SAT, SMAP and MSL, not a new benchmark claim."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
import warnings

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'toast-matplotlib'))
from tranad_plus.core.common import digest, hash_file, write_json, save_npz, now, run_lock, source_manifest


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('all', 'analysis', 'baselines', 'train'), default='all')
    p.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
    p.add_argument('--output-root', type=Path, default=PROJECT/'results/v20/mechanisms')
    p.add_argument('--data-root', type=Path, default=PROJECT/'data')
    p.add_argument('--parent-root', type=Path, default=PROJECT/'results/v13/architectures')
    p.add_argument('--integer-root', type=Path, default=PROJECT/'results/v17/integer')
    p.add_argument('--dataset', action='append', choices=('OPS-SAT', 'SMAP', 'MSL'))
    p.add_argument('--variant', action='append', choices=('reference', 'shared_decoder', 'single_pass', 'encoder_only'))
    p.add_argument('--seed', action='append', type=int)
    p.add_argument('--format', dest='formats', action='append', choices=('fp32', 'int8', 'int4'))
    p.add_argument('--only', help='Exact case id from --dry-run')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--dry-run', action='store_true'); p.add_argument('--check', action='store_true')
    p.add_argument('--status', action='store_true'); p.add_argument('--verify', action='store_true')
    p.add_argument('--no-progress', action='store_true'); p.add_argument('--fail-fast', action='store_true')
    p.add_argument('--stop-after-epoch', type=int, help='Intentional recovery test; exits 75')
    a = p.parse_args(argv)
    if a.cpu_threads < 1 or (a.stop_after_epoch is not None and a.stop_after_epoch < 1):
        p.error('Positive thread count/stop epoch required')
    return a


def plan(spec, args):
    if spec.get('name') != 'v20_detection_mechanisms' or spec.get('schema_version') != 1:
        raise ValueError('Unknown V20 plan')
    for key, required in (('datasets', {'OPS-SAT', 'SMAP', 'MSL'}),
                          ('variants', {'reference', 'shared_decoder', 'single_pass', 'encoder_only'}),
                          ('formats', {'fp32', 'int8', 'int4'}),
                          ('conditions', {'observed_current', 'hidden_current'})):
        if set(spec[key]) != required or len(spec[key]) != len(required):
            raise ValueError('Missing/duplicate '+key)
    if spec['evaluation']['fixed_alpha'] != .5 or spec['evaluation']['bf_threshold_steps'] != 200:
        raise ValueError('Predeclared alpha=.5 and BF200 required')
    if spec['storage']['save_pointwise_scores'] or spec['storage']['save_reconstructions']:
        raise ValueError('Compact score-summary archive required')
    selected = {'datasets': args.dataset or spec['datasets'], 'variants': args.variant or spec['variants'],
                'seeds': args.seed or spec['seeds'], 'formats': args.formats or spec['formats']}
    for key, value in selected.items():
        if len(value) != len(set(value)) or not set(value).issubset(spec[key]):
            raise ValueError('Unknown/duplicate '+key)
    base = json.loads((PROJECT/'tranad_plus/core/configs/baselines.json').read_text())
    jobs = []
    def profile(ds, variant, config, seed):
        info = base['datasets'][ds]
        return {**base['training'], **info['training'], **info['configurations'][config],
                'dataset': ds, 'features': info['features'], 'variant': variant,
                'configuration': config, 'seed': seed}
    for stage in ('analysis', 'baselines', 'train'):
        if args.stage not in ('all', stage):
            continue
        for ds in selected['datasets']:
            if stage == 'analysis':
                for config in spec['analysis_configurations']:
                    for variant in selected['variants']:
                        for seed in selected['seeds']:
                            for fmt in selected['formats']:
                                identity = f'{ds}/{variant}/{config}/seed_{seed}'
                                jobs.append({'id': f'analysis/{identity}/{fmt}', 'stage': stage, 'format': fmt,
                                             'profile': profile(ds, variant, config, seed), 'parent': identity})
            elif stage == 'baselines':
                for name in spec['baselines']:
                    p = profile(ds, 'reference', spec['training_configuration'], spec['seeds'][0])
                    jobs.append({'id': f'baselines/{ds}/{name}', 'stage': stage, 'format': 'fp32', 'baseline': name, 'profile': p})
            else:
                for variant in selected['variants']:
                    for seed in selected['seeds']:
                        for condition in spec['conditions']:
                            p = profile(ds, variant, spec['training_configuration'], seed)
                            p.update(epochs=spec['epochs'], condition=condition)
                            jobs.append({'id': f'train/{ds}/{variant}/seed_{seed}/{condition}',
                                         'stage': stage, 'format': 'fp32', 'profile': p})
    for job in jobs:
        if job['stage'] != 'analysis':
            job['profile']['identifier'] = 'v20_'+job['id'].replace('/', '_')
    jobs = [j for j in jobs if not args.only or args.only == j['id']]
    if not jobs:
        raise ValueError('No selected cases')
    return jobs


def output_path(args):
    out = args.output_root.expanduser().resolve()
    protected = [PROJECT/'tranad_plus', PROJECT/'docs', PROJECT/'tesi', args.data_root.resolve(),
                 args.parent_root.resolve(), args.integer_root.resolve()]
    protected += [PROJECT/'results'/name for name in ('legacy',)+tuple(f'v{i}' for i in range(10, 20))]
    if any(out == p or p in out.parents or out in p.parents for p in protected):
        raise ValueError('Output overlaps protected source/data/earlier results')
    return out


def completed(folder, signature=None):
    path = folder/'status.json'
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    if signature is not None and value['signature'] != signature:
        raise RuntimeError('Existing case source/data/configuration differs')
    if value['status'] != 'complete':
        return False
    for name, expected in value['artifacts'].items():
        p = folder/name
        if not p.resolve().is_relative_to(folder.resolve()) or hash_file(p) != expected:
            raise RuntimeError('Changed completed artifact: '+str(p))
    return True


def parent_record(job, args, files):
    from tranad_plus.campaigns.thresholds.provenance import verify_source, numerical_sources
    source = args.integer_root/job['parent']
    report, manifest = verify_source(source)
    p = report['profile']
    for key in ('dataset', 'variant', 'configuration', 'seed', 'window_sz', 'dim_feedforward', 'num_layers', 'batch_sz', 'features', 'lr'):
        if p[key] != job['profile'][key]:
            raise RuntimeError('Historical parent profile mismatch: '+key)
    if manifest['binding']['files'] != files:
        raise RuntimeError('Historical parent input file hashes/order differ')
    numerical_sources(manifest['binding']['sources'])
    for fmt in ('fp32', 'int8', 'int4'):
        path = args.parent_root/job['parent']/'checkpoints/final.ckpt' if fmt == 'fp32' else source/'models'/f'{fmt}.npz'
        if hash_file(path) != report['formats'][fmt]['storage']['packed_model_sha256']:
            raise RuntimeError('Historical model/export hash differs: '+str(path))
    return report, manifest


def identity(job):
    p = job['profile']; baseline = job.get('baseline')
    return {'stage': job['stage'], 'family': 'historical_real_samples' if job['stage'] == 'analysis' else 'controlled_full_windows',
            'dataset': p['dataset'], 'variant': baseline or p['variant'], 'configuration': p['configuration'],
            'window': p['window_sz'], 'ff': p['dim_feedforward'], 'condition': p.get('condition', 'not_applicable'),
            'seed': None if baseline else p['seed'], 'format': job['format']}


def main(argv=None):
    args = arguments(argv); spec = json.loads(args.plan.read_text())
    jobs = plan(spec, args); output = output_path(args)
    counts = {s: sum(j['stage'] == s for j in jobs) for s in ('analysis', 'baselines', 'train')}
    print(f"V20 | existing-model evaluations={counts['analysis']} | simple baselines={counts['baselines']} | NEW TRAININGS={counts['train']}", flush=True)
    print('Three datasets | four architectures | fixed alpha=.5 | score separation + calibrated alarms + BF200/POT diagnostics', flush=True)
    print('Historical and controlled data families remain separate. No dense F1 grid, no checkpoint sweep, no pointwise archive.', flush=True)
    print('Output: '+str(output), flush=True)
    if args.dry_run:
        for i, j in enumerate(jobs, 1):
            p = j['profile']
            print(f"{i:03}/{len(jobs):03} {j['id']} | W{p['window_sz']} FF{p['dim_feedforward']} "
                  f"BS{p['batch_sz']}"+(f" E{p['epochs']}" if j['stage'] == 'train' else ' | no training'))
        return 0
    if args.status or args.verify:
        done = 0
        for j in jobs:
            f = output/j['id']; status = f/'status.json'
            if status.exists() and json.loads(status.read_text()).get('status') == 'complete':
                if args.verify:
                    completed(f)
                done += 1
        print(f'Completed {done}/{len(jobs)} | pending {len(jobs)-done}')
        campaign = output/'campaign.json'
        if campaign.exists():
            print(campaign.read_text())
        return 1 if args.verify and done != len(jobs) else 0
    os.environ.update(TRANADPLUS_CUDA_PRECISION='faithful', TRANADPLUS_ADAM_IMPL='standard',
                      TRANADPLUS_GPU_CACHE='auto', TRANADPLUS_DATA_ROOT=str(args.data_root.resolve()),
                      TOAST_NO_TQDM='1' if args.no_progress else '0', TOAST_FORCE_TQDM='0' if args.no_progress else '1')
    from tranad_plus.core.bootstrap import activate_baseline
    activate_baseline()
    import numpy as np
    import torch
    from importlib.metadata import version
    from gpu_settings import configure
    from tranad_plus.core.data import inventory, load_data
    from tranad_plus.core.models import build_model
    from tranad_plus.core.training.engine import load_checkpoint, TrainingInterrupted
    from tranad_plus.campaigns.exploration.progress import bar, record
    from .progress import logged
    from tranad_plus.campaigns.precision.resources import analytic_operations
    from tranad_plus.campaigns.precision.storage import check_output_budget
    from .data import control_data, historical_layout
    from .control import build_control
    from .engine import train, stream_scores
    from .metrics import evaluate_scores
    from .reports import summarize
    torch.set_num_threads(args.cpu_threads); device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use CPU for read-only --check')
    configure()
    shared = list((PROJECT/'tranad_plus/campaigns/integer').glob('*.py'))
    shared += [PROJECT/'tranad_plus/campaigns'/n for n in ('thresholds/provenance.py', 'thresholds/metrics.py',
                                                        'precision/resources.py', 'precision/storage.py', 'exploration/progress.py')]
    sources = {'core': source_manifest(), 'campaign': {p.name: hash_file(p) for p in Path(__file__).parent.glob('*.py')},
               'shared': {str(p.relative_to(PROJECT)): hash_file(p) for p in shared}}
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                   'device': str(device), 'cpu_threads': args.cpu_threads,
                   'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   **{p: version(p) for p in ('numpy', 'scipy', 'pandas', 'scikit-learn')}}
    files, parents, control_checked, estimated_weights = {}, {}, set(), 0
    # Full read-only preflight: fail before training if earlier artifacts/data are missing.
    with bar(jobs, total=len(jobs), desc='V20 PREFLIGHT', unit='case') as progress:
        for job in progress:
            p = job['profile']; ds = p['dataset']
            if ds not in files:
                files[ds] = inventory(p, args.data_root)
            if job['stage'] == 'analysis' and job['parent'] not in parents:
                parents[job['parent']] = parent_record(job, args, files[ds])
            if job['stage'] != 'analysis' and ds not in control_checked:
                bundle, layouts = control_data(p, args.data_root, files[ds], spec['native_validation_fraction'])
                print(f"CONTROL {ds}: complete-window targets="+str({s: len(v.labels) for s, v in layouts.items()}), flush=True)
                del bundle, layouts
                control_checked.add(ds)
            if job['stage'] == 'train':
                m = build_control(p)
                estimated_weights += sum(x.numel()*x.element_size() for x in m.parameters())
    estimate = 6*estimated_weights+len(jobs)*512*1024+32*1024**2
    print(f'Conservative archive allowance: {estimate/1024**2:.1f} MiB; budget={spec["storage"]["max_planned_gib"]:g} GiB', flush=True)
    if estimate > spec['storage']['max_planned_gib']*1024**3:
        raise RuntimeError('Selected archive exceeds compact budget; select a smaller stage')
    if args.check:
        print('CHECK PASS: data hashes, historical weights, complete-window controls, numerical sources and budget. No files written.')
        return 0
    needs_integer = any(j['format'] != 'fp32' for j in jobs)
    if needs_integer:
        from tranad_plus.campaigns.integer.kernel import preflight, kernel_audit
        from tranad_plus.campaigns.integer.runtime import load as load_integer
        hardware = preflight(device)
    else:
        hardware = None
    output.mkdir(parents=True, exist_ok=True)
    failed = []; active_key, bundle, layouts = None, None, None
    campaign = {'started': now(), 'status': 'running', 'counts': counts, 'failed': [], 'current_case': None}
    with run_lock(output):
        if hardware is not None:
            write_json(output/'integer_preflight.json', hardware)
        for stage in ('analysis', 'baselines', 'train'):
            selected = [j for j in jobs if j['stage'] == stage]
            if not selected:
                continue
            with bar(selected, total=len(selected), desc='V20 '+stage.upper(), unit='case') as progress:
                for i, job in enumerate(progress, 1):
                    p = job['profile']; ds = p['dataset']; folder = output/job['id']
                    historical = parents.get(job.get('parent'))
                    if historical:
                        old, manifest = historical
                        actual = old['profile']
                        for key in ('identifier',):
                            if key in actual:
                                p[key] = actual[key]
                        model_path = (args.parent_root/job['parent']/'checkpoints/final.ckpt' if job['format'] == 'fp32'
                                      else args.integer_root/job['parent']/'models'/f"{job['format']}.npz")
                        parent_binding = {'signature': old['signature'], 'model_sha256': hash_file(model_path)}
                    else:
                        parent_binding = None
                    binding = {'identity': identity(job), 'profile': p, 'plan': spec, 'sources': sources,
                               'files': files[ds], 'execution': environment, 'parent': parent_binding}
                    signature = digest(binding)
                    previous = folder/'manifest.json'
                    if previous.exists() and json.loads(previous.read_text())['signature'] != signature:
                        raise RuntimeError('Refusing to overwrite different V20 provenance: '+str(folder))
                    try:
                        if completed(folder, signature):
                            print(f'[{i}/{len(selected)}] SKIP verified {job["id"]}', flush=True)
                            continue
                        with run_lock(folder):
                            write_json(folder/'manifest.json', {'signature': signature, 'binding': binding})
                            write_json(folder/'status.json', {'status': 'running', 'signature': signature, 'updated': now()})
                            write_json(folder/'config.json', p)
                            check_output_budget(output, spec['storage']['max_planned_gib'])
                            campaign['current_case'] = job['id']; write_json(output/'campaign.json', campaign)
                            key = (stage == 'analysis', ds, p['window_sz'])
                            if key != active_key:
                                if stage == 'analysis':
                                    bundle = load_data(p, args.data_root, files[ds])
                                    layouts = historical_layout(bundle, p, args.data_root)
                                else:
                                    bundle, layouts = control_data(p, args.data_root, files[ds], spec['native_validation_fraction'])
                                active_key = key
                            with logged(folder/'run.log'):
                                print(f"[{i:03}/{len(selected):03}] START {job['id']} | W{p['window_sz']} FF{p['dim_feedforward']} BS{p['batch_sz']}", flush=True)
                                write_json(folder/'data_inventory.json', bundle.inventory)
                                if stage == 'train':
                                    checkpoint = train(p, bundle, folder, signature, device, spec['evaluation'], args.stop_after_epoch)
                                    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                                    model = build_control(p); model.load_state_dict(saved['model_state'], strict=True)
                                    model = model.to(device).eval()
                                elif stage == 'analysis':
                                    if job['format'] == 'fp32':
                                        saved = load_checkpoint(model_path, actual, manifest['binding']['parent_signature'], final=True)
                                        model = build_model(actual); model.load_state_dict(saved['model_state'], strict=True)
                                        model = model.to(device).eval()
                                    else:
                                        model = load_integer(model_path, device)
                                else:
                                    model = None
                                alpha = spec['evaluation']['fixed_alpha'] if model is not None and model.output_count == 2 else None
                                scores, inference = {}, {}
                                for split, dataset in bundle.datasets.items():
                                    record(folder, 'scoring', case=job['id'], split=split, alpha=alpha)
                                    class_labels = np.where(layouts[split].real, layouts[split].labels, -1)
                                    scores[split], inference[split] = stream_scores(
                                        model, dataset, device, spec['evaluation']['batch_size_by_dataset'][ds], alpha,
                                        f'V20 {job["format"]} {split}', labels=class_labels, baseline=job.get('baseline'))
                                settings = {**spec['evaluation'], 'pot_q': p['pot_q']}
                                calibration_source = ('heldout_nominal_validation' if 'validation' in scores
                                                      else 'historical_native_optimization_scores_in_sample; not unseen validation')
                                record(folder, 'metrics', case=job['id'])
                                with bar(total=1, desc='V20 separation / calibrated alarms / BF200 / POT', position=1, unit='report', leave=False) as metrics_progress:
                                    with warnings.catch_warnings(record=True) as fit_warnings:
                                        warnings.simplefilter('always')
                                        measured, segment_arrays = evaluate_scores(scores, layouts, ds, settings, calibration_source)
                                    measured['fit_warnings'] = sorted({str(w.message) for w in fit_warnings})
                                    metrics_progress.update(1)
                                if model is None:
                                    resources = {'weight_representation_bytes': 0, 'model_tensor_bytes': 0,
                                                 'note': 'Deterministic baseline; input buffers/scaler not counted as learned model weights'}
                                else:
                                    parameter_bytes = sum(x.numel()*x.element_size() for x in model.parameters())
                                    if job['format'] != 'fp32':
                                        parameter_bytes += sum(x.numel()*x.element_size() for name, x in model.named_buffers()
                                                               if not name.endswith('.pe'))
                                    resources = {'weight_representation_bytes': parameter_bytes,
                                                 'model_tensor_bytes': sum(x.numel()*x.element_size() for x in list(model.parameters())+list(model.buffers())),
                                                 'analytic_operations': analytic_operations(p),
                                                 'operation_note': 'Inference matrix MAC estimate, excludes score calculation and optimizer/backward',
                                                 'original_architecture_parameters': sum(x.numel() for x in build_model(p).parameters())}
                                report = {'identity': identity(job), 'fixed_alpha': alpha, 'metrics': measured, 'resources': resources,
                                          'inference': inference, 'checkpoint_selection': 'fixed_horizon; validation never selects an epoch',
                                          'scope': 'Historical real-sample diagnostics' if stage == 'analysis' else 'Matched complete-window control, separate from original benchmark',
                                          'format_note': ('W4A4 on INT8 instructions, INT32 accumulation; other documented operations FP32'
                                                          if job['format'] == 'int4' else job['format'])}
                                if historical:
                                    reference = old['formats'][job['format']]['oracle']
                                    report['historical_paper_endpoints_context_only'] = {m: {k: v for k, v in (reference.get(m) or {}).items()
                                        if k in ('alpha', m+'_f1', m+'_precision', m+'_recall', m+'_threshold')} for m in ('bf', 'pot')}
                                write_json(folder/'result.json', report)
                                if segment_arrays is not None and spec['storage']['save_opssat_segment_scores']:
                                    save_npz(folder/'fragment_scores.npz', **segment_arrays)
                                del scores, model
                            artifacts = {str(q.relative_to(folder)): hash_file(q) for q in folder.rglob('*') if q.is_file()
                                         and q.name not in ('status.json', 'run.log', 'progress.json', 'error.json', '.lock')}
                            write_json(folder/'status.json', {'status': 'complete', 'signature': signature,
                                                             'updated': now(), 'artifacts': artifacts})
                        summarize(output)
                    except TrainingInterrupted:
                        write_json(folder/'status.json', {'status': 'interrupted', 'signature': signature, 'updated': now()})
                        campaign.update(status='interrupted', updated=now()); write_json(output/'campaign.json', campaign)
                        return 75
                    except Exception as exc:
                        write_json(folder/'error.json', {'reason': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
                        write_json(folder/'status.json', {'status': 'failed', 'signature': signature, 'updated': now()})
                        failed.append(job['id']); campaign['failed'] = failed
                        print('FAILED '+job['id']+': '+str(exc), flush=True)
                        if args.fail_fast:
                            raise
        summarize(output)
        campaign.update(status='failed' if failed else 'complete', failed=failed, updated=now())
        write_json(output/'campaign.json', campaign)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
