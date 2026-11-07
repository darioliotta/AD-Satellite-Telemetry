#!/usr/bin/env python3
"""V14: first-ablation windows, frozen context, and 50->100 epoch controls."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
if __package__ in (None, ""):
    __package__ = "tranad_v14"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "toast-diagnostics-mpl"))

from tranad_variants.bootstrap import activate_baseline
from tranad_variants.common import digest, hash_file, now, run_lock, source_manifest, write_json
from .progress import bar, logged, scoring_bars, record


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "windows", "context", "continue", "both"), default="all")
    parser.add_argument("--plan", type=Path, default=Path(__file__).parent / "plan.json")
    parser.add_argument("--results-root", type=Path, default=PROJECT / "results", help="Parent of original v10/v12 studies")
    parser.add_argument("--parent-root", type=Path, default=PROJECT / "results/v13/architectures")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "results/v14")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--dataset", choices=("OPS-SAT", "SMAP", "MSL"), action="append",
                        help="Defaults to all three datasets; repeat to select a subset")
    parser.add_argument("--variant", choices=("reference", "encoder_only"), action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=48)
    parser.add_argument("--gpu-cache", choices=("auto", "off"), default="auto")
    parser.add_argument("--extra-epochs", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="List jobs without importing torch or writing files")
    parser.add_argument("--check", action="store_true", help="Verify source, input, and parent artifacts; no writes")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal bars; timing JSON/logs are still saved")
    parser.add_argument("--stop-after-epoch", type=int, help="Recovery check: save and exit 75 after this absolute epoch")
    return parser.parse_args(argv)


def plan(args):
    spec = json.loads(args.plan.read_text())
    if spec.get("schema_version") != 1 or spec.get("name") != "v14_exploration":
        raise ValueError("Unsupported V14 campaign plan")
    jobs = []
    stages = ("context", "continue") if args.stage in ("both", "all") else (() if args.stage == "windows" else (args.stage,))
    for stage in stages:
        settings = spec["context" if stage == "context" else "continuation"]
        datasets = args.dataset or settings["datasets"]
        configurations = settings["configurations"]
        for ds in datasets:
            for configuration in configurations:
                for seed in args.seed or settings["seeds"]:
                    for variant in args.variant or settings["variants"]:
                        relative = Path(ds) / variant / configuration / f"seed_{seed}"
                        jobs.append((stage, relative))
    return jobs


def validate_output(args):
    output = args.output_root.resolve()
    for protected in (args.parent_root.resolve(), args.data_root.resolve(),
                      PROJECT / "tranad_plus", PROJECT / "tranad_variants", PROJECT / "tranad_v14",
                      args.results_root.resolve() / "v10", args.results_root.resolve() / "v12"):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f"Output overlaps protected source/data/parent tree: {protected}")
    return output


def read_parent(parent, sources, files, *, recovery=False):
    from tranad_variants.evaluation.engine import verified_metrics
    report = json.loads((parent / "metrics.json").read_text())
    manifest = json.loads((parent / "manifest.json").read_text())
    if report["profile"] != json.loads((parent / "config.json").read_text()):
        raise RuntimeError("Parent profile mismatch")
    verified_metrics(parent, report["signature"])
    old = manifest["sources"]
    if old["baseline"] != sources["baseline"]:
        raise RuntimeError("Parent numerical baseline differs")
    # The original 72 and added 168 differ in planning.py only. Verify every
    # numerical source and require the same source-file set otherwise.
    numerical = lambda m: {k: v for k, v in m["variants"].items() if k != "planning.py"}
    if numerical(old) != numerical(sources):
        raise RuntimeError("Parent variant numerical implementations differ")
    if manifest["input_files"] != files:
        raise RuntimeError("Local input data differs from parent hashes/file order")
    if manifest["evaluation_settings"] != report["evaluation_settings"]:
        raise RuntimeError("Parent evaluation settings differ")
    if report["profile"]["epochs"] != 50:
        raise RuntimeError("This study expects the completed 50-epoch v13 parents")
    if recovery and not (parent / "checkpoints/resume.ckpt").is_file():
        raise FileNotFoundError(f"Recovery checkpoint required: {parent / 'checkpoints/resume.ckpt'}")
    return report, manifest


def completed(folder, signature):
    path = folder / "diagnostic_status.json"
    if not path.exists():
        return False
    status = json.loads(path.read_text())
    if status["signature"] != signature:
        raise RuntimeError("Existing diagnostic signature differs")
    if status["status"] != "complete":
        return False
    for name, expected in status["artifacts"].items():
        if hash_file(folder / name) != expected:
            raise RuntimeError(f"Diagnostic artifact changed: {folder / name}")
    return True


def validate_execution(old, current):
    for key in ("torch", "cuda", "device", "cpu_threads", "gpu_cache", "precision", "optimizer", "gpu",
                "python", "numpy", "pandas", "sklearn", "scipy"):
        if old[key] != current[key]:
            raise RuntimeError(f"Continuation execution differs from parent: {key}: {old[key]} vs {current[key]}")


def window_phase(args, output, sources, spec, campaign, save_campaign):
    from .windows import entries, run_window_study
    settings = spec['window_study']
    parents = entries(args.results_root, settings, args.dataset, args.seed)
    models_sha = sources['baseline']['TranADPlus/src/models.py']
    folder = output / 'window_study'
    if args.check:
        result = run_window_study(args.results_root, args.data_root, folder, settings, models_sha,
                                  datasets=args.dataset, seeds=args.seed, check=True)
        print(f"CHECK OK V14 windows: {result['checked_runs']} original runs", flush=True)
        return
    binding = {'stage': 'windows', 'settings': settings,
               'sources': campaign['diagnostic_sources'],
               'parents': {str(p.relative_to(args.results_root.resolve())): hash_file(p) for _, p, _ in parents}}
    signature = digest(binding)
    with run_lock(folder):
        manifest = folder / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
            raise RuntimeError('Existing V14 window study differs from its plan/source/input reports')
        if completed(folder, signature):
            campaign['runs'].append({'stage': 'windows', 'status': 'skipped_verified_complete'})
            save_campaign()
            return
        if not manifest.exists():
            write_json(manifest, {'signature': signature, 'created': now(), 'binding': binding})
        with logged(folder / 'run.log'):
            result = run_window_study(args.results_root.resolve(), args.data_root, folder, settings,
                                      models_sha, datasets=args.dataset, seeds=args.seed)
        record(folder, 'complete', **result)
        artifacts = {p.name: hash_file(p) for p in sorted(folder.iterdir())
                     if p.suffix in ('.csv', '.pdf', '.png') or p.name in ('manifest.json', 'provenance.json', 'README.md', 'progress.json')}
        write_json(folder / 'diagnostic_status.json', {'signature': signature, 'status': 'complete',
                                                      'artifacts': artifacts, 'updated': now()})
        campaign['runs'].append({'stage': 'windows', 'status': 'complete', **result})
        save_campaign()


def main(argv=None):
    args = arguments(argv)
    if args.extra_epochs < 1 or args.cpu_threads < 1:
        raise ValueError('Extra epochs and CPU threads must be positive')
    if args.no_progress:
        os.environ['TOAST_NO_TQDM'] = '1'
    else:
        os.environ['TOAST_FORCE_TQDM'] = '1'
    output = validate_output(args)
    jobs = plan(args)
    spec = json.loads(args.plan.read_text())
    window_selected = args.stage in ('all', 'windows')
    print('V14 | Window-size diagnostics, context sensitivity, training duration', flush=True)
    print(f"Window study: {'selected (original v10/v12)' if window_selected else 'off'} | "
          f"context jobs: {sum(s == 'context' for s, _ in jobs)} | "
          f"continuations: {sum(s == 'continue' for s, _ in jobs)} | "
          f'additional epochs: {args.extra_epochs}', flush=True)
    print(f'Output: {output}', flush=True)
    if args.dry_run:
        if window_selected:
            from .windows import entries
            print(f"WINDOWS: {len(entries(args.results_root, spec['window_study'], args.dataset, args.seed))} saved runs; no training", flush=True)
        for i, (stage, relative) in enumerate(jobs, 1):
            print(f'{i:03d}/{len(jobs):03d} {stage}: {relative}', flush=True)
        return 0
    sources = source_manifest()
    diagnostic_sources = {p.name: hash_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))
                          if not p.name.startswith('test_')}
    diagnostic_sources['plan.json'] = hash_file(args.plan)
    campaign = {'name': 'v14_exploration', 'started': now(), 'status': 'running',
                'runs': [], 'planned_jobs': len(jobs) + int(window_selected),
                'diagnostic_sources': diagnostic_sources}
    report_path = output / '_campaign_logs' / ('v14_' + now().replace(':', '-') + '.json')
    def save_campaign():
        if not args.check:
            write_json(report_path, campaign)
            write_json(output / 'campaign_latest.json', campaign)
    if window_selected:
        try:
            window_phase(args, output, sources, spec, campaign, save_campaign)
        except Exception as exc:
            campaign.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
            save_campaign()
            traceback.print_exc()
            return 1
    if not jobs:
        campaign.update(status='complete', finished=now())
        save_campaign()
        return 0
    activate_baseline()
    os.environ['TRANADPLUS_CUDA_PRECISION'] = 'faithful'
    os.environ['TRANADPLUS_ADAM_IMPL'] = 'standard'
    os.environ['TRANADPLUS_GPU_CACHE'] = args.gpu_cache
    os.environ['TRANADPLUS_DATA_ROOT'] = str(args.data_root.resolve())
    import torch
    from tranad_variants.data import inventory, load_data
    from tranad_variants.models import build_model
    from tranad_variants.training.engine import load_checkpoint, seed_all, TrainingInterrupted
    from .context import run_context
    from .continuation import run_continuation, validate_recovery
    from gpu_settings import configure
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if not args.check:
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Use CPU or CUDA')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU execution must be requested explicitly')
        configure()
    execution = {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'device': str(device),
                 'cpu_threads': args.cpu_threads, 'gpu_cache': args.gpu_cache,
                 'precision': 'fp32_tf32_disabled', 'optimizer': 'Adam_standard',
                 'python': platform.python_version(), 'numpy': version('numpy'), 'pandas': version('pandas'),
                 'sklearn': version('scikit-learn'), 'scipy': version('scipy'),
                 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' and torch.cuda.is_available() else None}
    cache = {}
    with bar(jobs, total=len(jobs), desc='V14 | checkpoint jobs', position=0) as job_bar:
        for index, (stage, relative) in enumerate(job_bar, 1):
            parent = args.parent_root.resolve() / relative
            folder = output / ('continuation' if stage == 'continue' else 'context') / relative
            started = time.monotonic()
            try:
                profile = json.loads((parent / 'config.json').read_text())
                if (profile['dataset'], profile['variant'], profile['configuration'], f"seed_{profile['seed']}") != relative.parts:
                    raise RuntimeError('Parent profile does not match its folder')
                files = inventory(profile, args.data_root)
                report, manifest = read_parent(parent, sources, files, recovery=stage == 'continue')
                if stage == 'continue':
                    if not args.check or device.type == torch.device(manifest['execution']['device']).type:
                        validate_execution(manifest['execution'], execution)
                if args.check:
                    load_checkpoint(parent / 'checkpoints/final.ckpt', profile, report['signature'], final=True)
                    if stage == 'continue':
                        validate_recovery(parent, profile, report['signature'])
                    continue
                binding = {'stage': stage, 'parent_signature': report['signature'],
                           'parent_artifacts': report['artifacts'], 'profile': profile,
                           'diagnostic_sources': diagnostic_sources, 'execution': execution,
                           'extra_epochs': args.extra_epochs if stage == 'continue' else None,
                           'parent_recovery_sha256': hash_file(parent / 'checkpoints/resume.ckpt') if stage == 'continue' else None}
                signature = digest(binding)
                with run_lock(folder):
                    existing = folder / 'manifest.json'
                    if existing.exists() and json.loads(existing.read_text())['signature'] != signature:
                        raise RuntimeError('Existing V14 diagnostic differs in configuration/source/environment')
                    if completed(folder, signature):
                        campaign['runs'].append({'stage': stage, 'run': str(relative), 'status': 'skipped_verified_complete'})
                        save_campaign()
                        continue
                    if not existing.exists():
                        write_json(existing, {'signature': signature, 'created': now(), 'binding': binding,
                                              'parent_folder': str(parent), 'settings': report['evaluation_settings']})
                    key = digest({'profile': {k: v for k, v in profile.items() if k not in ('variant', 'identifier', 'seed')}, 'files': files})
                    if key not in cache:
                        cache[key] = load_data(profile, args.data_root, files)
                    bundle = cache[key]
                    write_json(folder / 'diagnostic_status.json', {'signature': signature, 'status': 'running', 'updated': now()})
                    with logged(folder / 'run.log'), scoring_bars():
                        print(f'START [{index}/{len(jobs)}] {stage} {relative}', flush=True)
                        if stage == 'context':
                            saved = load_checkpoint(parent / 'checkpoints/final.ckpt', profile, report['signature'], final=True)
                            seed_all(profile['seed'])
                            model = build_model(profile).to(device)
                            model.load_state_dict(saved['model_state'], strict=True)
                            run_context(model, bundle, parent / 'scores.npz', folder, profile, device,
                                        report['evaluation_settings'], args.data_root)
                            artifact_names = ('metrics.json', 'scores.npz', 'manifest.json')
                        else:
                            run_continuation(parent, folder, report, signature, bundle, device,
                                             report['evaluation_settings'], args.extra_epochs, args.stop_after_epoch)
                            artifact_names = ('metrics.json', 'comparison.json', 'scores.npz', 'checkpoints/final.ckpt',
                                              'checkpoints/resume.ckpt', 'epochs.jsonl', 'config.json', 'manifest.json')
                    record(folder, 'complete', elapsed_session_seconds=time.monotonic()-started)
                    artifact_names += ('progress.json',)
                    write_json(folder / 'diagnostic_status.json', {'signature': signature, 'status': 'complete', 'updated': now(),
                               'artifacts': {n: hash_file(folder / n) for n in artifact_names}})
                    campaign['runs'].append({'stage': stage, 'run': str(relative), 'status': 'complete',
                                             'elapsed_seconds': time.monotonic()-started})
            except TrainingInterrupted as exc:
                campaign['runs'].append({'stage': stage, 'run': str(relative), 'status': 'interrupted_after_checkpoint'})
                campaign.update(status='interrupted_after_checkpoint', reason=str(exc))
                save_campaign()
                return 75
            except Exception as exc:
                traceback.print_exc()
                campaign['runs'].append({'stage': stage, 'run': str(relative), 'status': 'failed', 'reason': str(exc)})
                campaign.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
                save_campaign()
                return 1
            save_campaign()
    if args.check:
        print(f'CHECK OK V14: {len(jobs)} checkpoint jobs', flush=True)
    else:
        campaign.update(status='complete', finished=now())
        save_campaign()
        from .summarize import summarize
        summarize(output)
        print(f'V14 complete: {report_path}', flush=True)
    return 0


if __name__ == "__main__":
    # Support both `python -m tranad_v14.run` and the short file command.
    if __package__ in (None, ""):
        __package__ = "tranad_v14"
    raise SystemExit(main())
