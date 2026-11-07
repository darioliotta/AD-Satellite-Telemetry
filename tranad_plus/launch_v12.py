#!/usr/bin/env python3
"""TOAST v12 campaign runner. Stages: check, dry-run, train, evaluate, all.

Preserves original TranAD+ architecture, optimizer and training loss. Evaluation
runs ONLY after the training campaign is complete (unless --only is specified
for evaluation, which is intentionally unsupported). Re-running is resumable.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'experiment_plan_v12.json'


def validated_plan():
    from campaign_v12 import make_plan, BASES, GRID, SEEDS
    found = json.loads(PLAN.read_text(encoding='utf-8'))
    if found != make_plan():
        raise ValueError('The frozen v12 plan has changed. Restore the supplied experiment_plan_v12.json.')
    rows = found['experiments']
    if len(rows) != 126 or len({e['identifier'] for e in rows}) != 126:
        raise ValueError('Expected exactly 126 distinct runs')
    if {e['seed'] for e in rows} != set(SEEDS) or {e['epochs'] for e in rows} != {50}:
        raise ValueError('Seed/epoch validation failed')
    expected = {'OPS-SAT': 42, 'SMAP': 42, 'MSL': 42}
    if {key: sum(x['base_profile'] == BASES[key]['base'] for x in rows) for key in BASES} != expected:
        raise ValueError('Unexpected dataset coverage in frozen v12 plan')
    if found['measurement']['paper_alpha_search'] is not True:
        raise ValueError('The 21-alpha search is mandatory in v12 evaluation')
    return found


def env_paths(args):
    output = (args.results_root or ROOT.parent / 'results').expanduser().resolve()
    data = (args.data_root or ROOT.parent / 'data').expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('The results directory cannot be inside the source directory')
    env = os.environ.copy()
    env.update(TRANADPLUS_RESULTS_VERSION='v12', TRANADPLUS_CAMPAIGN_SUBDIR='study',
               TRANADPLUS_OUTPUT_ROOT=str(output), TRANADPLUS_DATA_ROOT=str(data),
               TRANADPLUS_GPU_CACHE=args.gpu_cache,
               TRANADPLUS_CUDA_PRECISION=args.precision)
    env['TRANADPLUS_GPU_CACHE_MAX_GIB']=str(args.gpu_cache_max_gib)
    env['TRANADPLUS_ADAM_IMPL']=args.adam
    env.pop('TRANADPLUS_CAMPAIGN_PLAN', None)
    return env, data, output / 'v12' / 'study'


def decide_device(requested):
    if requested != 'auto':
        return requested
    import torch
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def ensure_execution_policy(folder, args, device, *, create):
    """Fail closed if a study mixes CUDA numerical policies under the same IDs.

    Cache limits and pinned memory change the delivery path, not the model's
    mathematical hyperparameters. Fused Adam and TF32 do affect arithmetic.
    """
    policy = {'version': 'v12', 'device': device, 'precision': args.precision,
              'adam': args.adam, 'plan_sha256': __import__('hashlib').sha256(PLAN.read_bytes()).hexdigest(), 'split_policy': 'v10_aligned_opssat_20pct_val_smap_msl_official_train_test'}
    file = folder / '_campaign_logs' / 'execution_policy.json'
    if file.exists():
        recorded = json.loads(file.read_text())
        if recorded != policy:
            raise RuntimeError(f'Execution policy mismatch at {file}. Existing={recorded}, requested={policy}. '
                               'Do not mix numerical modes under the same checkpoint IDs.')
    elif create:
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('x', encoding='utf-8') as stream:
            json.dump(policy, stream, indent=2)
            stream.write('\n')
    else:
        raise RuntimeError(f'Missing execution policy manifest: {file}')
    return file


def verify_all_checkpoints(plan, folder, env):
    # Must not silently evaluate an incomplete design or old checkpoint.
    os.environ.update({k: env[k] for k in ('TRANADPLUS_RESULTS_VERSION', 'TRANADPLUS_CAMPAIGN_SUBDIR',
                                              'TRANADPLUS_OUTPUT_ROOT', 'TRANADPLUS_DATA_ROOT')})
    from campaign_plan import register_plan_profiles
    from config import PROFILES, get_profile
    from run_experiments import checkpoint_metadata
    register_plan_profiles(PROFILES, PLAN)
    missing = []
    for entry in plan['experiments']:
        profile = get_profile(entry['identifier'])
        path = folder / profile['dataset'] / (profile['identifier'] + '.ckpt')
        if not path.is_file() or not path.stat().st_size:
            missing.append(path.name)
        else:
            checkpoint_metadata(path, profile)
    if missing:
        raise RuntimeError(f'Evaluation requires all 126 validated checkpoints; missing {len(missing)}. Examples: {missing[:5]}')


def write_complexity(plan, folder):
    from tools.report_stage1_complexity import report
    destination = folder / '_campaign_logs' / 'architecture_complexity.csv'
    rows = report(PLAN)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        with destination.open(newline='') as stream:
            old = list(csv.DictReader(stream))
        if old != [{k:str(v) for k,v in row.items()} for row in rows]:
            raise RuntimeError(f'Frozen complexity inventory mismatch: {destination}')
        return destination
    with destination.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return destination


def subprocess_run(args, env):
    print('Executing:', ' '.join(map(str,args)), flush=True)
    return subprocess.call(args, cwd=ROOT, env=env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--check', action='store_true')
    selection.add_argument('--dry-run', action='store_true')
    selection.add_argument('--stage', choices=('train', 'evaluate', 'all'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--gpu-cache', choices=('auto', 'off'), default='auto')
    parser.add_argument('--precision', choices=('faithful', 'tf32'), default='faithful',
                        help='faithful: upstream FP32, no TF32; tf32 is a separate numerical experiment')
    parser.add_argument('--gpu-cache-max-gib', type=float, default=16.0,
                        help='Upper bound for resident data cache (GiB); 0 < value <= 32')
    parser.add_argument('--adam', choices=('standard','fused'), default='standard',
                        help='fused is an OPT-IN numerical experiment for CUDA only')
    parser.add_argument('--only', action='append', help='Train only selected run IDs; not valid for evaluate/all')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--audit-splits', action='store_true', help='With --check: inspect real labels and train/test partition')
    args = parser.parse_args(argv)
    if not 0 < args.gpu_cache_max_gib <= 32:
        parser.error('--gpu-cache-max-gib must be in (0, 32]')
    if args.adam == 'fused' and not decide_device(args.device).startswith('cuda'):
        parser.error('--adam fused requires a CUDA device')
    if args.audit_splits and not args.check:
        parser.error('--audit-splits requires --check')
    if (args.only or args.keep_going) and args.stage != 'train' and not args.dry_run:
        parser.error('--only/--keep-going allowed only with --stage train or --dry-run')
    try:
        plan = validated_plan()
        ids = {e['identifier'] for e in plan['experiments']}
        if args.only and not set(args.only).issubset(ids):
            parser.error('Unknown run IDs: ' + str(sorted(set(args.only)-ids)))
        env,data,folder = env_paths(args)
        device = decide_device(args.device)
        print(f'TOAST v12 | {len(plan["experiments"])} training runs | v10-grid-completion x seed | {device} | {args.precision} | Adam={args.adam} | GPU-cache={args.gpu_cache}', flush=True)
        print(f'Code: {ROOT}\nData: {data}\nResults: {folder}', flush=True)
        if args.stage == 'evaluate':
            ensure_execution_policy(folder,args,device,create=False)
            verify_all_checkpoints(plan,folder,env)
            evaluation = dict(plan)
            evaluation['measurement'] = {**plan['measurement'], 'enabled': True, 'paper_alpha_search': True}
            with tempfile.TemporaryDirectory(prefix='toast12_eval_') as temporary:
                file = Path(temporary) / 'evaluate.json'
                file.write_text(json.dumps(evaluation, indent=2)+'\n')
                return subprocess_run([sys.executable, str(ROOT/'run_experiments.py'), '--plan', str(file),
                                        '--stage', 'evaluate', '--device', device], env)
        if args.check:
            code = subprocess_run([sys.executable, str(ROOT/'tools/preflight.py'), '--plan', str(PLAN),
                                    '--device', device], env)
            if code or not args.audit_splits:
                return code
            return subprocess_run([sys.executable, str(ROOT/'tools/audit_splits.py'),
                                   '--data-root', str(data), '--output',
                                   str(folder/'_audit'/'split_manifest.json')], env)
        command = [sys.executable, str(ROOT/'run_experiments.py'), '--plan', str(PLAN), '--device', device]
        if args.dry_run:
            command.append('--dry-run')
        else:
            if plan['measurement']['enabled']:
                raise RuntimeError('Training and evaluation must be separate stages')
            ensure_execution_policy(folder,args,device,create=True)
            print('Complexity:', write_complexity(plan,folder),flush=True)
            command += ['--stage','train']
            if args.keep_going:command += ['--keep-going']
        for name in args.only or ():
            command += ['--only',name]
        result = subprocess_run(command,env)
        if result or args.stage != 'all':
            return result
        # Dispatch independent process so env/profile/config snapshots are fresh.
        return subprocess_run([sys.executable, str(ROOT/'launch_v12.py'), '--stage','evaluate',
                               '--device',device, '--precision',args.precision,
                               '--data-root',str(data), '--results-root',str(folder.parents[1]),
                               '--gpu-cache',args.gpu_cache,'--gpu-cache-max-gib',str(args.gpu_cache_max_gib),
                               '--adam',args.adam],env)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2,f'TOAST v12 ERROR: {exc}\n')

if __name__ == '__main__':
    sys.exit(main())
