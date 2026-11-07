#!/usr/bin/env python3
"""Single pre-training A100 profiling entrypoint for the THREE TOAST datasets.

Runs training-step profiler and test-set 21-alpha forward profiler, then builds
CSV, JSON summary and throughput PNG. Does not launch campaign training or
write model checkpoints. Requires CUDA. Run only when GPU is idle.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASETS=('OPS-SAT','SMAP','MSL')


def tabulate(outdir):
    rows=[]
    for name in DATASETS:
        slug=name.lower().replace('-','')
        for phase,filename in (('train',f'{slug}_train.json'),('evaluation',f'{slug}_eval.json')):
            path=outdir/filename
            if not path.exists():continue
            report=json.loads(path.read_text())
            for item in report.get('results',[]):
                rows.append({'dataset':name,'phase':phase,'mode':item.get('mode',''),
                             'batch':item.get('batch_size',report.get('batch_size','')),
                             'windows_per_second':item.get('windows_per_second',''),
                             'milliseconds_per_step':item.get('ms_per_step',item.get('milliseconds_per_step','')),
                             'peak_allocated_gib':item.get('cuda_peak_allocated_gib',item.get('max_allocated_gib','')),
                             'error':item.get('error',item.get('skipped',''))})
    dest=outdir/'a100_summary.csv'
    with dest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','phase','mode','batch','windows_per_second','milliseconds_per_step','peak_allocated_gib','error'])
        writer.writeheader();writer.writerows(rows)
    (outdir/'a100_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def chart(outdir,rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[PREFLIGHT] Matplotlib missing: skip PNG (CSV and JSON available)',flush=True)
        return
    for phase in ('train','evaluation'):
        subset=[r for r in rows if r['phase']==phase and isinstance(r['windows_per_second'],(float,int))]
        if not subset:continue
        fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for ax,name in zip(axs,DATASETS):
            items=[r for r in subset if r['dataset']==name]
            ax.barh([r['mode'] for r in items],[r['windows_per_second'] for r in items])
            ax.set_title(name)
            ax.set_xlabel('Test/train windows per second')
            ax.grid(axis='x',alpha=.2)
        fig.suptitle('A100 throughput | '+phase+' | TOAST v12',weight='bold')
        fig.savefig(outdir/f'a100_{phase}_throughput.png',dpi=180)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--cache-max-gib',type=float,default=16)
    p.add_argument('--include-alpha-recompute',action='store_true')
    p.add_argument('--skip-training',action='store_true')
    p.add_argument('--skip-evaluation',action='store_true')
    args=p.parse_args()
    if args.steps<1 or args.warmup<0 or not (0<args.cache_max_gib<=32):p.error('Invalid profiler arguments')
    if args.skip_training and args.skip_evaluation:p.error('Both phases cannot be skipped')
    import torch
    if not args.device.startswith('cuda') or not torch.cuda.is_available():p.error('A CUDA GPU is required')
    from project_paths import OUTPUT_ROOT
    destination=args.output_dir or OUTPUT_ROOT/'v12'/'profiling'/'preflight_a100'
    destination.mkdir(parents=True,exist_ok=True)
    manifest={'gpu':torch.cuda.get_device_name(args.device),'torch':torch.__version__,
              'datasets':list(DATASETS),'warmup':args.warmup,'steps':args.steps,
              'note':'Measurements only; no retraining, no F1/POT and no checkpoint changes'}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for dataset in DATASETS:
        slug=dataset.lower().replace('-','')
        jobs=[]
        if not args.skip_training:
            jobs.append(('train','profile_a100.py',f'{slug}_train.json'))
        if not args.skip_evaluation:
            jobs.append(('evaluation','profile_evaluation_a100.py',f'{slug}_eval.json'))
        for label,script,filename in jobs:
            cmd=[sys.executable,str(ROOT/'tools'/script),'--dataset',dataset,
                 '--device',args.device,'--warmup',str(args.warmup),'--steps',str(args.steps),
                 '--output',str(destination/filename)]
            cmd.extend(['--gpu-cache-max-gib' if label=='train' else '--cache-max-gib',str(args.cache_max_gib)])
            if label=='evaluation' and args.include_alpha_recompute:cmd.append('--compare-recompute')
            print(f'\n[PREFLIGHT] {dataset}: {label} | launching {script}',flush=True)
            try:subprocess.run(cmd,check=True,cwd=ROOT)
            except subprocess.CalledProcessError as exc:
                print(f'[PREFLIGHT] FAILURE in {dataset} {label}: exit {exc.returncode}',flush=True)
                raise
    rows=tabulate(destination)
    chart(destination,rows)
    print('\n[PREFLIGHT] Completed. Summary:',destination/'a100_summary.csv',flush=True)
    print('[PREFLIGHT] No full training campaign was launched.',flush=True)

if __name__=='__main__':main()
