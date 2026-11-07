#!/usr/bin/env python3
"""Plot A100 measured speedups from three real-data baseline benchmark JSON files.

Fails closed if any dataset is missing, device isn't CUDA, or modes are absent.
Never mixes results from different GPUs or precision settings. Does not
produce synthetic latency claims.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DSETS=('OPS-SAT','SMAP','MSL')
MODES=('dataloader_standard','resident_standard','resident_fused')


def assemble(folder):
    files=[folder/f'{ds.lower().replace("-", "")}_baseline.json' for ds in DSETS]
    docs=[]
    for path in files:
        if not path.is_file():raise FileNotFoundError(f'Missing real GPU profile: {path}')
        d=json.loads(path.read_text())
        if d['device'].split(':')[0]!='cuda' or not d.get('gpu'):
            raise ValueError(f'{path}: CUDA/A100 benchmark required; CPU smoke results are not scientific GPU evidence')
        docs.append(d)
    if len({d['gpu'] for d in docs})!=1:
        raise ValueError('Mixed GPU models not comparable')
    if any('A100' not in d['gpu'] for d in docs):
        raise ValueError('TOAST A100 profiler figures require NVIDIA A100 benchmarks')
    if len({d.get('precision','faithful') for d in docs})!=1:
        raise ValueError('Mixed numerical policies not comparable')
    rows=[]
    for ds,d in zip(DSETS,docs):
        if d['dataset']!=ds:raise ValueError(f'Wrong dataset in {ds} profile')
        bymode={r.get('mode'):r for r in d['results']}
        if not all(mode in bymode and 'ms_per_step' in bymode[mode] for mode in MODES):
            raise ValueError(f'{ds}: missing measured baseline/resident/fused mode')
        reference=bymode[MODES[0]]['ms_per_step']
        for mode in MODES:
            r=bymode[mode]
            rows.append({'dataset':ds,'mode':mode,'gpu':d['gpu'],
                         'precision':d.get('precision','faithful'),
                         'batch_size':d['batch_size'], 'measured_steps':r['measured_steps'],
                         'milliseconds_per_step':r['ms_per_step'],
                         'windows_per_second':r['windows_per_second'],
                         'peak_allocated_gib':r['cuda_peak_allocated_gib'],
                         'speedup_vs_loader':reference/r['ms_per_step']})
    return docs,rows


def plot(rows,destination,gpu):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'figure.facecolor':'white','axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':210})
    labels=['DataLoader + Adam','VRAM cache + Adam','VRAM cache + fused Adam']
    colors=['#637993','#15978f','#cf8b38']
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.1),layout='constrained')
    positions=np.arange(len(DSETS));width=.26
    for k,mode in enumerate(MODES):
        r=[next(x for x in rows if x['dataset']==d and x['mode']==mode) for d in DSETS]
        xs=positions+(k-1)*width
        heights=np.array([x['speedup_vs_loader'] for x in r])
        axes[0].bar(xs,heights,width,color=colors[k],label=labels[k])
        for x,h in zip(xs,heights):axes[0].text(x,h+0.03,f'{h:.2f}×',ha='center',fontsize=9)
        axes[1].bar(xs,[x['milliseconds_per_step'] for x in r],width,color=colors[k])
    for ax in axes:
        ax.set_xticks(positions,DSETS)
        ax.grid(axis='y',alpha=.13)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x',labelsize=11)
    axes[0].axhline(1,color='#9ba8b3',linestyle='--',linewidth=1)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel('Speedup (same dataset and batch size)')
    axes[0].set_title('Throughput improvement  |  relative speed')
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel('Measured milliseconds per optimizer step')
    axes[1].set_title('Training-step time  |  lower is faster')
    fig.legend(loc='lower center',bbox_to_anchor=(.5,-.045),ncol=3,frameon=False)
    fig.suptitle('TOAST v12  |  Real-data A100 training microbenchmark',fontsize=16,fontweight='bold',color='#193752')
    fig.text(.5,-.03,f'{gpu}  •  baseline W/H  •  no accuracy inference  •  fixed batch size per dataset',
             ha='center',fontsize=9,color='#64748b')
    destination.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(destination/f'gpu_training_speedup.{ext}',bbox_inches='tight',pad_inches=.3)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT.parent/'results'/'v12'/'profiling')
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    docs,rows=assemble(args.root)
    output=args.out or args.root/'_figures'
    plot(rows,output,docs[0]['gpu'])
    with (output/'gpu_profile_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    print('[PROFILE] Real A100 plots written:',output)

if __name__=='__main__':main()
