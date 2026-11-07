#!/usr/bin/env python3
"""V12 structural and verified real-evaluation reports (irregular experiment grid)."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def report_all(*, results_root: Path, output: Path, structural_only: bool = False):
    import pandas as pd
    from campaign_v12 import BASES, SEEDS
    from tools.report_stage1_complexity import report
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from observability.measure_experiment import theoretical_matmul_macs
    structure=pd.DataFrame(report(ROOT/'experiment_plan_v12.json'))
    output.mkdir(parents=True,exist_ok=True)
    structure.to_csv(output/'v12_architecture.csv',index=False)
    representative=structure[structure.identifier.str.endswith(f'_s{SEEDS[0]}')]
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.6),layout='constrained')
    for ax,(dataset,baseline) in zip(axs,BASES.items()):
        part=representative[representative.dataset==dataset]
        prior=theoretical_matmul_macs(baseline['features'],baseline['window'],baseline['feedforward'])['macs_per_window_batch1']
        for ff,grp in part.groupby('feedforward'):
            grp=grp.sort_values('window')
            ax.plot(grp.window,100*grp.macs_per_window_batch1/prior,marker='o',label=f'FF={ff}')
        ax.set(title=dataset,xlabel='Window size (timesteps)',ylabel='MAC (% reference)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('TOAST v12: theoretical compute, not A100 time')
    fig.savefig(output/'v12_structural.png',dpi=180,bbox_inches='tight')
    fig.savefig(output/'v12_structural.pdf',bbox_inches='tight')
    plt.close(fig)
    if structural_only:
        return structure
    csvfile=results_root/'v12'/'study'/'all_experiments.csv'
    if not csvfile.is_file():
        raise FileNotFoundError(f'Cannot produce performance report; missing {csvfile}')
    data=pd.read_csv(csvfile)
    data=data[data.identifier.str.startswith('toast12_',na=False)].copy()
    expected=set(structure.identifier)
    if set(data.identifier)!=expected or len(data)!=len(expected) or data.identifier.duplicated().any():
        raise ValueError(f'Incomplete v12: {len(data)} rows, expected {len(expected)} unique validated IDs')
    if data.checkpoint_sha256.isna().any() or data.settings_digest.nunique()!=1:
        raise ValueError('Missing checkpoint digest or inconsistent evaluation settings')
    for key in ('average_precision_raw','roc_auc_raw','pot_f1_original','best_f1_original'):
        if pd.to_numeric(data[key],errors='coerce').isna().any():
            raise ValueError(f'Missing or nonnumeric {key}')
    merged=data.merge(structure[['identifier','window','feedforward','trainable_parameters','macs_per_window_batch1']],
                      on='identifier',validate='one_to_one')
    summary=merged.groupby(['dataset','window','feedforward'],as_index=False)[
        ['average_precision_raw','roc_auc_raw','pot_f1_original','best_f1_original']
    ].agg(['mean','std'])
    summary.columns=['_'.join(filter(None,map(str,col))) for col in summary.columns.to_flat_index()]
    summary.to_csv(output/'v12_quality_mean_std.csv',index=False)
    for metric in ('average_precision_raw','pot_f1_original','best_f1_original'):
        fig,axs=plt.subplots(1,3,figsize=(13.5,4.4),layout='constrained')
        for ax,dataset in zip(axs,BASES):
            part=merged[merged.dataset==dataset]
            grouped=part.groupby(['window','feedforward'])[metric].mean().reset_index()
            sc=ax.scatter(grouped.window,grouped.feedforward,c=grouped[metric],s=110,vmin=0,vmax=1,cmap='viridis')
            for entry in grouped.itertuples():
                ax.annotate(f'{getattr(entry,metric):.3f}',(entry.window,entry.feedforward),
                            xytext=(5,5),textcoords='offset points',fontsize=8)
            ax.set(title=dataset,xlabel='Window size',ylabel='FF dimension')
        fig.colorbar(sc,ax=axs.tolist(),label=metric,shrink=.8)
        fig.suptitle(metric+' (paper-style F1 is a test-label oracle if optimized)')
        fig.savefig(output/f'v12_{metric}.png',dpi=180,bbox_inches='tight')
        fig.savefig(output/f'v12_{metric}.pdf',bbox_inches='tight')
        plt.close(fig)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--structural-only',action='store_true')
    ap.add_argument('--results-root',type=Path,default=ROOT.parent/'results')
    ap.add_argument('--output',type=Path,default=ROOT.parent/'results'/'v12'/'figures')
    args=ap.parse_args()
    report_all(results_root=args.results_root.expanduser().resolve(),
               output=args.output.expanduser().resolve(),structural_only=args.structural_only)
    print('Reports written to',args.output)

if __name__=='__main__': main()
