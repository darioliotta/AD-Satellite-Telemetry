#!/usr/bin/env python3
"""Validate and summarize the COMPLETE 6x5 v10+v12 architecture grid.

Never equates oracle/test-optimized metrics to unbiased held-out deployment
performance. This script reads *completed* campaign CSVs only; no training.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from campaign_v12 import BASES, HISTORICAL, GRID, SEEDS, new_pairs

METRICS=('pot_f1_original','best_f1_original','best_f1_dense_oracle',
         'average_precision_raw','roc_auc_raw','latency_p50_ms')
V10_LEVELS=(25,50,75,100)


def attach_levels(df,version):
    import pandas as pd
    records=[]
    for r in df.itertuples():
        dataset=str(r.dataset); name=str(r.identifier)
        if dataset not in BASES: raise ValueError(f'Unknown dataset: {dataset}')
        prefix=dataset.lower().replace('-','')
        if version=='v10':
            match=re.fullmatch(r'toast10_'+prefix+r'_w(025|050|075|100)_h(025|050|075|100)_s(\d+)',name)
            if not match: raise ValueError(f'Invalid v10 run ID: {name}')
            w=HISTORICAL[dataset]['windows'][V10_LEVELS.index(int(match.group(1)))]
            ff=HISTORICAL[dataset]['feedforwards'][V10_LEVELS.index(int(match.group(2)))]
        else:
            match=re.fullmatch(r'toast12_'+prefix+r'_w(\d+)_h(\d+)_s(\d+)',name)
            if not match:raise ValueError(f'Invalid v12 run ID: {name}')
            w=int(match.group(1));ff=int(match.group(2))
            if (w,ff) not in new_pairs(dataset): raise ValueError(f'Not a new v12 cell: {name}')
        seed=int(match.group(3))
        if seed not in SEEDS:raise ValueError(f'Unexpected seed in {name}')
        records.append((w,ff,seed))
    out=df.copy()
    out['window']=[a for a,b,c in records]
    out['feedforward']=[b for a,b,c in records]
    out['seed_parsed']=[c for a,b,c in records]
    out['campaign']=version
    return out


def combine(v10_csv:Path,v12_csv:Path,output:Path):
    import pandas as pd
    v10=attach_levels(pd.read_csv(v10_csv),'v10')
    v12=attach_levels(pd.read_csv(v12_csv),'v12')
    if len(v10)!=144 or len(v12)!=126:
        raise ValueError(f'Expected v10=144 and v12=126 rows; got {len(v10)} and {len(v12)}')
    for frame in (v10,v12):
        if frame.identifier.duplicated().any():raise ValueError('Duplicated run identifiers')
        if 'seed' in frame.columns and any(frame.seed.astype(int)!=frame.seed_parsed):
            raise ValueError('Seed mismatch between identifier and CSV')
        for metric in METRICS:
            if metric not in frame:raise ValueError('Missing '+metric)
            if pd.to_numeric(frame[metric],errors='coerce').isna().any():
                raise ValueError(f'Nonfinite/undefined metric in {metric}; investigate before merging')
    merged=pd.concat([v10,v12],ignore_index=True)
    for dataset in BASES:
        subset=merged[merged.dataset==dataset]
        expected={(dataset,w,ff,seed) for w in GRID[dataset]['windows']
                  for ff in GRID[dataset]['feedforwards'] for seed in SEEDS}
        observed={(r.dataset,int(r.window),int(r.feedforward),int(r.seed_parsed))
                  for r in subset.itertuples()}
        if len(subset)!=90 or observed!=expected:
            raise ValueError(f'{dataset}: missing/duplicated grid cells or seeds')
    output.mkdir(parents=True,exist_ok=True)
    cols=['campaign','dataset','identifier','seed_parsed','window','feedforward',*METRICS]
    merged[cols].sort_values(['dataset','window','feedforward','seed_parsed']).to_csv(
        output/'combined_v10_v12_270_runs.csv',index=False)
    summary=merged.groupby(['dataset','window','feedforward'],as_index=False)[list(METRICS)].agg(['mean','std'])
    summary.columns=['_'.join(filter(None,map(str,c))) for c in summary.columns.to_flat_index()]
    if len(summary)!=90:raise ValueError('Expected 90 unique architecture cells')
    summary.to_csv(output/'combined_grid_90_architectures_mean_std.csv',index=False)
    print(f'[GRID] {len(merged)} runs = 144 v10 + 126 v12; {len(summary)} cells (3 x 6 x 5); 3 seeds/cell')
    print(f'[GRID] Wrote {output}')
    print('[CAUTION] Review v10 versus v12 scoring implementation parity before interpreting numeric differences.')
    return merged, summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,default=ROOT.parent/'results')
    p.add_argument('--output',type=Path,default=ROOT.parent/'results'/'v12'/'combined_grid')
    a=p.parse_args()
    root=a.results_root.expanduser().resolve()
    combine(root/'v10'/'study'/'all_experiments.csv',root/'v12'/'study'/'all_experiments.csv',
            a.output.expanduser().resolve())

if __name__=='__main__':main()
