#!/usr/bin/env python3
"""Write architecture parameter count for v12 incremental full-factorial grid completion without running training or datasets.

This is a structural count, NOT measured latency/MACs/performance.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from TranADPlus.src.models import TranAD
from observability.measure_experiment import theoretical_matmul_macs


def report(plan):
    entries=json.loads(Path(plan).read_text(encoding='utf-8'))['experiments']
    features={'opssat_shared_ws10_1f':1,'smap_plus_ws10':25,'msl_plus_ws50_dff2x':55}
    rows=[]
    for e in entries:
        if e['base_profile'] not in features:raise ValueError('Unsupported stage-one base: '+e['base_profile'])
        n_feats=features[e['base_profile']]
        w,h=e['window_sz'],e['dim_feedforward']
        model=TranAD(n_feats=n_feats,dim_feedforward=h,batch_sz=1,window_sz=w,
                     num_encoder_layers=1,num_decoder_layers=1)
        params=sum(p.numel() for p in model.parameters())
        expected=82*n_feats**2+12*n_feats*h+47*n_feats+3*h
        if params!=expected:raise AssertionError(f'{e["identifier"]}: {params} != {expected}')
        rows.append({'dataset':{'opssat_shared_ws10_1f':'OPS-SAT','smap_plus_ws10':'SMAP',
                        'msl_plus_ws50_dff2x':'MSL'}[e['base_profile']],
                    'identifier':e['identifier'],'factor':e['ablation_factor'],
                    'level':e['ablation_level'],'features':n_feats,'window':w,
                    'feedforward':h,'trainable_parameters':params,'fp32_weight_bytes':params*4,
                    'macs_per_window_batch1': theoretical_matmul_macs(n_feats,w,h)['macs_per_window_batch1'],
                    'matmul_flops_per_window_batch1': theoretical_matmul_macs(n_feats,w,h)['matmul_flops_per_window_batch1'],
                    'reference':e['ablation_reference']})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'experiment_plan_v12.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rows=report(args.plan)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Wrote {len(rows)} structural parameter profiles: {args.output}')

if __name__=='__main__':main()
