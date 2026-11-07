#!/usr/bin/env python3
"""Read-only split audit for TOAST v12, with one report schema for all datasets.

SMAP/MSL official-train label inspection is a necessary check; a zero count
of label anomalies does not prove absence of unlabelled contamination.
OPS-SAT uses explicit fragment IDs; point labels are weak fragment labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def audit_native(dataset,data_root):
    import numpy as np
    base=Path(data_root)/'processed'/'tranad_plus'/dataset
    splits={}
    for split in ('Train','Test'):
        tr=base/split/'Tracks'; lb=base/split/'Labels'
        trfiles={f.stem:f for f in tr.glob('*.npy')}
        lbfiles={f.stem:f for f in lb.glob('*.npy')}
        if not trfiles or trfiles.keys()!=lbfiles.keys():
            raise ValueError(f'{dataset} {split}: missing/mismatched tracks and labels')
        points=positive=0
        shape_features=set()
        for key,track in sorted(trfiles.items()):
            x=np.load(track,mmap_mode='r',allow_pickle=False)
            y=np.load(lbfiles[key],mmap_mode='r',allow_pickle=False)
            if x.shape[0]!=y.shape[0]:raise ValueError(f'{dataset} {split} {key}: point count mismatch')
            if not np.isfinite(x).all():raise ValueError(f'{dataset} {split} {key}: non-finite data')
            if not np.isin(y,[0,1]).all():raise ValueError(f'{dataset} {split} {key}: nonbinary labels')
            points+=len(x);positive+=int(np.count_nonzero(y));shape_features.add(tuple(x.shape[1:]))
        splits[split.lower()]={'tracks':len(trfiles),'points':points,
                               'labelled_anomalous_points':positive,'feature_shapes':[list(x) for x in sorted(shape_features)]}
    if splits['train']['labelled_anomalous_points']:
        raise ValueError(f'{dataset}: labelled anomalies in official training data: {splits["train"]["labelled_anomalous_points"]}')
    return {'dataset':dataset,'training':'official nominal training tracks',
            'validation':{'exists':False,'reason':'paper-compatible v12 train/test protocol'},
            'test':'official test tracks (labels reserved for evaluation)',
            'splits':splits, 'disjointness':'train/test supplied as separate source files; IDs checked separately',
            'labels':'pointwise labels'}


def audit_opssat(data_root):
    import pandas as pd
    csv=Path(data_root)/'raw'/'OPSSAT'/'segments.csv'
    df=pd.read_csv(csv,usecols=['segment','channel','train','anomaly'])
    groups=df.groupby('segment',sort=False).agg(channel=('channel','first'),train=('train','first'),
                                                  anomaly=('anomaly','first'),points=('train','size'))
    if df.groupby('segment').train.nunique().max()!=1 or df.groupby('segment').anomaly.nunique().max()!=1 or df.groupby('segment').channel.nunique().max()!=1:
        raise ValueError('OPS-SAT mixed metadata within a segment')
    mask_tr=(groups.train==1)&(groups.anomaly==0)
    mask_excl=(groups.train==1)&(groups.anomaly==1)
    mask_te=groups.train==0
    if not mask_tr.any() or not mask_te.any():raise ValueError('OPS-SAT missing official split')
    def fields(mask):
        return {'fragments':int(mask.sum()), 'points':int(groups.loc[mask,'points'].sum()),
                'labelled_anomalous_fragments':int(groups.loc[mask,'anomaly'].sum())}
    return {'dataset':'OPS-SAT','training':'official train nominal fragments only',
            'validation':{'exists':True,'policy':'v10 per-channel 20% holdout of official train normal fragments, plus all official train anomalous fragments','split_seed':2026, 'validation_fraction':0.2,'normal_fragment_candidates':fields(mask_tr), 'anomalous_fragment_candidates':fields(mask_excl)},
            'test':'official test fragments (held out; weak fragment labels)',
            'splits':{'train':fields(mask_tr),'test':fields(mask_te),'excluded_from_training':fields(mask_excl)},
            'disjointness':'train/test split by unique fragment ID',
            'labels':'fragment-level weak labels, replicated across points for pointwise proxies',
            'note':'OPS-SAT and SMAP/MSL use same train/test roles but not identical event/point label semantics'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    from project_paths import DATA_ROOT, OUTPUT_ROOT
    data=(args.data_root or DATA_ROOT).expanduser().resolve()
    audits=[audit_opssat(data),audit_native('SMAP',data),audit_native('MSL',data)]
    document={'protocol':'TOAST v12 v10-aligned per dataset: OPS-SAT supervised validation 20pct; SMAP/MSL official train/test',
              'alpha_policy':'21 alpha selected on test labels only for legacy/oracle reproduction; not unbiased deployment performance',
              'datasets':audits}
    target=args.output or OUTPUT_ROOT/'v12'/'_audit'/'split_manifest.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(document,indent=2,ensure_ascii=False)+'\n')
    for record in audits:
        print('[SPLIT]',record['dataset'],record['splits'],flush=True)
    print('[SPLIT] Wrote',target,flush=True)
    print('[SPLIT] OPS-SAT uses v10 20% validation; SMAP/MSL keep official train/test. Test-label optimized Best F1/POT are oracle.',flush=True)

if __name__=='__main__':main()
