"""Fixed diagnostic protocol: maximum per fragment, BF200 on validation, no PA.

Does not train or search protocol choices on test. Retains original p95 studies.
Integer exports lacking saved maxima are evaluated on the explicit CPU integer
reference backend; these values are not labelled as original A100 measurements.
"""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,statistics as st,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tranad_plus'))
import numpy as np
from tqdm import tqdm
from tranad_plus.reports import recover_opssat_fragments as recovery

BASE=ROOT/'results/analysis_opssat_fragments'
OUT=BASE/'max_validation_bf200'
VARIANTS=['reference','shared_decoder','single_pass','encoder_only']
CONFIGS=['reference','window_only','ff_only','compact']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def writecsv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def averaged(group,**identity):
    r={**identity,'evaluations':len(group),'training_seeds':len({v['seed'] for v in group})}
    for k in ['precision','recall','f1','FPR','alarm_fraction','AUC_score','AP_score']:
        r[k+'_pct']=100*st.mean(v[k] for v in group)
        byseed=defaultdict(list)
        for v in group:byseed[v['seed']].append(v[k]*100)
        seedmeans=[st.mean(x) for x in byseed.values()]
        r[k+'_seed_SD_pp']=st.stdev(seedmeans) if len(seedmeans)>1 else None
    return r

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    replay=OUT/'integer_reference_replay';(replay/'cases').mkdir(parents=True,exist_ok=True)
    recovery.OUT=replay;recovery.DATA_HASH=sha(ROOT/'data/raw/OPSSAT/segments.csv')
    recovery.DEVICE='cpu'
    existing={}
    for path in (BASE/'cases').glob('*.json'):
        r=json.loads(path.read_text());p=r['profile'];ident=r['identity'];family=None
        if ident.startswith(('legacy:results/v10/','legacy:results/v12/')):family='ablation'
        elif ident.startswith('archive:results/v13/'):family='architecture'
        elif ident.startswith('quant:results/v16/'):family='weights_only'
        elif ident.startswith('quant:results/v17/'):family='integer_CPU_reference'
        if family:
            key=(family,p.get('variant','reference'),p.get('configuration',''),p['seed'],p.get('format','fp32'),p['window_sz'],p['dim_feedforward'])
            assert key not in existing,key
            existing[key]=(path,r)
    import torch
    torch.set_num_threads(1)
    pending=[]
    for variant in VARIANTS:
        for config in CONFIGS:
            for seed in range(2026,2031):
                for fmt in ['int8','int4']:
                    if any(k[:5]==('integer_CPU_reference',variant,config,seed,fmt) for k in existing):continue
                    folder=ROOT/f'results/v17/integer/OPS-SAT/{variant}/{config}/seed_{seed}'
                    pending.append((folder,fmt))
    for folder,fmt in tqdm(pending,desc='MAX protocol | recover integer maxima | NO TRAINING',unit='model'):
        recovery.replay_quant(folder,fmt,'integer')
    for path in (replay/'cases').glob('*.json'):
        r=json.loads(path.read_text());p=r['profile']
        key=('integer_CPU_reference',p['variant'],p['configuration'],p['seed'],p['format'],p['window_sz'],p['dim_feedforward'])
        assert key not in existing,key
        existing[key]=(path,r)
    rows=[];sources=[]
    for key,(path,r) in sorted(existing.items()):
        family,variant,config,seed,fmt,w,ff=key
        assert 'max' in r['metrics'] and 'validation_BF200' in r['metrics']['max'],r['identity']
        arrays=path.with_suffix('.npz');assert sha(arrays)==r['fragment_arrays_sha256']
        with np.load(arrays,allow_pickle=False) as z:
            vi,ti=z['validation_ids'],z['test_ids'];vy,ty=z['validation_labels'],z['test_labels'];vs,ts=z['validation_max'],z['test_max']
            assert not np.intersect1d(vi,ti).size
            assert len(ty)==529 and int(ty.sum())==113
            # Threshold selector takes ONLY validation arrays. Test labels enter counts below.
            threshold,index=recovery.bf200(vs,vy)
            c=recovery.counts(ty,ts>threshold);old=r['metrics']['max']['validation_BF200']
            assert threshold==old['threshold']
            for k in ['TP','FP','TN','FN','precision','recall','f1']:assert abs(c[k]-old[k])<1e-12
        rows.append({'family':family,'architecture':variant,'configuration':config,'window':w,'ff':ff,
            'seed':seed,'format':fmt,'alpha':r['recovery'].get('alpha'),'epochs':r['profile']['epochs'],
            'threshold':threshold,'threshold_index':index,'threshold_steps':200,
            **{k:c[k] for k in ['precision','recall','f1','TP','FP','TN','FN']},
            'FPR':c['FP']/(c['FP']+c['TN']),'alarm_fraction':(c['TP']+c['FP'])/529,
            'AUC_score':r['metrics']['max']['AUC_score'],'AP_score':r['metrics']['max']['AP_score'],
            'backend':r['recovery']['mode'],'source':str(path.relative_to(ROOT))})
        sources.append({'source':str(path.relative_to(ROOT)),'json_sha256':sha(path),'arrays_sha256':sha(arrays)})
    counts=Counter(r['family'] for r in rows)
    assert dict(counts)=={'ablation':90,'architecture':80,'integer_CPU_reference':160,'weights_only':160},counts
    writecsv('per_model.csv',rows)
    groups=defaultdict(list)
    for r in rows:
        if r['family']=='ablation':groups[r['window'],r['ff']].append(r)
    ablation=[averaged(v,window=w,ff=ff) for (w,ff),v in sorted(groups.items())]
    assert len(ablation)==30 and all(x['training_seeds']==3 for x in ablation)
    writecsv('ablation.csv',ablation)
    architecture=[]
    for variant in VARIANTS:
        for config in CONFIGS:
            g=[r for r in rows if r['family']=='architecture' and r['architecture']==variant and r['configuration']==config]
            assert len(g)==5
            architecture.append(averaged(g,architecture=variant,configuration=config))
    writecsv('architecture.csv',architecture)
    quant=[]
    for family in ['weights_only','integer_CPU_reference']:
        for fmt in ['fp32','int8','int4']:
            g=[r for r in rows if r['family']==('architecture' if fmt=='fp32' else family) and r['format']==fmt]
            assert len(g)==80
            quant.append(averaged(g,family=family,format=fmt))
    writecsv('quantization.csv',quant)
    def table(headers,values):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in values])
    cols=['precision','recall','f1','FPR','alarm_fraction','AUC_score']
    def cells(r):return [f"{r[k+'_pct']:.2f}" for k in cols]
    text=['# OPS-SAT: maximum + validation BF200, fixed protocol','',
      'No training. Fixed alpha .5 for two-output models, none for single output. One maximum score and one vote per fragment. BF200 uses 200 linear thresholds from min(validation score) to mean(validation score)+3std(validation score). Choose highest validation F1, freeze threshold on test. No PA, no threshold/alpha selection on test, no replacement with nominal-quantile thresholds.','',
      'This is a retrospective diagnostic protocol fixed in the conversation before running these comparisons. The test set had already been inspected in earlier studies; these are not blind, independently preregistered confirmatory results.','',
      'All integer quantized rows use the explicit CPU reference implementation, not measurements claimed from the A100. Source exports and numerical checks are preserved. Existing historical p95 and A100 results remain unchanged. FP32 references use original archived scores; floating-point platform drift remains a limitation.','',
      '## Ablation, three seeds','',table(['W','FF','Precision','Recall','F1','FPR','Alarm %','AUC-score'],[[r['window'],r['ff'],*cells(r)] for r in ablation]),'',
      '## Architectures at W10 / FF8, five seeds','',table(['Architecture','Precision','Recall','F1','FPR','Alarm %','AUC-score'],[[r['architecture'],*cells(r)] for r in architecture if r['configuration']=='reference']),'',
      '## Quantization, 4 architectures x 4 settings x 5 seeds','',table(['Family','Format','Precision','Recall','F1','FPR','Alarm %','AUC-score'],[[r['family'],r['format'],*cells(r)] for r in quant]),'',
      'All-positive baseline: P=21.36%, R=100%, F1=35.20%, FPR=100%, alarm fraction=100%, constant-score AUC=.5. Validation has 321 anomalous and 254 nominal fragments; test has 113 anomalous and 416 nominal. A threshold maximizing validation F1 can favor near-all-positive decisions; score discrimination must also be assessed. These results do not justify selecting another protocol because its test numbers look better.','']
    (OUT/'REPORT.md').write_text('\n'.join(text))
    recovery.write(OUT/'manifest.json',{'protocol':'maximum_validation_BF200','alpha_two_outputs':.5,'threshold_steps':200,
        'point_adjustment':False,'selection_uses_test_labels':False,'test_previously_seen_in_other_analyses':True,
        'coverage':dict(counts),'training_added':0,'sources':sources,'script_sha256':sha(Path(__file__)),
        'integer_scope':'CPU reference integer evaluation; not original A100 maxima',
        'baseline_source_sha256':recovery.DATA_HASH})
    print('COMPLETE',dict(counts),flush=True)

if __name__=='__main__':main()
