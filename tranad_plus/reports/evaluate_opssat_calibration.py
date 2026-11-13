"""Exploratory fragment calibration controls, reusing all saved maxima.

Primary: maximum validation balanced accuracy (Youden J), exact validation ROC
operating points. Ties select the largest threshold / fewest false alarms.
Secondary: 95th percentile of nominal validation fragment scores (5% target).
No training, inference, PA, alpha search, or test-label threshold selection.
"""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,sys,statistics as st
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tranad_plus'))
import numpy as np
from sklearn.metrics import roc_curve
from tranad_plus.reports.recover_opssat_fragments import counts

SOURCE=ROOT/'results/analysis_opssat_fragments/max_validation_bf200/per_model.csv'
OUT=SOURCE.parent.parent/'calibration_controls'
VARIANTS=['reference','shared_decoder','single_pass','encoder_only']
CONFIGS=['reference','window_only','ff_only','compact']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def youden(validation_scores,validation_labels):
    s=np.asarray(validation_scores);y=np.asarray(validation_labels).astype(bool)
    candidates=np.r_[np.nextafter(s.min(),-np.inf),np.unique(s)]
    positive,negative=np.sort(s[y]),np.sort(s[~y])
    if not len(positive) or not len(negative):raise ValueError('Both validation classes required')
    tp=len(positive)-np.searchsorted(positive,candidates,side='right')
    fp=len(negative)-np.searchsorted(negative,candidates,side='right')
    j=tp/len(positive)-fp/len(negative)
    # A fixed, conservative tie policy; never inspect test labels here.
    best=int(np.flatnonzero(np.isclose(j,j.max(),rtol=0,atol=1e-14))[-1])
    threshold=float(candidates[best])
    fpr,tpr,_=roc_curve(y,s,drop_intermediate=False)
    assert abs(float(j[best])-float((tpr-fpr).max()))<1e-12
    return threshold,float(j[best]),len(candidates)

def writecsv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def average(group,**identity):
    r={**identity,'evaluations':len(group),'training_seeds':len({x['seed'] for x in group})}
    for k in ['precision','recall','f1','balanced_accuracy','FPR','alarm_fraction','AUC_score']:
        r[k+'_pct']=100*st.mean(x[k] for x in group)
        byseed=defaultdict(list)
        for x in group:byseed[x['seed']].append(100*x[k])
        a=[st.mean(v) for v in byseed.values()]
        r[k+'_seed_SD_pp']=st.stdev(a) if len(a)>1 else None
    return r

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with SOURCE.open() as f:models=list(csv.DictReader(f))
    assert len(models)==490
    rows=[];provenance=[];checks=[]
    for model in models:
        source=ROOT/model['source'];r=json.loads(source.read_text());path=source.with_suffix('.npz')
        assert sha(path)==r['fragment_arrays_sha256']
        with np.load(path,allow_pickle=False) as z:
            vs,vy,vi=z['validation_max'],z['validation_labels'],z['validation_ids']
            ts,ty,ti=z['test_max'],z['test_labels'],z['test_ids']
            assert not np.intersect1d(vi,ti).size
            assert len(ty)==529 and int(ty.sum())==113
            best,j,n=youden(vs,vy)
            nominal_threshold=float(np.quantile(vs[vy==0],.95))
            for method,threshold in [('validation_Youden_J',best),('nominal_validation_FPR05',nominal_threshold)]:
                val=counts(vy,vs>threshold);test=counts(ty,ts>threshold)
                row={k:model[k] for k in ['family','architecture','configuration','format','backend','source']}
                row.update({k:int(model[k]) for k in ['window','ff','seed','epochs']})
                row.update({'alpha':None if model['alpha']=='' else float(model['alpha']),
                    'calibration':method,'threshold':threshold,'selection_uses_test_labels':False,
                    **{k:test[k] for k in ['precision','recall','f1','TP','FP','TN','FN']},
                    'balanced_accuracy':test['AUC_bin'],'FPR':test['FP']/(test['FP']+test['TN']),
                    'alarm_fraction':(test['TP']+test['FP'])/529,'AUC_score':float(model['AUC_score']),
                    'validation_balanced_accuracy':val['AUC_bin'],'validation_FPR':val['FP']/(val['FP']+val['TN']),
                    'validation_recall':val['recall'],'validation_candidates':n if method=='validation_Youden_J' else 1})
                rows.append(row)
            checks.append({'source':model['source'],'validation_ROC_optimum_verified':True,'best_validation_J':j})
        provenance.append({'source':model['source'],'source_sha256':sha(source),'arrays_sha256':sha(path)})
    writecsv('per_model.csv',rows)
    arch=[];ablation=[];quant=[]
    for method in ['validation_Youden_J','nominal_validation_FPR05']:
        part=[r for r in rows if r['calibration']==method]
        g=defaultdict(list)
        for r in part:
            if r['family']=='ablation':g[r['window'],r['ff']].append(r)
        assert len(g)==30
        ablation.extend(average(v,calibration=method,window=w,ff=ff) for (w,ff),v in sorted(g.items()))
        for v in VARIANTS:
            for c in CONFIGS:
                a=[r for r in part if r['family']=='architecture' and r['architecture']==v and r['configuration']==c]
                assert len(a)==5
                arch.append(average(a,calibration=method,architecture=v,configuration=c))
        for family in ['weights_only','integer_CPU_reference']:
            for fmt in ['fp32','int8','int4']:
                a=[r for r in part if r['family']==('architecture' if fmt=='fp32' else family) and r['format']==fmt]
                assert len(a)==80
                quant.append(average(a,calibration=method,family=family,format=fmt))
    writecsv('architecture.csv',arch);writecsv('ablation.csv',ablation);writecsv('quantization.csv',quant)
    manifest={'status':'complete','models':490,'evaluation_rows':len(rows),'training_added':0,'inference_added':0,
        'primary':'validation_Youden_J','secondary':'nominal_validation_FPR05','aggregation':'maximum_per_fragment',
        'alpha':'fixed .5 for two-output models; none for single output','PA':False,
        'primary_objective':'maximize (validation_TPR + validation_TNR)/2; exact validation ROC operating points',
        'tie_policy':'largest threshold among equal validation optima','test_used_for_threshold_selection':False,
        'retrospective_exploratory_analysis':True,'all_preceding_protocols_preserved':True,
        'secondary_target':'5% false positive fragment rate on nominal validation; not guaranteed on test, not a mission requirement',
        'primary_reference':'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html',
        'source_table_sha256':sha(SOURCE),'script_sha256':sha(Path(__file__)),
        'families':dict(Counter(r['family'] for r in rows if r['calibration']=='validation_Youden_J')),
        'provenance':provenance,'independent_ROC_checks':checks,
        'integer_platform':'explicit CPU integer reference; not original A100 measurements'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETE',len(models),'models,',len(rows),'calibration controls, no training/inference')

if __name__=='__main__':main()
