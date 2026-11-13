"""Pointwise threshold/alpha selection. PA is used only in separate reports."""
import numpy as np


def point_metrics(labels, predictions):
    y=np.asarray(labels).reshape(-1);p=np.asarray(predictions).reshape(-1)
    if y.shape!=p.shape or not np.isin(y,[0,1]).all() or not np.isin(p,[0,1]).all():
        raise ValueError('Aligned binary labels and predictions required')
    y,p=y.astype(bool),p.astype(bool)
    return from_counts(int((y&p).sum()),int((~y&p).sum()),int((~y&~p).sum()),int((y&~p).sum()))


def from_counts(tp,fp,tn,fn):
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    return {'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'precision':precision,'recall':recall,
            'auc_binary':.5*(recall+tn/(tn+fp)) if tp+fn and tn+fp else None,
            'TP':int(tp),'FP':int(fp),'TN':int(tn),'FN':int(fn)}


def bf200(scores,labels):
    scores=np.asarray(scores).reshape(-1);y=np.asarray(labels).reshape(-1).astype(bool)
    if len(scores)!=len(y) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite nonempty aligned scores required')
    thresholds=np.linspace(float(scores.min()),float(scores.mean()+3*scores.std()),200)
    positive=np.sort(scores[y]);negative=np.sort(scores[~y])
    tp=len(positive)-np.searchsorted(positive,thresholds,side='right')
    fp=len(negative)-np.searchsorted(negative,thresholds,side='right')
    denominator=len(positive)+tp+fp
    values=np.divide(2*tp,denominator,out=np.zeros(200),where=denominator>0)
    index=int(np.argmax(values));threshold=float(thresholds[index])
    # Final counts use exactly the same strict comparison as runtime inference.
    return {**point_metrics(y,scores>threshold),'threshold':threshold,'threshold_index':index,
            'threshold_steps':200,'point_adjustment':False}


def select(grid,labels,alpha_records,advance=None):
    """BF selects alpha+threshold; POT selects alpha among pre-fitted thresholds."""
    y=np.asarray(labels).reshape(-1)
    if not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ValueError('Both binary classes required')
    if len(grid)!=len(alpha_records):raise ValueError('Score/alpha rows differ')
    if not np.isfinite(grid).all():raise FloatingPointError('Nonfinite scores; no masking')
    candidates=[];winners={'bf':None,'pot':None}
    for i,record in enumerate(alpha_records):
        alpha=record['alpha'];scores=grid[i]
        bf={**bf200(scores,y),'alpha':alpha}
        threshold=record.get('pot_threshold')
        pot=None
        if threshold is not None:
            if not np.isfinite(threshold):raise ValueError('Nonfinite stored POT threshold')
            pot={**point_metrics(y,scores>threshold),'alpha':alpha,'threshold':float(threshold),
                 'point_adjustment':False,'threshold_source':'v17_SPOT_for_this_alpha'}
        for key,value in (('bf',bf),('pot',pot)):
            if value is not None and (winners[key] is None or value['f1']>winners[key]['f1']):winners[key]=value
        candidates.append({'alpha':alpha,'bf':bf,'pot':pot})
        if advance:advance(i,alpha,bf['f1'],None if pot is None else pot['f1'])
    if winners['pot'] is None:raise RuntimeError('No valid stored POT threshold')
    return winners,candidates


def comparison(grid,labels,records,reference,groups=None,advance=None):
    selected,candidates=select(grid,labels,records,advance)
    # This import and every PA call occur AFTER raw threshold selection.
    from tranad_plus.core.evaluation.metrics import fixed_metrics
    results={'no_pa_optimized':{},'no_pa_at_pa_selection':{},'pa_at_no_pa_selection':{},'pa_at_pa_selection':{}}
    parity={};alphas=[r['alpha'] for r in records]
    for method in ('bf','pot'):
        winner=selected[method];original=reference['oracle'][method]
        original_alpha=original['alpha'];threshold=float(original[method+'_threshold'])
        original_scores=grid[alphas.index(original_alpha)]
        direct=point_metrics(labels,original_scores>threshold)
        direct.update(alpha=original_alpha,threshold=threshold,point_adjustment=False)
        results['no_pa_optimized'][method]=winner
        results['no_pa_at_pa_selection'][method]=direct
        for protocol,endpoint in (('pa_at_no_pa_selection',winner),('pa_at_pa_selection',direct)):
            scores=grid[alphas.index(endpoint['alpha'])]
            adjusted=fixed_metrics(scores,labels,endpoint['threshold'],method,groups)
            result={k:adjusted[method+'_'+suffix] for k,suffix in [('f1','f1'),('precision','precision'),('recall','recall'),('auc_binary','ROC/AUC'),('TP','TP'),('TN','TN'),('FP','FP'),('FN','FN')]}
            result.update(alpha=endpoint['alpha'],threshold=endpoint['threshold'],point_adjustment=True)
            results[protocol][method]=result
        old_raw=original.get('raw_without_PA',{})
        parity[method]={'raw_confusion_counts_match_v17':all(direct[k]==old_raw.get(method+'_'+k) for k in ('TP','TN','FP','FN')),
                        'PA_confusion_counts_match_v17':all(results['pa_at_pa_selection'][method][k]==original[method+'_'+k] for k in ('TP','TN','FP','FN')),
                        'raw_f1_difference_from_v17':direct['f1']-old_raw.get(method+'_f1',direct['f1'])}
    return {'protocols':results,'alpha_search':candidates,'v17_endpoint_parity':parity,
            'all_positive_baseline':point_metrics(labels,np.ones(len(labels),dtype=np.int8)),
            'auc_names':{'without_PA':'AUC-bin','with_PA':'AUC-PA'},
            'auc_definition':'ROC-AUC of binary predictions = (TPR+TNR)/2; not ROC-AUC of continuous scores',
            'selection':'Test-label oracle on original 21-alpha/200-threshold grids, pointwise F1 objective',
            'pot_definition':'Original V17 SPOT thresholds per alpha; only alpha is reselected without PA',
            'label_granularity':'weak_fragment_labels' if groups else 'point_labels'}
