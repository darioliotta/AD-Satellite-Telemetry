"""OPS-SAT anomaly scoring and leakage-aware, segment-level model selection.

No torch dependency. Never fit a calibration on test labels. Normal training
scores are the only data used to estimate calibrator parameters and thresholds.
The validation labels are used only for selecting pre-declared candidates.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CALIBRATIONS = ('identity', 'robust_channel')
AGGREGATIONS = ('max', 'p95')


def _array(values, dtype=None):
    a = np.asarray(values, dtype=dtype)
    if a.ndim != 1 or (a.dtype.kind in 'fi' and not np.isfinite(a).all()):
        raise ValueError('Expected finite 1D array')
    return a


def mix_scores(decoder1, decoder2, target, alpha):
    if not 0 <= float(alpha) <= 1:
        raise ValueError('alpha must lie in [0,1]')
    d1, d2, y = (np.asarray(a, dtype=np.float64) for a in (decoder1, decoder2, target))
    if d1.ndim != 1 or d1.shape != d2.shape or d1.shape != y.shape:
        raise ValueError('Decoder/target dimensions do not match')
    return np.square(float(alpha) * d1 + (1-float(alpha)) * d2 - y)


def fit_calibrator(train_scores, train_channels, method='identity'):
    if method not in CALIBRATIONS:
        raise ValueError(f'Unknown calibration {method}')
    s, channels = _array(train_scores, float), _array(train_channels)
    if len(s) != len(channels) or not len(s):
        raise ValueError('Mismatched or empty training scores')
    params = {}
    for ch in sorted(set(map(str,channels))):
        subset=s[channels == ch]
        if not len(subset):raise ValueError(f'No training points for {ch}')
        if method == 'identity':
            params[ch]={'center':0.0, 'scale':1.0,'n_fit':int(len(subset))}
        else:
            q25, q75=np.quantile(subset,[0.25,0.75]); median=float(np.median(subset))
            iqr=float(q75-q25)
            # Absolute epsilon protects near-constant normal scores; recorded for audit.
            scale=max(iqr, 1e-12)
            params[ch]={'center':median,'scale':scale,'n_fit':int(len(subset)),
                        'iqr':iqr,'epsilon_floor_applied':bool(iqr<1e-12)}
    return {'method':method,'channels':params,'fit_source':'normal_training_scores_only'}


def transform_scores(scores, channels, calibrator):
    s, ch=_array(scores,float),_array(channels)
    if len(s)!=len(ch):raise ValueError('Score/channel length mismatch')
    output=np.empty_like(s)
    for channel, setting in calibrator['channels'].items():
        mask=ch == channel
        output[mask]=(s[mask]-setting['center'])/setting['scale']
    if not set(map(str,ch)).issubset(calibrator['channels']):
        raise ValueError('Unseen test/validation channel; no train calibration available')
    return output


def point_channels(groups):
    return np.concatenate([np.repeat(g['channel'],len(g['values'])) for g in groups])


def segment_statistics(scores, groups, kind):
    if kind not in AGGREGATIONS:raise ValueError('Unsupported fragment aggregation')
    s=_array(scores,float)
    offset=0;values=[];labels=[];ids=[];channels=[]
    for g in groups:
        n=len(g['values'])
        if n<1:raise ValueError('Empty fragment')
        part=s[offset:offset+n]
        if len(part)!=n:raise ValueError('Unaligned fragment score lengths')
        values.append(float(np.max(part) if kind=='max' else np.quantile(part,0.95)))
        labels.append(int(g['label']));ids.append(int(g['id']));channels.append(g['channel'])
        offset+=n
    if offset!=len(s):raise ValueError('Extra scores not accounted for by fragments')
    return np.array(values),np.array(labels,dtype=np.int8),np.array(ids),np.array(channels)


def auc_ap(labels,scores):
    y,s=_array(labels,int),_array(scores,float)
    if len(y)!=len(s):raise ValueError('Metric length mismatch')
    if len(np.unique(y))<2:
        return {'roc_auc':None,'average_precision':None,'reason':'single_class'}
    return {'roc_auc':float(roc_auc_score(y,s)),
            'average_precision':float(average_precision_score(y,s))}


def selected_scoring(train_outputs, validation_outputs,train_groups,val_groups,
                     *, alphas=ALPHAS,calibrations=CALIBRATIONS,aggregations=AGGREGATIONS):
    """Choose hyperparameters using *only* validation fragment-level labels.

    Objective: fragment ROC-AUC; ties resolved by stable candidate order.
    Return ranked candidates and chosen configuration, never test statistics.
    """
    if not val_groups or not any(int(g['label']) for g in val_groups) or not any(not int(g['label']) for g in val_groups):
        raise ValueError('Validation must contain normal and anomalous fragments')
    train_channels=point_channels(train_groups)
    val_channels=point_channels(val_groups)
    candidates=[];best=None
    for alpha in alphas:
        tr=mix_scores(*train_outputs,alpha)
        va=mix_scores(*validation_outputs,alpha)
        for method in calibrations:
            cal=fit_calibrator(tr,train_channels,method)
            normalized=transform_scores(va,val_channels,cal)
            for agg in aggregations:
                fs,fl,_,_=segment_statistics(normalized,val_groups,agg)
                quality=auc_ap(fl,fs)
                rec={'alpha':float(alpha),'calibration':method,'aggregation':agg,
                     'validation_fragment_auc':quality['roc_auc'],
                     'validation_fragment_ap':quality['average_precision'],
                     'validation_n_fragments':len(fl),'validation_n_anomalous_fragments':int(fl.sum())}
                candidates.append(rec)
                if quality['roc_auc'] is not None and (best is None or quality['roc_auc'] > best['validation_fragment_auc']):
                    best=rec
    if best is None:raise ValueError('No valid validation ranking')
    return {'chosen':best.copy(),'candidates':candidates,
            'selection_target':'fragment_roc_auc_validation_only',
            'warning':'Validation selection is supervised; do not optimize selection on test labels'}


def evaluate_selected(train_outputs,test_outputs,train_groups,test_groups,selection,train_quantile=0.995):
    """Apply already-selected rules, no test-label-based choices."""
    choice=selection['chosen']
    tr=mix_scores(*train_outputs,choice['alpha'])
    te=mix_scores(*test_outputs,choice['alpha'])
    trch=point_channels(train_groups);tech=point_channels(test_groups)
    cal=fit_calibrator(tr,trch,choice['calibration'])
    calibrated_train=transform_scores(tr,trch,cal)
    calibrated_test=transform_scores(te,tech,cal)
    fragments,yfrag,ids,channels=segment_statistics(calibrated_test,test_groups,choice['aggregation'])
    point_labels=np.concatenate([np.repeat(int(g['label']),len(g['values'])) for g in test_groups])
    # Channel-conditional train-only thresholds avoid one channel dominating alarms.
    thresholds={c:float(np.quantile(calibrated_train[trch==c],train_quantile)) for c in cal['channels']}
    alerts=np.array([calibrated_test[i]>thresholds[ch] for i,ch in enumerate(tech)],dtype=np.int8)
    pf1=precision_recall_fscore_support(point_labels,alerts,average='binary',zero_division=0)
    pmetrics=auc_ap(point_labels,calibrated_test)
    fmetrics=auc_ap(yfrag,fragments)
    by_channel={}
    for ch in sorted(cal['channels']):
        p=tech==ch;f=channels==ch
        by_channel[ch]={'points':int(p.sum()),'fragments':int(f.sum()),
                        'pointwise':auc_ap(point_labels[p],calibrated_test[p]),
                        'fragment':auc_ap(yfrag[f],fragments[f])}
    return {'selection':selection,'calibrator':cal,'thresholds_train_only_by_channel':thresholds,
            'pointwise':{**pmetrics,'precision':float(pf1[0]),'recall':float(pf1[1]),'f1':float(pf1[2]),
                         'n_points':int(len(point_labels)),'n_anomalous_points':int(point_labels.sum()),
                         'point_adjusted':False},
            'fragment':{**fmetrics,'aggregation':choice['aggregation'],
                        'n_fragments':int(len(yfrag)),'n_anomalous_fragments':int(yfrag.sum())},
            'by_channel':by_channel,
            'protocol':'Supervised configuration selection on held-out training fragments only; test unseen during selection',
            'weak_labels':'Fragment-level labels repeated on points; fragment-level AUC is primary'}
