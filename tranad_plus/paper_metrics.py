"""Fast, testable equivalents of upstream threshold evaluation.

The *paper* metric keeps 200 LINEAR thresholds and point adjustment, including
upstream's special first-sample behavior. The supplementary *dense* diagnostic
searches all distinct observed score thresholds and is explicitly test-oracle.
Neither metric is a deployment-time estimate of generalization performance.
"""
from __future__ import annotations
import numpy as np
from TranADPlus.src import omnianomaly


def _segments(labels, boundaries=None):
    """Return contiguous positive runs, treating optional boundary indices as separators."""
    y = np.asarray(labels).astype(bool).ravel()
    start = y & ~np.r_[False, y[:-1]]
    end = y & ~np.r_[y[1:], False]
    if boundaries is not None:
        for i in boundaries:
            if 0 < int(i) < len(y):
                start[int(i)] = bool(y[int(i)])
                end[int(i)-1] = bool(y[int(i)-1])
    starts = np.flatnonzero(start)
    ends = np.flatnonzero(end) + 1
    if len(starts) != len(ends):
        raise AssertionError('Invalid anomaly segmentation')
    return starts, ends


def _stats(score, label, thresholds, boundaries=None, preserve_upstream_index_zero=True):
    """Fast F1 for adjusted predictions. Native protocol merges adjacent positive runs.

    With explicit boundaries, the first sample of every fragment is treated as
    if the fragment had a preceding nominal sentinel (OPS-SAT v9 policy).
    """
    score = np.asarray(score).ravel()
    label = np.asarray(label).astype(bool).ravel()
    thresholds = np.asarray(thresholds, dtype=np.float64).ravel()
    if score.ndim != 1 or score.size != label.size or score.size == 0 or not np.isfinite(score).all():
        raise ValueError('Scores/labels must be finite, nonempty, aligned 1D vectors')
    starts, ends = _segments(label,boundaries)
    maxima = np.array([score[a:b].max() for a,b in zip(starts,ends)])
    lengths = ends - starts
    if len(maxima):
        order = np.argsort(maxima)
        sorted_max = maxima[order]
        cum_lengths = np.r_[0, np.cumsum(lengths[order])]
        tp = int(label.sum()) - cum_lengths[np.searchsorted(sorted_max, thresholds, side='right')]
        if preserve_upstream_index_zero and boundaries is None and label[0]:
            tp -= ((maxima[0] > thresholds) & (score[0] <= thresholds)).astype(np.int64)
    else:
        tp = np.zeros(len(thresholds),dtype=np.int64)
    negatives = np.sort(score[~label])
    fp = len(negatives) - np.searchsorted(negatives,thresholds,side='right')
    fn = int(label.sum())-tp
    tn = int((~label).sum())-fp
    precision = np.divide(tp,tp+fp,out=np.zeros(len(tp),dtype=np.float64),where=(tp+fp)>0)
    recall = np.divide(tp,tp+fn,out=np.zeros(len(tp),dtype=np.float64),where=(tp+fn)>0)
    f1 = np.divide(2*precision*recall,precision+recall,
                   out=np.zeros(len(tp),dtype=np.float64),where=(precision+recall)>0)
    return {'f1':f1,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'precision':precision,'recall':recall}


def paper_best_f1(score,labels,steps=200):
    """Return the same dict as omnianomaly.bf_search on a native concatenated series.

    The fast path only selects a threshold; final metrics/latency are calculated
    once by the original implementation for exact parity.
    """
    scores=np.asarray(score).reshape(-1)
    labels=np.asarray(labels).reshape(-1)
    if np.unique(labels).size != 2:
        return omnianomaly.bf_search(scores,labels.copy(),float(scores.min()),float(scores.mean()+3*scores.std()),steps,verbose=False)
    thresholds=np.linspace(float(np.min(scores)),float(np.mean(scores)+3*np.std(scores)),steps)
    optimum=_stats(scores,labels,thresholds)
    index=int(np.argmax(optimum['f1']))
    # Avoid tiny floating-point ties changing the selected threshold.
    selected=float(thresholds[index])
    target,latency=omnianomaly.calc_seq(scores,labels.copy(),selected)
    return {'bf_f1':float(target[0]),'bf_precision':float(target[1]),'bf_recall':float(target[2]),
            'bf_TP':int(target[3]),'bf_TN':int(target[4]),'bf_FP':int(target[5]),
            'bf_FN':int(target[6]),'bf_ROC/AUC':float(target[7]),
            'bf_threshold':selected,'bf_latency':float(latency)}


def dense_best_f1(score,labels,boundaries=None):
    """Exact score-threshold sweep for the specified point-adjustment definition.

    This intentionally uses test labels to find the best possible *oracle* F1.
    """
    s=np.asarray(score).reshape(-1)
    y=np.asarray(labels).reshape(-1)
    candidates=np.r_[np.nextafter(float(np.min(s)), -np.inf),np.unique(s)]
    t=_stats(s,y,candidates,boundaries=boundaries,
             preserve_upstream_index_zero=(boundaries is None))
    k=int(np.argmax(t['f1']))
    return {'bf_dense_f1':float(t['f1'][k]),'bf_dense_precision':float(t['precision'][k]),
            'bf_dense_recall':float(t['recall'][k]),'bf_dense_threshold':float(candidates[k]),
            'bf_dense_n_candidate_thresholds':int(len(candidates)),
            'bf_dense_point_adjustment': 'upstream_global' if boundaries is None else 'fragment_aware',
            'bf_dense_oracle_uses_test_labels':True}


def decoder_arrays(model,loader,device,progress_title="EVAL decoder"):
    from console_telemetry import progress_batches
    """Compute BOTH decoder outputs once: alpha belongs to evaluation, not training."""
    import torch
    first,second,target,labels=[],[],[],[]
    model.eval()
    with torch.inference_mode():
        for windows,anoms in progress_batches(loader, title=progress_title):
            windows=windows.to(device).permute(1,0,2)
            last=windows[-1:,:,:]
            z1,z2=model(windows,last)
            first.append(z1[0].detach().cpu().numpy())
            second.append(z2[0].detach().cpu().numpy())
            target.append(last[0].detach().cpu().numpy())
            label=anoms[:,-1,:] if loader.dataset.padding else anoms[:,0,:]
            labels.append((label.sum(dim=1)>0).numpy().astype(np.int8))
    return tuple(np.concatenate(part,axis=0) for part in (first,second,target,labels))


def anomaly_scores(arrays,alpha):
    a,b,target,_=arrays
    return np.mean((float(alpha)*a+(1.0-float(alpha))*b-target)**2,axis=1)


def native_metrics_from_arrays(train_arrays,test_arrays,alpha,pot_q):
    tr=anomaly_scores(train_arrays,alpha)
    te=anomaly_scores(test_arrays,alpha)
    label=test_arrays[3]
    pot,_,_=omnianomaly.pot_eval(tr,te,label.copy(),pot_q)
    bf=paper_best_f1(te,label)
    return {**pot,**bf}, dense_best_f1(te,label)
