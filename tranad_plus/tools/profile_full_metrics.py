"""Full, *real-split* TranAD+ timing evaluation: train/test decoder outputs,
21 alpha blends, original 200-grid BF, dense oracle BF, POT/SPOT and OPS-SAT
fragment-aware point adjustment. Test labels used ONLY for profiling oracle metrics.

This module never selects a scientific checkpoint or writes predictions.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


def make_eval_data(name, hp):
    """Rebuild the original official splits and train-fitted scalers, once per run."""
    if name == 'OPS-SAT':
        from opssat_pipeline import prepare_data, FragmentWindows
        tr, te, scalers, inventory = prepare_data(hp)
        return (FragmentWindows(tr, scalers, hp['window_sz'], with_labels=False),
                FragmentWindows(te, scalers, hp['window_sz'], with_labels=False),
                te, inventory)
    from TranADPlus.src.datasets import TSDataset_tracks
    from config import PROCESSED_DATA_DIR
    base=PROCESSED_DATA_DIR / name
    def paired(split):
        a=sorted((base/split/'Tracks').glob('*.npy'))
        b=sorted((base/split/'Labels').glob('*.npy'))
        if not a or len(a)!=len(b) or [x.stem for x in a]!=[x.stem for x in b]:
            raise ValueError(f'{name} {split}: tracks/labels missing or mismatched')
        return list(map(str,a)), list(map(str,b))
    trp,trl=paired('Train'); tep,tel=paired('Test')
    train=TSDataset_tracks(window_size=hp['window_sz'],track_paths=trp,anom_paths=trl,
                           padding=hp['padding'],scaling=True,downsample=hp.get('downsample'))
    test=TSDataset_tracks(window_size=hp['window_sz'],track_paths=tep,anom_paths=tel,
                          padding=hp['padding'],scaling=True,scaler=train.scaler,
                          downsample=hp.get('downsample'))
    return train,test,None,{'train_tracks':len(trp),'test_tracks':len(tep),
                             'train_points_scored':len(train),'test_points_scored':len(test)}


def eval_batches(ds, mode, batch_size, device, *, max_windows=0):
    """Return (iterable of windows, cache prep seconds, n) with the same order.
    Label collection is independent of the selected memory path.
    """
    from torch.utils.data import Subset
    t0=time.perf_counter()
    n=len(ds) if not max_windows else min(len(ds),max_windows)
    if n<1: raise ValueError('Empty evaluation split')
    if mode in ('ram','vram'):
        # OPS-SAT stores fragment-safe windows; SMAP/MSL use native merged tracks.
        if hasattr(ds,'windows'):
            windows=ds.windows[:n].to(dtype=torch.float32).contiguous()
        else:
            track=ds.big_track
            # Same index -> native TSDataset_tracks windows, float32 on GPU as in native forward.
            windows=track.unfold(0,int(ds.window_size),1).permute(0,2,1)[:n].to(dtype=torch.float32).contiguous()
        if mode=='vram':
            if torch.device(device).type=='cuda':
                import os
                free,total=torch.cuda.mem_get_info(device)
                required=windows.numel()*windows.element_size()
                budget=min(int(float(os.environ.get('TRANADPLUS_GPU_CACHE_MAX_GIB','16'))*2**30),int(free*.5))
                if required>budget:
                    raise MemoryError(f'Evaluation VRAM cache requires {required/2**30:.2f} GiB, budget {budget/2**30:.2f} GiB')
            windows=windows.to(device)
        elif torch.device(device).type=='cuda':
            windows=windows.pin_memory()
        batches=[windows[i:i+batch_size] for i in range(0,n,batch_size)]
    else:
        class WindowsOnly(torch.utils.data.Dataset):
            def __init__(self, original,n):self.original=original;self.n=n
            def __len__(self):return self.n
            def __getitem__(self,k):
                item=self.original[k]
                return item[0] if isinstance(item,(tuple,list)) else item
        batches=DataLoader(WindowsOnly(ds,n),batch_size=batch_size,shuffle=False,num_workers=0,
                           pin_memory=mode=='pinned' and torch.device(device).type=='cuda')
    sync(device)
    return batches,time.perf_counter()-t0,n


def extract(model,ds,mode,bs,device,*,max_windows=0):
    """Score the actual official train/test windows; cache both decoders once."""
    sources,prep,n=eval_batches(ds,mode,bs,device,max_windows=max_windows)
    a=[];b=[];y=[]
    sync(device);start=time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for item in sources:
            x=item.to(device=device,dtype=torch.float32,
                      non_blocking=mode in ('pinned','ram')).permute(1,0,2)
            target=x[-1:]
            o1,o2=model(x,target)
            a.append(o1[0].float().cpu().numpy())
            b.append(o2[0].float().cpu().numpy())
            y.append(target[0].float().cpu().numpy())
    sync(device);elapsed=time.perf_counter()-start
    if hasattr(ds,'point_labels'):
        labels=np.asarray(ds.point_labels[:n],dtype=np.int8)
    elif ds.big_anom is not None:
        offset=ds.window_size-1 if ds.padding else 0
        labels=(ds.big_anom[offset:offset+n].numpy().sum(axis=1)>0).astype(np.int8)
    else: raise ValueError('No test labels; cannot compute full metrics')
    return (np.concatenate(a),np.concatenate(b),np.concatenate(y),labels),{
        'seconds':elapsed,'data_preparation_seconds':prep,'windows':n}


def pot_bounded(train_score,test_score,labels,q,max_attempts=100):
    """Upstream SPOT initialization/threshold/PA, bounded to avoid infinite hang.
    If successful at the same SPOT level, metrics are identical to upstream pot_eval.
    """
    from TranADPlus.src.spot import SPOT
    from TranADPlus.src import omnianomaly
    level=.98
    error=None
    for attempt in range(max_attempts):
        try:
            s=SPOT(q);s.fit(train_score,test_score)
            s.initialize(level=level,min_extrema=False,verbose=False)
            break
        except Exception as e:
            error=f'{type(e).__name__}: {e}'
            level*=.999
    else:
        raise RuntimeError(f'SPOT did not initialize in {max_attempts} attempts: {error}')
    result=s.run(dynamic=False)
    threshold=float(np.mean(result['thresholds']))
    pred,latency=omnianomaly.adjust_predicts(test_score,labels.copy(),threshold,calc_latency=True)
    f1,prec,rec,tp,tn,fp,fn,auc=omnianomaly.calc_point2point(pred,labels.copy())
    return {'pot_f1':float(f1),'pot_precision':float(prec),'pot_recall':float(rec),
            'pot_TP':int(tp),'pot_TN':int(tn),'pot_FP':int(fp),'pot_FN':int(fn),
            'pot_ROC/AUC':float(auc),'pot_threshold':threshold,'pot_latency':float(latency),
            'pot_spot_level':level,'pot_spot_attempts':attempt+1}


def metric_sweep(name,train_arr,test_arr,groups,pot_q,*,alpha_count=21,full_labels=True):
    """Complete per-alpha evaluation, returns cost per component and best-of-alpha.
    These metrics are test-label oracles; values are NOT scientific study results.
    """
    from paper_metrics import paper_best_f1,dense_best_f1,anomaly_scores
    from opssat_pipeline import original_metrics
    if alpha_count<1:raise ValueError('alpha_count must be positive')
    alpha_values=np.linspace(0,1,alpha_count)
    components={'scores_seconds':0.,'pot_seconds':0.,'bf200_seconds':0.,'dense_seconds':0.}
    rows=[]
    for alpha in alpha_values:
        start=time.perf_counter()
        tr=anomaly_scores(train_arr,float(alpha))
        te=anomaly_scores(test_arr,float(alpha))
        components['scores_seconds']+=time.perf_counter()-start
        labels=test_arr[3]
        if name=='OPS-SAT':
            # This adapter is not part of original repository; correct fragment boundaries.
            # Each call executes BF200, dense BF and bounded POT exactly as official OPS evaluator.
            start=time.perf_counter()
            bf,pot=original_metrics(tr,te,labels,groups,pot_q)
            delta=time.perf_counter()-start
            components['opssat_combined_metrics_seconds']=components.get('opssat_combined_metrics_seconds',0)+delta
            row={'alpha':float(alpha),'bf_f1':bf['bf_f1'],'bf_dense_f1':bf['bf_dense_f1'],
                 'pot_f1':pot.get('pot_f1'),'pot_status':pot.get('status')}
            if pot.get('status')!='measured':row['pot_error']=pot.get('reason')
        else:
            start=time.perf_counter()
            try:pot=pot_bounded(tr,te,labels,pot_q)
            except Exception as exc:pot={'pot_f1':None,'pot_error':str(exc)}
            components['pot_seconds']+=time.perf_counter()-start
            start=time.perf_counter();bf=paper_best_f1(te,labels)
            components['bf200_seconds']+=time.perf_counter()-start
            start=time.perf_counter();dense=dense_best_f1(te,labels)
            components['dense_seconds']+=time.perf_counter()-start
            row={'alpha':float(alpha),'bf_f1':bf['bf_f1'],'bf_dense_f1':dense['bf_dense_f1'],
                 'pot_f1':pot['pot_f1']}
            if 'pot_error' in pot:row['pot_error']=pot['pot_error']
        rows.append(row)
    def max_field(field):
        values=[(r[field],r['alpha']) for r in rows if r[field] is not None]
        return {'value':float(max(values)[0]),'alpha':float(max(values)[1])} if values else None
    return {'alpha_count':alpha_count,'pot':max_field('pot_f1'),
            'best_f1_original':max_field('bf_f1'),
            'best_f1_dense_oracle':max_field('bf_dense_f1'),
            'pot_failed_alpha_count':sum(x['pot_f1'] is None for x in rows),
            'pot_errors':[{'alpha':x['alpha'],'error':x.get('pot_error')} for x in rows if x['pot_f1'] is None],
            'components':components,'label_source':'official_test_oracle_for_timing_only',
            'full_alpha_results':rows}


def evaluate_full(model,profile,ds_train,ds_test,groups,mode,bs,device,*,max_windows=0,alpha_count=21):
    sync(device);t0=time.perf_counter()
    train_arr,train_perf=extract(model,ds_train,mode,bs,device,max_windows=max_windows)
    test_arr,test_perf=extract(model,ds_test,mode,bs,device,max_windows=max_windows)
    sync(device);scoring_seconds=time.perf_counter()-t0
    # OPS-SAT partial smoke runs end at an arbitrary test point; fragment-aware
    # adjustment must use exactly the measured prefix, never the full groups.
    if profile['dataset']=='OPS-SAT' and max_windows:
        trimmed=[];remaining=len(test_arr[3])
        for group in groups:
            if remaining<=0:break
            used=min(remaining,len(group['values']))
            trimmed.append({**group,'values':group['values'][:used]})
            remaining-=used
        if remaining:raise AssertionError('OPS-SAT test subset exceeds fragment lengths')
        groups=trimmed
    t1=time.perf_counter()
    metrics=metric_sweep(profile['dataset'],train_arr,test_arr,groups,float(profile['pot_q']),alpha_count=alpha_count)
    result={'score_forward_and_preparation_seconds':scoring_seconds,'metric_seconds':time.perf_counter()-t1,
            'train':train_perf,'test':test_perf,'metrics':metrics,
            'evaluation_full_test':max_windows==0 or max_windows>=len(ds_test),
            'evaluation_full_train':max_windows==0 or max_windows>=len(ds_train)}
    return result
