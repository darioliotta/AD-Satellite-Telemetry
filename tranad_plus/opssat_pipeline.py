"""OPS-SAT raw-signal adapter for the integrated TranAD+ campaign.

A single shared n_feats=1 original TranAD across 9 telemetry channels.
Never construct windows across fragment boundaries. Per-channel MinMax fitting
on NOMINAL official training fragments only. Labels are fragment-level weak
labels; per-point metrics are proxies, not precise onset/offset scores.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset

from config import NATIVE_PACKAGE, RESULTS_DIR, checkpoint_path
from TranADPlus.src.models import TranAD
from TranADPlus.src.omnianomaly import adjust_predicts
from TranADPlus.src.spot import SPOT
from observability.measure_experiment import write_json_atomic, hash_file
from opssat_scoring import selected_scoring, evaluate_selected, point_channels

DATASET_NAME = 'OPS-SAT'


def source_csv(profile):
    value = profile.get('segments_csv')
    if not value:
        raise ValueError('OPS-SAT profile requires segments_csv; set it in experiment_plan.json')
    file = Path(value).expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError(f'OPS-SAT segments.csv not found: {file}')
    return file


def _select(groups, cap, seed, label_stratified=False):
    if cap is None:
        return groups
    cap = int(cap)
    if cap < 1:
        raise ValueError('fragment cap must be positive')
    rng = np.random.default_rng(int(seed))
    by_chan = {}
    for g in groups:
        by_chan.setdefault(g['channel'], []).append(g)
    selected = []
    for ch, arr in sorted(by_chan.items()):
        if label_stratified:
            normal = [g for g in arr if g['label'] == 0]
            anomaly = [g for g in arr if g['label'] == 1]
            if normal and anomaly:
                # preserve both classes for quick tests when possible
                n0 = min(len(normal), max(1, cap // 2))
                n1 = min(len(anomaly), max(1, cap - n0))
                while n0+n1 < min(cap,len(arr)):
                    if n0 < len(normal): n0 += 1
                    elif n1 < len(anomaly): n1 += 1
                    else: break
                chosen = ([normal[int(i)] for i in rng.choice(len(normal), n0, replace=False)]
                          + [anomaly[int(i)] for i in rng.choice(len(anomaly), n1, replace=False)])
            else:
                chosen = [arr[int(i)] for i in rng.choice(len(arr), min(cap,len(arr)),replace=False)]
        else:
            chosen = [arr[int(i)] for i in rng.choice(len(arr),min(cap,len(arr)),replace=False)]
        selected.extend(chosen)
    return sorted(selected,key=lambda g:g['id'])


def prepare_data(profile, *, return_validation=False):
    file=source_csv(profile)
    df=pd.read_csv(file,usecols=['channel','segment','timestamp','value','train','anomaly','label'])
    if df.empty or df['value'].isna().any() or not np.isfinite(df['value'].to_numpy()).all():
        raise ValueError('OPS-SAT has missing/nonfinite values or no rows')
    if not df['train'].isin([0,1]).all() or not df['anomaly'].isin([0,1]).all():
        raise ValueError('OPS-SAT nonbinary labels/split')
    if not ((df['label']=='anomaly').to_numpy()==(df['anomaly']==1).to_numpy()).all():
        raise ValueError('OPS-SAT inconsistent text anomaly labels')
    channels=profile.get('channels')
    if channels:
        allowed=set(channels)
        df=df[df.channel.isin(allowed)]
        if set(df.channel.unique())!=allowed: raise ValueError('Some requested OPS-SAT channels not found')
    groups=[]
    # groupby sort=False preserves the time order as provided in the CSV; never combine segments
    for segid, block in df.groupby('segment',sort=False):
        if any(block[c].nunique()!=1 for c in ('channel','train','anomaly','label')):
            raise ValueError(f'OPS-SAT mixed metadata within segment {segid}')
        groups.append({'id':int(segid),'channel':str(block.channel.iloc[0]),
                       'values':block.value.to_numpy(dtype=np.float32),
                       'timestamps':block.timestamp.astype(str).to_numpy(dtype='U40'),
                       'train':int(block.train.iloc[0]),'label':int(block.anomaly.iloc[0])})
    train=[g for g in groups if g['train']==1 and g['label']==0]
    test=[g for g in groups if g['train']==0]
    split_seed=profile.get('split_seed',profile.get('seed',2026))
    train=_select(train,profile.get('max_train_fragments_per_channel'),split_seed)
    test=_select(test,profile.get('max_test_fragments_per_channel'),split_seed,True)
    # The original split remains unchanged when validation_fraction=0.
    # A fraction of nominal *official-train* fragments is held out (by channel),
    # alongside all anomalous official-train fragments. These groups NEVER enter
    # the optimizer or the MinMax fit. Test labels are NEVER used for selection.
    fraction=float(profile.get('validation_fraction',0.0))
    if not 0.0 <= fraction < 0.5:
        raise ValueError('validation_fraction must be in [0, 0.5)')
    validation=[]
    if fraction:
        rng=np.random.default_rng(int(split_seed)+13579)
        heldout=set()
        for channel in sorted({g['channel'] for g in train}):
            arr=[g for g in train if g['channel']==channel]
            if len(arr)>=2:
                k=min(len(arr)-1,max(1,int(round(len(arr)*fraction))))
                heldout.update(arr[int(i)]['id'] for i in rng.choice(len(arr),k,replace=False))
        validation=[g for g in train if g['id'] in heldout]
        train=[g for g in train if g['id'] not in heldout]
        validation_anomalies=[g for g in groups if g['train']==1 and g['label']==1
                              and (not profile.get('channels') or g['channel'] in profile['channels'])]
        validation.extend(_select(validation_anomalies,
                                  profile.get('max_validation_anomalies_per_channel'),
                                  int(split_seed)+24680))
        validation=sorted(validation,key=lambda g:g['id'])
    if {g['id'] for g in validation} & ({g['id'] for g in test}|{g['id'] for g in train}):
        raise RuntimeError('OPS-SAT train/validation/test fragment leakage')
    if fraction and (not any(g['label'] for g in validation) or not any(not g['label'] for g in validation)):
        raise ValueError('Validation must contain both normal and anomalous fragments')
    if not train or not test: raise ValueError('Empty training or testing OPS-SAT split')
    if {g['id'] for g in train} & {g['id'] for g in test}:
        raise RuntimeError('OPS-SAT train/test overlap')
    scalers={}
    for channel in sorted({g['channel'] for g in train+test}):
        arrays=[g['values'] for g in train if g['channel']==channel]
        if not arrays:
            raise ValueError(f'No nominal training fragments available for channel {channel}')
        scalers[channel]=MinMaxScaler().fit(np.concatenate(arrays).reshape(-1,1))
    inventory={'source_csv':str(file),'source_sha256':hash_file(file),
             'n_original_fragments':len(groups),'n_train_fragments':len(train),'n_test_fragments':len(test),
             'n_train_points':sum(len(g['values']) for g in train),
             'n_test_points':sum(len(g['values']) for g in test),
             'n_training_anomalies_excluded':sum(g['train']==1 and g['label']==1 for g in groups),
             'validation_fraction':fraction,'split_seed':int(split_seed),'n_validation_fragments':len(validation),
             'n_validation_anomalous_fragments':sum(g['label'] for g in validation),
             'validation_fragment_ids':[g['id'] for g in validation],
             'channels':sorted(scalers),'train_fragment_ids':[g['id'] for g in train],
             'test_fragment_ids':[g['id'] for g in test],
             'feature_definition':'one raw value per timestep; per-channel normalization fitted on nominal training only',
             'window_policy':'causal left-edge pad independently for each fragment',
             'split_policy':('official train column; train normal-only; held-out normal + official-train anomalies for supervised validation; untouched official test' if fraction else 'official train column; normal-only training; all test labels')}
    return (train,test,scalers,inventory,validation) if return_validation else (train,test,scalers,inventory)


class FragmentWindows(Dataset):
    padding=True
    def __init__(self, groups,scalers,window,with_labels=True):
        self.groups=groups
        self.with_labels=with_labels
        matrices=[]; labels=[]; segment_ids=[]
        for g in groups:
            signal=scalers[g['channel']].transform(g['values'].reshape(-1,1)).astype(np.float32)
            padded=np.pad(signal,((window-1,0),(0,0)),mode='edge')
            view=np.lib.stride_tricks.sliding_window_view(padded,window_shape=window,axis=0)
            w=np.moveaxis(view,2,1).copy()
            if w.shape!=(len(signal),window,1): raise AssertionError('window shape')
            matrices.append(torch.from_numpy(w))
            labels.append(np.full(len(signal),g['label'],dtype=np.int8))
            segment_ids.append(np.full(len(signal),g['id'],dtype=np.int32))
        self.windows=torch.cat(matrices)
        self.point_labels=np.concatenate(labels)
        self.segment_ids=np.concatenate(segment_ids)
    def __len__(self):return len(self.windows)
    def __getitem__(self,idx):
        if self.with_labels:
            return self.windows[idx],int(self.point_labels[idx])
        return self.windows[idx]


def _atomic_torch(data,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.opssat_',suffix='.tmp');os.close(fd)
    try:
        torch.save(data,tmp)
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def ff_width(value, features=1):
    if isinstance(value,str) and value.endswith('x'):
        return max(1,int(float(value[:-1])*features))
    return int(value)


def make_model(profile):
    if not bool(profile.get('padding',True)):
        raise ValueError('OPS-SAT requires per-fragment left padding')
    if profile.get('downsample') is not None:
        raise ValueError('OPS-SAT downsample has not been implemented')
    return TranAD(n_feats=1,window_sz=int(profile['window_sz']),batch_sz=int(profile['batch_sz']),
                  dim_feedforward=ff_width(profile['dim_feedforward']),
                  num_encoder_layers=int(profile['num_layers']),
                  num_decoder_layers=int(profile['num_layers'])).float()


def _seed(seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _rng():
    value={'torch_cpu':torch.get_rng_state(),'numpy':(lambda state: (state[0], state[1].tolist(), state[2], state[3], state[4]))(np.random.get_state()),'python':random.getstate()}
    if torch.cuda.is_available():value['cuda']=torch.cuda.get_rng_state_all()
    return value


def _restore_rng(value):
    torch.set_rng_state(value['torch_cpu']);np.random.set_state((value['numpy'][0],np.asarray(value['numpy'][1],dtype=np.uint32),*value['numpy'][2:]));random.setstate(value['python'])
    if torch.cuda.is_available() and 'cuda' in value:torch.cuda.set_rng_state_all(value['cuda'])


def train(profile,device,*,stop_after_epoch=None):
    from gpu_settings import configure
    configure()
    identifier=profile['identifier'];seed=int(profile.get('seed',2026));_seed(seed)
    tr,te,scalers,inventory=prepare_data(profile)
    data=FragmentWindows(tr,scalers,int(profile['window_sz']),False)
    from gpu_batches import training_batches
    batches, transfer_path=training_batches(data,batch_size=int(profile['batch_sz']),
        shuffle=bool(profile['shuffle']),device=device,num_workers=0)
    model=make_model(profile).to(device)
    from gpu_settings import make_adam
    optimizer=make_adam(model.parameters(),lr=float(profile['lr']),device=device)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,5,gamma=0.9)
    native=checkpoint_path(identifier)
    recovery=checkpoint_path(identifier).with_suffix(".resume.ckpt")
    if native.is_file():raise FileExistsError(f'Final checkpoint already exists: {native}')
    signature={'profile':{k:v for k,v in profile.items() if k!='segments_csv'},
               'adam_impl':os.environ.get('TRANADPLUS_ADAM_IMPL','standard'),
               'precision':os.environ.get('TRANADPLUS_CUDA_PRECISION','faithful'),
               'data_sha256':inventory['source_sha256'],'train_fragment_ids':inventory['train_fragment_ids']}
    start_epoch=0
    if recovery.is_file():
        saved=torch.load(recovery,map_location='cpu',weights_only=True)
        if saved.get('signature')!=signature:
            raise RuntimeError('OPS-SAT recovery checkpoint dataset/profile mismatch')
        start_epoch=int(saved['epoch'])+1
        model.load_state_dict(saved['model_state_dict']);optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict']);_restore_rng(saved['rng_state'])
        print(f'OPS-SAT RESUME epoch {start_epoch+1}/{profile["epochs"]}',flush=True)
    from epoch_telemetry import write_epoch
    from console_telemetry import emit_epoch
    final_loss=None
    for epoch in range(start_epoch,int(profile['epochs'])):
        start=time.perf_counter();model.train()
        loss_total=torch.zeros((),device=device,dtype=torch.float64);batch_count=0
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        for windows in batches:
            x=windows.to(device, non_blocking=(transfer_path == 'cpu_pinned')).permute(1,0,2)
            target=x[-1:,:,:]
            z1,z2=model(x,target)
            weight=1.0/(epoch+1)
            loss=(weight*(z1-target).square()+(1-weight)*(z2-target).square()).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            loss_total.add_(loss.detach().double());batch_count+=1
        if not batch_count:raise RuntimeError('No batches')
        final_loss=float((loss_total/batch_count).item());scheduler.step()
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        write_epoch(DATASET_NAME,identifier,epoch+1,final_loss,optimizer.param_groups[0]['lr'],elapsed,
                    batch_count,len(data),str(device),
                    torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
                    torch.cuda.max_memory_reserved(device) if str(device).startswith('cuda') else None)
        _atomic_torch({'epoch':epoch,'signature':signature,'last_loss':final_loss,
                       'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                       'scheduler_state_dict':scheduler.state_dict(),'rng_state':_rng()},recovery)
        emit_epoch(dataset=DATASET_NAME, identifier=identifier,
                   epoch=epoch+1, epochs=int(profile['epochs']), loss=final_loss,
                   lr=optimizer.param_groups[0]['lr'], seconds=elapsed,
                   batches=batch_count, windows=len(data),
                   peak_allocated_bytes=(torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None))
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            raise RuntimeError('Simulated interruption after durable recovery checkpoint')
    if final_loss is None:final_loss=float(torch.load(recovery,map_location='cpu',weights_only=True)['last_loss'])
    hp={**profile,'dataset_str':DATASET_NAME,'model_str':'TranAD','identifier':identifier,
        'features':1,'dim_feedforward':ff_width(profile['dim_feedforward']),
        'final_loss':final_loss,'scaling':True}
    _atomic_torch({'hp_dict':hp,'model_states':model.state_dict(),
                   'optimizer_states':optimizer.state_dict(),'scheduler_states':scheduler.state_dict(),
                   'opssat_inventory':inventory},native)
    recovery.unlink(missing_ok=True)
    print(f'OPS-SAT completed; final native checkpoint: {native}',flush=True)


def score_decoders(model,dataset,device,batch_size,progress_title="EVAL decoder"):
    from console_telemetry import progress_batches
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    from gpu_batches import maybe_cuda_batches
    resident=maybe_cuda_batches(dataset,batch_size=batch_size,shuffle=False,device=device)
    batches=resident if resident is not None else loader
    first=[];second=[];targets=[]
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch in progress_batches(batches, title=progress_title):
            windows=batch[0] if isinstance(batch,(tuple,list)) else batch
            # score_decoders receives either CUDA-resident batches or an unpinned CPU DataLoader.
            # No training-local transfer_path variable exists in this function.
            x=windows.to(device, non_blocking=False).permute(1,0,2);y=x[-1:,:,:]
            z1,z2=model(x,y)
            first.append(z1[0].cpu().numpy().reshape(-1))
            second.append(z2[0].cpu().numpy().reshape(-1))
            targets.append(y[0].cpu().numpy().reshape(-1))
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    return (np.concatenate(first),np.concatenate(second),np.concatenate(targets),
            {'elapsed_seconds_including_data_transfer_and_scoring':elapsed,
             'points_scored':len(dataset),'points_per_second':len(dataset)/elapsed if elapsed else None,
             'gpu_peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)) if str(device).startswith('cuda') else None,
             'gpu_peak_reserved_bytes':int(torch.cuda.max_memory_reserved(device)) if str(device).startswith('cuda') else None})


def point_metrics(labels,preds,prefix):
    labels=np.asarray(labels,dtype=np.int8);preds=np.asarray(preds,dtype=np.int8)
    tn,fp,fn,tp=confusion_matrix(labels,preds,labels=[0,1]).ravel()
    return {f'{prefix}_f1':float(f1_score(labels,preds,zero_division=0)),
            f'{prefix}_precision':float(precision_score(labels,preds,zero_division=0)),
            f'{prefix}_recall':float(recall_score(labels,preds,zero_division=0)),
            f'{prefix}_TP':int(tp),f'{prefix}_TN':int(tn),f'{prefix}_FP':int(fp),f'{prefix}_FN':int(fn),
            # Original misnamed AUC computed on BINARY adjusted predictions; balanced accuracy.
            f'{prefix}_ROC/AUC':float((tp/(tp+fn)+tn/(tn+fp))/2) if tp+fn and tn+fp else None}


def adjust_per_fragment(scores,labels,groups,threshold):
    predictions=[];offset=0
    for group in groups:
        n=len(group['values'])
        # Upstream adjust_predicts backfill stops at index 1, so a prepended
        # nominal sentinel preserves intended adjustment of fragment sample 0.
        padded_scores=np.r_[threshold, scores[offset:offset+n]]
        padded_labels=np.r_[0, labels[offset:offset+n]]
        predictions.append(adjust_predicts(padded_scores,padded_labels,threshold)[1:].astype(np.int8))
        offset+=n
    if offset!=len(scores):raise AssertionError('Segment length mismatch')
    return np.concatenate(predictions)


def _pot_threshold(train_scores,test_scores,q):
    # Same SPOT thresholding algorithm as the authors, but limit failed initialization attempts
    # instead of the upstream unbounded loop.
    level=0.98
    error=None
    for _ in range(35):
        try:
            spot=SPOT(q);spot.fit(train_scores,test_scores)
            spot.initialize(level=level,min_extrema=False,verbose=False)
            value=spot.run(dynamic=False)
            threshold=float(np.mean(value['thresholds']))
            if np.isfinite(threshold):return threshold,None
            raise ValueError('Nonfinite POT threshold')
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            level*=0.999
    return None,error


def original_metrics(train_scores,scores,labels,groups,pot_q,steps=200):
    """Paper BF grid and POT with PA strictly constrained by OPS-SAT fragments.

    Anomaly intervals are whole weakly annotated fragments. BF counts use a
    vectorized sweep, avoiding 200 Python loops across all samples. The final
    selected threshold is re-evaluated with the v9 fragment PA function.
    """
    from paper_metrics import _stats, dense_best_f1
    low=float(np.min(scores)); high=float(np.mean(scores)+3*np.std(scores))
    thresholds=np.linspace(low,high,steps)
    boundaries=np.cumsum([len(g['values']) for g in groups])[:-1]
    statistics=_stats(scores,labels,thresholds,boundaries=boundaries,
                      preserve_upstream_index_zero=False)
    best_at=int(np.argmax(statistics['f1']))
    thr=float(thresholds[best_at])
    adjusted=adjust_per_fragment(scores,labels,groups,thr)
    bf_best={**point_metrics(labels,adjusted,'bf'),'bf_threshold':thr}
    detected=0;delay=0;offset=0
    for group in groups:
        n=len(group['values'])
        if group['label']:
            hit=np.flatnonzero(scores[offset:offset+n]>thr)
            if len(hit):detected+=1;delay+=int(hit[0])
        offset+=n
    bf_best['bf_latency']=float(delay/(detected+1e-4))
    bf_dense=dense_best_f1(scores,labels,boundaries=boundaries)
    bf_best.update(bf_dense)
    pot_thr,pot_error=_pot_threshold(train_scores,scores,float(pot_q))
    if pot_thr is None:
        pot={'pot_f1':None,'pot_precision':None,'pot_recall':None,'pot_TP':None,
             'pot_TN':None,'pot_FP':None,'pot_FN':None,'pot_ROC/AUC':None,
             'pot_threshold':None,'status':'not_available','reason':pot_error}
    else:
        pot={**point_metrics(labels,adjust_per_fragment(scores,labels,groups,pot_thr),'pot'),
             'pot_threshold':pot_thr,'status':'measured'}
    return bf_best,pot


def _auc(labels,scores):
    if len(set(map(int,labels)))!=2:return {'roc_auc_continuous_scores':None,'average_precision_continuous_scores':None}
    return {'roc_auc_continuous_scores':float(roc_auc_score(labels,scores)),
            'average_precision_continuous_scores':float(average_precision_score(labels,scores))}


def evaluate_and_measure(profile,settings,device):
    from gpu_settings import configure
    configure()
    from observability.measure_experiment import architecture_statistics,benchmark,load_hardware_report
    from pipeline_metrics import measurement_key, epoch_history, update_summary_csv
    identifier=profile['identifier'];tr,te,scalers,inventory,validation=prepare_data(profile,return_validation=True)
    result_dir=RESULTS_DIR/DATASET_NAME;result_dir.mkdir(parents=True,exist_ok=True)
    final=result_dir/f'{identifier}.ckpt'
    if not final.is_file():
        raise FileNotFoundError(f'Missing archived OPS-SAT final checkpoint: {final}')
    saved=torch.load(final,map_location='cpu',weights_only=True)
    if saved['opssat_inventory']['source_sha256']!=inventory['source_sha256']:
        raise RuntimeError('OPS-SAT source CSV changed since training')
    for key in ('train_fragment_ids','validation_fragment_ids'):
        if saved['opssat_inventory'].get(key,[]) != inventory.get(key,[]):
            raise RuntimeError(f'OPS-SAT {key} mismatch between checkpoint and evaluation')
    model=make_model(profile)
    model.load_state_dict(saved['model_states']);model=model.to(device).eval()
    tr_data=FragmentWindows(tr,scalers,int(profile['window_sz']))
    te_data=FragmentWindows(te,scalers,int(profile['window_sz']))
    warmup=int(settings['warmup']);iterations=int(settings['iterations'])
    sample=te_data.windows[0:1].permute(1,0,2)
    inference=benchmark(model,torch.device(device),sample,warmup,iterations)
    scorebatch=int(settings['score_batch_size'])
    t1,t2,tt,perf_tr=score_decoders(model,tr_data,device,scorebatch,'EVAL OPS-SAT TRAIN-SCORES')
    v1,v2,vv,perf_te=score_decoders(model,te_data,device,scorebatch,'EVAL OPS-SAT TEST-SCORES')
    alpha=float(settings['alpha']);beta=1-alpha
    tr_scores=(alpha*t1+beta*t2-tt)**2
    scores=(alpha*v1+beta*v2-vv)**2
    labels=te_data.point_labels
    # Supervised selection uses only held-out official-train fragments and never
    # touches official test labels until after candidate selection is frozen.
    selection_report={'status':'not_requested','reason':'validation_fraction=0; legacy protocol preserved'}
    validation_outputs=None; val_data=None
    if validation:
        val_data=FragmentWindows(validation,scalers,int(profile['window_sz']))
        a1,a2,at,perf_val=score_decoders(model,val_data,device,scorebatch,'EVAL OPS-SAT VAL-SCORES')
        validation_outputs=(a1,a2,at)
        choice=selected_scoring((t1,t2,tt),validation_outputs,tr,validation)
        selection_report={'status':'measured',
                          **evaluate_selected((t1,t2,tt),(v1,v2,vv),tr,te,choice,
                                              float(settings['threshold_train_quantile'])),
                          'validation_scoring_performance':perf_val,
                          'train_fragment_ids':inventory['train_fragment_ids'],
                          'validation_fragment_ids':inventory['validation_fragment_ids'],
                          'test_fragment_ids':inventory['test_fragment_ids']}
    threshold=float(np.quantile(tr_scores,float(settings['threshold_train_quantile'])))
    raw_pred=(scores>threshold).astype(np.int8)
    raw={**point_metrics(labels,raw_pred,'raw'),**_auc(labels,scores),
         'status':'measured','point_adjustment':False,'threshold':threshold,
         'threshold_source':f'training_score_quantile_{settings["threshold_train_quantile"]}',
         'alpha_fixed_pre_test':alpha,'beta_fixed_pre_test':beta,
         'n_points':len(labels),'n_anomalies':int(labels.sum()),
         'scoring_performance':{'train':perf_tr,'test':perf_te},
         'label_protocol':'fragment-level weak labels repeated on points (NOT precise onset)',
         'score_formula':'(alpha*decoder1+(1-alpha)*decoder2-target)^2; 1 feature'}
    raw['confusion_matrix']={name:raw[f'raw_{name.upper()}'] for name in ('tp','tn','fp','fn')}
    raw.update({'precision':raw['raw_precision'],'recall':raw['raw_recall'],'f1':raw['raw_f1']})
    adjusted=adjust_per_fragment(scores,labels,te,threshold)
    pa={'status':'measured',**point_metrics(labels,adjusted,'point_adjusted'),
        'point_adjustment':'per_fragment_only','threshold':threshold,
        'note':'Uses ground truth to fill each hit anomalous fragment; not online-deployable.'}
    seg_labels=np.array([g['label'] for g in te],dtype=np.int8)
    offset=0; seg_max=[];seg_q95=[];seg_hit=[];seg_details=[]
    for g in te:
        n=len(g['values']);s=scores[offset:offset+n]
        seg_max.append(float(np.max(s)));seg_q95.append(float(np.quantile(s,0.95)))
        seg_hit.append(int(np.any(s>threshold)))
        seg_details.append({'segment_id':g['id'],'channel':g['channel'],'label':g['label'],
                            'points':n,'max_score':seg_max[-1],'p95_score':seg_q95[-1],
                            'n_raw_alerts':int(np.sum(s>threshold))})
        offset+=n
    seg_max=np.asarray(seg_max);seg_q95=np.asarray(seg_q95)
    segment={**point_metrics(seg_labels,seg_hit,'segment'),**_auc(seg_labels,seg_max),
             'status':'measured','aggregation':'max(score per fragment); warning: length-sensitive',
             'alternative_p95_auc':_auc(seg_labels,seg_q95),'n_segments':len(te),
             'event_detection':'any sample above fixed training threshold'}
    # Optional paper-style alpha/beta grid (same 21 mixes as the original runner).
    # It is selected using *test labels*, so reported as an oracle diagnostic.
    alphas = np.linspace(0,1,21) if settings.get('paper_alpha_search',False) else [alpha]
    bf=None;pot=None; best_bf_alpha=None;best_pot_alpha=None;best_dense=None
    from console_telemetry import emit_evaluation
    for index, mix in enumerate(alphas, 1):
        mix=float(mix)
        emit_evaluation(dataset=DATASET_NAME, identifier=identifier,
                        index=index, total=len(alphas), alpha=mix)
        this_train=(mix*t1+(1-mix)*t2-tt)**2
        this_test=(mix*v1+(1-mix)*v2-vv)**2
        b,p=original_metrics(this_train,this_test,labels,te,float(profile['pot_q']))
        if best_dense is None or b['bf_dense_f1']>best_dense['bf_dense_f1']:
            best_dense={k:v for k,v in b.items() if k.startswith('bf_dense_')}
            best_dense['bf_dense_alpha']=mix
            best_dense['bf_dense_beta']=1-mix
        if bf is None or b['bf_f1']>bf['bf_f1']:
            bf=b;best_bf_alpha=mix
        if p['pot_f1'] is not None and (pot is None or pot['pot_f1'] is None or p['pot_f1']>pot['pot_f1']):
            pot=p;best_pot_alpha=mix
        elif pot is None:
            pot=p
    bf['alpha']=best_bf_alpha;bf['beta']=1.0-best_bf_alpha
    bf.update(best_dense)
    pot['alpha']=best_pot_alpha;pot['beta']=1.0-best_pot_alpha if best_pot_alpha is not None else None
    # Use the same two named groups as the native runner's latest_reports/rebuild_summary.
    original={'pot':pot,'bf':bf}
    digest=hash_file(final)
    from datetime import datetime, timezone
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    source={'checkpoint':{'file':final.name,'sha256':digest},
            'dataset':DATASET_NAME,'identifier':identifier,
            'profile':profile,'results':original,'schema_version':2,
            'profile_sha256':hashlib.sha256(json.dumps({k:v for k,v in profile.items() if k!='segments_csv'},
                    sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'protocol':'Original TranAD+ BF search and SPOT threshold; segment-constrained point adjustment; alpha grid if enabled',
            'paper_alpha_search':bool(settings.get('paper_alpha_search',False)),
            'dataset_inventory':inventory,'artifacts':{}}
    # Standalone original-evaluation JSON follows the run's existing naming convention.
    eval_file=result_dir/f'{identifier}_evaluation_{stamp}.json'
    write_json_atomic(eval_file,source)
    settings_token,identity=measurement_key(settings,device)
    artifact=None
    if settings.get('hardware_json_dir'):
        candidate=Path(settings['hardware_json_dir'])/f'{identifier}.json'
        if candidate.is_file():artifact=candidate
    from ablation import parse_ablation, evaluate_opssat_stage1
    stage1 = (evaluate_opssat_stage1(model,tr_data,te_data,te,device,settings)
              if settings.get('inference_ablation',False) else {'status':'not_requested'})
    integrated={
      'ablation_design':parse_ablation(profile),'ablation_inference':stage1,
      'schema_version':'thesis-integrated-v4',
      'dataset':DATASET_NAME,'identifier':identifier,'profile':profile,
      'settings_digest':settings_token,'measurement_settings':settings,
      'execution_identity':identity,
      'checkpoint':{'file':final.name,'sha256':digest,'size_bytes':final.stat().st_size},
      'source':{'models_py':'TranADPlus/src/models.py','models_sha256':hash_file(NATIVE_PACKAGE/'src'/'models.py'),
                'scoring_sha256':hash_file(Path(__file__).with_name('opssat_scoring.py')),
                'evaluation_sha256':hash_file(Path(__file__))},
      'architecture':{'features':1,'window_sz':profile['window_sz'],'dim_feedforward':ff_width(profile['dim_feedforward']),
                      'num_layers':profile['num_layers'],'source_dataset':'OPSSAT-AD'},
      'model_statistics':architecture_statistics(model),'inference':inference,
      'training':epoch_history(DATASET_NAME,identifier),
      'original_paper_evaluation':{'source':eval_file.name,'metrics':original,
          'caution':'Adapted original TranAD+ procedure, not a published TranAD+ OPS-SAT benchmark. ' +
                    'Point adjustment constrained by fragment boundaries; alpha search if enabled is test-label oracle. POT may be unavailable on small subsets.'},
      'raw_pointwise_evaluation':raw,'point_adjusted_pointwise_evaluation':pa,
      'fragment_level_evaluation':segment,
      'validation_selected_evaluation':selection_report,
      'dataset_inventory':inventory,'hardware_acceleration':load_hardware_report(artifact),
      'by_channel':{},
      'environment':{'device':device,'torch':torch.__version__,
                     'cuda':{'status':'measured','name':torch.cuda.get_device_name(device)} if str(device).startswith('cuda') else {'status':'not_applicable'}},
      'methodology':{'train':'normal-only official train fragments; per-channel scaler; shared model',
                    'inference':'batch-one two-stage TranAD+ forward only',
                    'evaluation':'weak segment labels: raw, point-adjusted, event-level, BF/POT diagnostics',
                    'hardware':'GPU statistics not onboard; FPGA requires real synthesis'}}
    # Per-channel audit is essential for a shared single-feature model.
    by_channel={}
    for ch in sorted(set(g['channel'] for g in te)):
        mask=np.isin(te_data.segment_ids,[g['id'] for g in te if g['channel']==ch])
        subseg=np.array([i for i,g in enumerate(te) if g['channel']==ch])
        this_seg_labels=seg_labels[subseg]
        this_seg_score=seg_max[subseg]
        this_seg_pred=np.asarray(seg_hit)[subseg]
        by_channel[ch]={
            'n_points':int(mask.sum()),'n_fragments':int(len(subseg)),
            'n_anomalous_fragments':int(this_seg_labels.sum()),
            'raw_pointwise':point_metrics(labels[mask],raw_pred[mask],'raw'),
            'fixed_threshold_point_adjusted':point_metrics(labels[mask],adjusted[mask],'adjusted'),
            'event_fragment':point_metrics(this_seg_labels,this_seg_pred,'segment'),
            'fragment_score_auc':_auc(this_seg_labels,this_seg_score),
        }
    integrated['by_channel']=by_channel
    integrated['fragment_diagnostics']=seg_details
    if settings.get('save_raw_scores'):
        raw_file=result_dir/f'{identifier}_scores_{digest[:12]}_{settings_token}.npz'
        if not raw_file.exists():
            fd,tmp=tempfile.mkstemp(dir=result_dir,prefix='.opssat_scores_',suffix='.npz');os.close(fd)
            try:
                artifact_arrays=dict(scores=scores,labels=labels,segment_ids=te_data.segment_ids,
                    train_scores=tr_scores,train_segment_ids=tr_data.segment_ids,
                    segment_labels=seg_labels,segment_scores_max=seg_max,segment_scores_p95=seg_q95,
                    fixed_threshold=np.array([threshold]))
                if settings.get('export_decoder_arrays',False):
                    # All arrays are ordered exactly like FragmentWindows, including
                    # original per-point time and zero-based index within fragment.
                    artifact_arrays.update(
                        test_decoder1=v1,test_decoder2=v2,test_target=vv,
                        train_decoder1=t1,train_decoder2=t2,train_target=tt,
                        test_channel=point_channels(te),train_channel=point_channels(tr),
                        test_timestamp=np.concatenate([g['timestamps'] for g in te]),
                        train_timestamp=np.concatenate([g['timestamps'] for g in tr]),
                        test_fragment_point_index=np.concatenate([np.arange(len(g['values'])) for g in te]),
                        train_fragment_point_index=np.concatenate([np.arange(len(g['values'])) for g in tr]))
                    if val_data is not None:
                        artifact_arrays.update(
                            validation_decoder1=validation_outputs[0],
                            validation_decoder2=validation_outputs[1],
                            validation_target=validation_outputs[2],
                            validation_labels=val_data.point_labels,
                            validation_segment_ids=val_data.segment_ids,
                            validation_channel=point_channels(validation),
                            validation_timestamp=np.concatenate([g['timestamps'] for g in validation]),
                            validation_fragment_point_index=np.concatenate([np.arange(len(g['values'])) for g in validation]))
                np.savez_compressed(tmp,**artifact_arrays)
                os.replace(tmp,raw_file)
            finally:Path(tmp).unlink(missing_ok=True)
        raw['scores_file']=raw_file.name;raw['scores_sha256']=hash_file(raw_file)
    integrated['score_artifacts']={'decoder_arrays_and_timestamps_exported':bool(settings.get('save_raw_scores') and settings.get('export_decoder_arrays',False)),
                                  'scores_file':raw.get('scores_file'),
                                  'label_granularity':'weak_fragment_label',
                                  'candidate_protocol':'validation only if validation_fraction>0'}
    report=result_dir/f'{identifier}_integrated_{digest[:12]}_{device.replace(":","-")}_{settings_token}.json'
    write_json_atomic(report,integrated)
    update_summary_csv(RESULTS_DIR/'all_experiments.csv')
    print(f'OPS-SAT evaluation: Best F1={bf["bf_f1"]:.4f}, POT F1={pot["pot_f1"]}; report={report}',flush=True)
    return eval_file,report
