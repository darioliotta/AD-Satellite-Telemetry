#!/usr/bin/env python3
"""Read-only, real TEST-data GPU evaluation throughput profiling for TOAST v12.

Measures forward + computation of scores for 21 alpha combinations, not F1/POT,
threshold search, checkpoint I/O, nor full campaign wall-clock. Uses a freshly
initialized baseline model to isolate runtime cost; never claims detection quality.
No writes except the selected JSON profiling report. Does not modify the dataset.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def make_test_dataset(dataset, profile):
    from torch.utils.data import Dataset
    from opssat_pipeline import prepare_data, FragmentWindows
    if dataset=='OPS-SAT':
        _,test,scalers,inventory=prepare_data(profile)
        return FragmentWindows(test,scalers,profile['window_sz'],False), {
            'test_fragments':inventory['n_test_fragments'],
            'test_points':inventory['n_test_points'],
            'source_sha256':inventory['source_sha256']}
    from config import PROCESSED_DATA_DIR
    from TranADPlus.src.datasets import TSDataset_tracks
    train_files=sorted(str(x) for x in (PROCESSED_DATA_DIR/dataset/'Train'/'Tracks').glob('*.npy'))
    test_files=sorted(str(x) for x in (PROCESSED_DATA_DIR/dataset/'Test'/'Tracks').glob('*.npy'))
    if not train_files or not test_files:
        raise FileNotFoundError(f'Missing processed tracks for {dataset} in {PROCESSED_DATA_DIR}')
    train=TSDataset_tracks(window_size=profile['window_sz'],track_paths=train_files,
                         padding=profile['padding'],scaling=True,
                         downsample=profile.get('downsample'))
    test=TSDataset_tracks(window_size=profile['window_sz'],track_paths=test_files,
                        padding=profile['padding'],scaling=True,scaler=train.scaler,
                        downsample=profile.get('downsample'))
    return test,{'test_tracks':len(test_files),'train_tracks_for_scaler':len(train_files)}


def alpha_scores(z1,z2,target, *, alphas):
    """Score all alpha blends using the ORIGINAL blend-then-square definition.

    z1,z2,target: (1,B,F). output (A,B). No labels or threshold selection.
    """
    # Broadcast across alpha dimension without rerunning the Transformer.
    a=alphas.reshape(-1,1,1,1)
    mixed=a*z1.unsqueeze(0)+(1-a)*z2.unsqueeze(0)
    return (mixed-target.unsqueeze(0)).square().mean(dim=(1,3))


def data_iterator(dataset, mode, batch_size, device):
    import torch
    from torch.utils.data import DataLoader
    if mode=='vram':
        from gpu_batches import maybe_cuda_batches
        resident=maybe_cuda_batches(dataset,batch_size=batch_size,shuffle=False,device=device)
        if resident is None:
            raise RuntimeError('VRAM caching ineligible (check storage dtype, layout and memory budget)')
        return resident
    return DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,
                      pin_memory=(mode=='pinned'))


def forward_one(model,batch,device,alphas,*,non_blocking):
    import torch
    if not isinstance(batch,torch.Tensor):
        raise TypeError(f'Expected tensor-only dataset, received {type(batch).__name__}')
    x=batch.to(device,non_blocking=non_blocking).permute(1,0,2)
    target=x[-1:,:,:]
    z1,z2=model(x,target)
    return alpha_scores(z1,z2,target,alphas=alphas)


def bench(model,dataset,*,mode,batch_size,device,warmup,steps,alphas):
    import torch
    batches=data_iterator(dataset,mode,batch_size,device)
    if len(batches)==0: raise ValueError('Empty test dataset')
    model.eval()
    iteration=iter(batches)
    windows=0
    with torch.inference_mode():
        # Do a fixed number of steps for a fair speed comparison; wrap around.
        for i in range(warmup+steps):
            try: batch=next(iteration)
            except StopIteration:
                iteration=iter(batches)
                batch=next(iteration)
            if i==warmup:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start=time.perf_counter()
            scores=forward_one(model,batch,device,alphas,non_blocking=(mode=='pinned'))
            if i>=warmup: windows+=batch.shape[0]
        # Ensure CUDA work actually completed before stopping clock.
        torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
    out={'mode':mode,'steps':steps,'windows':windows,'elapsed_seconds':round(elapsed,6),
         'milliseconds_per_step':round(1000*elapsed/steps,4),
         'windows_per_second':round(windows/elapsed,2),
         'max_allocated_gib':round(torch.cuda.max_memory_allocated(device)/2**30,4),
         'score_shape_last_batch':list(scores.shape)}
    del batches,scores
    return out


def bench_alpha_recompute(model,dataset,*,mode,batch_size,device,warmup,steps):
    """Reference overhead: 21 separate forwards, one per alpha (not a recommended evaluation path)."""
    import torch
    batches=data_iterator(dataset,mode,batch_size,device)
    iterator=iter(batches)
    alphas=torch.linspace(0,1,21,device=device)
    model.eval()
    windows=0
    with torch.inference_mode():
        for i in range(warmup+steps):
            try:batch=next(iterator)
            except StopIteration:
                iterator=iter(batches);batch=next(iterator)
            if i==warmup:
                torch.cuda.synchronize(device);start=time.perf_counter()
            x=batch.to(device,non_blocking=mode=='pinned').permute(1,0,2)
            t=x[-1:,:,:]
            for alpha in alphas:
                z1,z2=model(x,t)
                _=((alpha*z1+(1-alpha)*z2-t).square().mean())
            if i>=warmup:windows+=batch.shape[0]
        torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
    return {'mode':'21_forwards_reference','steps':steps,'windows':windows,
            'elapsed_seconds':round(elapsed,6),'windows_per_second':round(windows/elapsed,2)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['OPS-SAT','SMAP','MSL'],required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--warmup',type=int,default=4)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--cache-max-gib',type=float,default=16)
    p.add_argument('--compare-recompute',action='store_true',help='Optional 21 full-forward reference (slow)')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    if args.warmup<0 or args.steps<1 or (args.batch_size is not None and args.batch_size<1):p.error('Invalid steps or batch size')
    if not (0<args.cache_max_gib<=32):p.error('cache max must be in (0,32]')
    import torch
    if not args.device.startswith('cuda') or not torch.cuda.is_available():p.error('CUDA GPU required for evaluation profiler')
    os.environ['TRANADPLUS_GPU_CACHE_MAX_GIB']=str(args.cache_max_gib)
    from TranADPlus.src.models import TranAD
    from project_paths import OUTPUT_ROOT
    from tools.profile_five_epochs import baseline_profile
    profile=baseline_profile(args.dataset)
    profile={**profile,'batch_sz':args.batch_size or profile['batch_sz']}
    started=time.perf_counter()
    dataset,inventory=make_test_dataset(args.dataset,profile)
    preparation=time.perf_counter()-started
    # Infer feature count from the actual loaded test tensors, with no dependency on Both/Tracks.
    f=int(dataset.windows.shape[-1]) if hasattr(dataset,'windows') else int(dataset.big_track.shape[-1])
    torch.manual_seed(2026)
    model=TranAD(n_feats=f,window_sz=profile['window_sz'],batch_sz=profile['batch_sz'],
                 dim_feedforward=profile['dim_feedforward'],num_encoder_layers=profile['num_layers'],
                 num_decoder_layers=profile['num_layers']).float().to(args.device).eval()
    print(f'[EVAL PROFILE] {args.dataset} | baseline W={profile["window_sz"]}, FF={profile["dim_feedforward"]}, batch={profile["batch_sz"]} | {torch.cuda.get_device_name(args.device)}',flush=True)
    alphas=torch.linspace(0,1,21,device=args.device)
    out={'dataset':args.dataset,'device':args.device,'gpu':torch.cuda.get_device_name(args.device),
         'precision':'FP32 (default); no AMP/TF32 enabled by this script',
         'model':'fresh randomly initialized baseline, runtime comparison ONLY',
         'test_samples':len(dataset),'test_inventory':inventory,'preparation_seconds':round(preparation,4),
         'window':profile['window_sz'],'feedforward':profile['dim_feedforward'],
         'batch_size':profile['batch_sz'],'alphas':21,'warmup':args.warmup,'steps':args.steps,
         'scope':'forward + compute 21 anomaly scores on TEST windows; excludes F1, POT, threshold search, I/O/reporting',
         'results':[]}
    for mode in ('ordinary','pinned','vram'):
        try:
            measurement=bench(model,dataset,mode=mode,batch_size=profile['batch_sz'],device=args.device,
                              warmup=args.warmup,steps=args.steps,alphas=alphas)
        except (RuntimeError,ValueError,TypeError) as e:
            measurement={'mode':mode,'error':str(e)}
        out['results'].append(measurement)
        print('[EVAL PROFILE]',measurement,flush=True)
    if args.compare_recompute:
        try:
            rec=bench_alpha_recompute(model,dataset,mode='pinned',batch_size=profile['batch_sz'],
                                      device=args.device,warmup=min(1,args.warmup),steps=max(1,min(3,args.steps)))
        except (RuntimeError,ValueError) as e:rec={'error':str(e)}
        out['recompute_reference']=rec
        print('[EVAL PROFILE][21 forwards]',rec,flush=True)
    destination=args.output or OUTPUT_ROOT/'v12'/'profiling'/f'eval_{args.dataset.lower().replace("-", "")}.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(out,indent=2)+'\n')
    print('[EVAL PROFILE] Report:',destination,flush=True)
    print('[EVAL PROFILE] Profiling only: no training, checkpoints, threshold fitting or metric selection.',flush=True)
    return out

if __name__=='__main__':main()
