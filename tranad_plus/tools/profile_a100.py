#!/usr/bin/env python3
"""Non-destructive real-data TRAIN-STEP microbenchmark for TOAST v12.

Uses training-only windows, temporary model/optimizer (no checkpoint writes),
fixed seed and a specified number of updates. Outputs per-mode throughput,
peak CUDA memory, and data preparation time. Does not assess accuracy or
promise overall 144-run wall-clock speedup. Run when A100 is idle.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_data(dataset, profile):
    import torch
    start=time.perf_counter()
    if dataset=='OPS-SAT':
        from opssat_pipeline import prepare_data, FragmentWindows
        train, _, scalers, inventory = prepare_data(profile)
        obj=FragmentWindows(train,scalers,profile['window_sz'],False)
        info={'train_fragments':inventory['n_train_fragments'], 'train_points':inventory['n_train_points'],
              'validation_fragments':inventory['n_validation_fragments'],
              'test_fragments':inventory['n_test_fragments']}
    else:
        from TranADPlus.src.datasets import TSDataset_tracks
        from config import PROCESSED_DATA_DIR
        folder=PROCESSED_DATA_DIR/dataset/'Train'/'Tracks'
        paths=sorted(str(f) for f in folder.glob('*.npy'))
        if not paths:raise FileNotFoundError(f'No training tracks: {folder}')
        obj=TSDataset_tracks(window_size=profile['window_sz'],track_paths=paths,
                             padding=profile['padding'],scaling=True,
                             downsample=profile.get('downsample'))
        info={'train_tracks':len(paths)}
    return obj,info,time.perf_counter()-start


def benchmark(profile, data, *, device, cache, fused, warmup, steps, pinned=False):
    import torch
    from torch.utils.data import DataLoader
    from gpu_batches import maybe_cuda_batches
    from TranADPlus.src.models import TranAD
    from gpu_settings import make_adam
    torch.manual_seed(2026)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(2026)
    f=int(data.windows.shape[-1]) if hasattr(data,'windows') else int(data.big_track.shape[-1])
    model=TranAD(n_feats=f,window_sz=profile['window_sz'],batch_sz=profile['batch_sz'],
                 dim_feedforward=profile['dim_feedforward'],num_encoder_layers=profile['num_layers'],
                 num_decoder_layers=profile['num_layers']).float().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=profile['lr'],**({'fused':True} if fused else {}))
    loader=DataLoader(data,batch_size=profile['batch_sz'],shuffle=False,num_workers=0,
                      pin_memory=pinned and device.startswith('cuda'))
    resident=None
    if cache:
        resident=maybe_cuda_batches(data,batch_size=profile['batch_sz'],shuffle=False,device=device)
    if cache and resident is None:
        return {'mode': 'resident_'+('fused' if fused else 'standard'), 'skipped':'not eligible for resident GPU caching'}
    source=resident if resident is not None else loader
    total=warmup+steps
    iterator=iter(source)
    elapsed_start=None
    measured_windows=0
    if device.startswith('cuda'):torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    for index in range(total):
        try: batch=next(iterator)
        except StopIteration:
            iterator=iter(source);batch=next(iterator)
        if index==warmup:
            if device.startswith('cuda'):
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            elapsed_start=time.perf_counter()
        windows=batch.to(device,non_blocking=pinned).permute(1,0,2)
        target=windows[-1:,:,:]
        z1,z2=model(windows,target)
        loss=(0.5*(z1-target).square()+0.5*(z2-target).square()).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        if index>=warmup:measured_windows+=windows.size(1)
    if device.startswith('cuda'):torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-elapsed_start
    out={'mode':('resident' if cache else ('pinned' if pinned else 'ordinary'))+'_'+('fused' if fused else 'standard'),
         'cache_used':resident is not None, 'optimizer':'fused' if fused else 'standard',
         'measured_steps':steps,'measured_windows':measured_windows,
         'batch_size':profile['batch_sz'],
         'elapsed_seconds':round(elapsed,5), 'ms_per_step':round(1000*elapsed/steps,5),
         'windows_per_second':round(measured_windows/elapsed,2),
         'cuda_peak_allocated_gib':round(torch.cuda.max_memory_allocated(device)/2**30,4) if device.startswith('cuda') else None}
    del model,optimizer,loader,resident
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('OPS-SAT','SMAP','MSL'),required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--data-root',type=Path,help='Override ../data (only for this profiler)')
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--output',type=Path)
    p.add_argument('--gpu-cache-max-gib',type=float,default=16.0)
    p.add_argument('--profile',default='baseline',choices=('baseline',))
    p.add_argument('--precision',choices=('faithful','tf32'),default='faithful')
    p.add_argument('--batch-sizes',help='Optional comma-separated batch-size sweep, e.g. 256,512,1024,2048; profiling only')
    args=p.parse_args()
    if args.warmup<0 or args.steps<1:p.error('warmup >= 0 and steps >= 1')
    if not 0 < args.gpu_cache_max_gib <= 32:p.error('GPU cache max must be in (0,32]')
    if args.data_root is not None:
        os.environ['TRANADPLUS_DATA_ROOT']=str(args.data_root.expanduser().resolve())
    import torch
    from campaign_v12 import BASES
    from config import PROFILES,get_profile
    from project_paths import OUTPUT_ROOT
    os.environ['TRANADPLUS_GPU_CACHE_MAX_GIB']=str(args.gpu_cache_max_gib)
    os.environ['TRANADPLUS_CUDA_PRECISION']=args.precision
    from gpu_settings import configure
    configure()
    from campaign_plan import register_plan_profiles
    register_plan_profiles(PROFILES, ROOT/'experiment_plan_v12.json')
    from tools.profile_five_epochs import baseline_profile
    profile=baseline_profile(args.dataset)
    identity=f'upstream_reference_{args.dataset}'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu for a functional test (not A100 performance)')
    print(f'[PROFILE] {args.dataset} | {identity} | batch={profile["batch_sz"]} | device={args.device}',flush=True)
    data,inventory,load_time=build_data(args.dataset,profile)
    report={'dataset':args.dataset,'identifier':identity,'device':args.device,
            'gpu':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
            'torch':torch.__version__,'precision':args.precision,'batch_size':profile['batch_sz'],
            'window_size':profile['window_sz'],'feedforward':profile['dim_feedforward'],
            'data_preparation_seconds':round(load_time,3),'dataset_windows':len(data),
            'split_inventory':inventory,'warmup_steps':args.warmup,'results':[]}
    modes=[(False,False,False),(False,False,True),(True,False,False)]
    if args.device.startswith('cuda'):modes.append((True,True,False))
    for resident,fused,pinned in modes:
        try:result=benchmark(profile,data,device=args.device,cache=resident,
                             fused=fused,warmup=args.warmup,steps=args.steps,pinned=pinned)
        except (RuntimeError,ValueError) as exc:
            result={'mode':('resident' if resident else ('pinned' if pinned else 'ordinary'))+'_'+('fused' if fused else 'standard'),
                    'error':str(exc)}
        report['results'].append(result)
        print(f'[PROFILE] {result}',flush=True)
    if args.batch_sizes:
        try:
            sizes=[int(x.strip()) for x in args.batch_sizes.split(',')]
        except ValueError:
            p.error('--batch-sizes must be comma-separated positive integers')
        if not sizes or any(n<1 for n in sizes) or len(set(sizes))!=len(sizes):
            p.error('--batch-sizes must be unique positive integers')
        report['batch_sweep']=[]
        for size in sizes:
            altered={**profile,'batch_sz':size}
            try:
                item=benchmark(altered,data,device=args.device,cache=args.device.startswith('cuda'),
                               fused=False,warmup=args.warmup,steps=args.steps)
            except RuntimeError as exc:
                item={'batch_size':size,'error':str(exc)}
            report['batch_sweep'].append(item)
            print(f'[PROFILE][BATCH {size}] {item}',flush=True)
    destination=args.output or OUTPUT_ROOT/'v12'/'profiling'/f'{args.dataset.lower().replace("-", "")}_baseline.json' 
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(f'[PROFILE] Wrote: {destination}',flush=True)
    print('[PROFILE] This is a training-step throughput measurement, NOT an inference latency or training accuracy comparison.',flush=True)

if __name__=='__main__':main()
