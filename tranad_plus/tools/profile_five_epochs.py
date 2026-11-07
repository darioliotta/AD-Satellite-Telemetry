#!/usr/bin/env python3
"""Faithful-baseline v12 five-epoch PROFILING (never the scientific campaign).

For SMAP/MSL the reference values are read from the provided original
TranADPlus.src.global_config.Config.TranAD; OPS-SAT is a documented TOAST adapter.
Default uses WHOLE train and test sets, complete 21-alpha POT/Best-F1-200/dense
oracle evaluation, and 2 repeats of dataset-specific batch sizes x 4 data paths x 2 schedules.
Everything is timing/diagnostic; no checkpoint, no scientific conclusions.
"""
from __future__ import annotations
import argparse,csv,gc,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.profile_a100 import build_data
from tools.profile_full_metrics import make_eval_data,evaluate_full,sync


def baseline_profile(name):
    from config import PROFILES,get_profile
    from campaign_plan import register_plan_profiles
    from campaign_v12 import BASES
    p=get_profile(BASES[name]['base'])
    if name=='OPS-SAT':
        b=BASES['OPS-SAT']
        return {**p,'window_sz':b['window'],'dim_feedforward':b['feedforward'],
                'batch_sz':b['batch'],'lr':b['lr'],'pot_q':b['pot_q'],
                'profiling_baseline_source':'TOAST OPS-SAT adapter (NO upstream OPS-SAT preset)'}
    from TranADPlus.src.global_config import Config
    original=Config().TranAD[name]
    required=['batch_sz','window_sz','dim_feedforward','lr','pot_q','padding','shuffle','num_workers','num_layers','downsample']
    if original['epochs']!=5 or original['batch_sz']!=8192:
        raise RuntimeError('Original configuration differs from the verified reference; re-audit required')
    return {**p,**{k:original[k] for k in required},
            'profiling_baseline_source':f'ORIGINAL TranADPlus/src/global_config.py TranAD[{name}]',
            'original_epochs':int(original['epochs'])}


def batch_choices(name,reference):
    """Profiling-only sweep: wide OPS-SAT exploration; upstream-centered SMAP/MSL."""
    if name == 'OPS-SAT':
        # No OPS-SAT upstream preset; large batches are worth profiling on A100.
        # Keep the TOAST batch=256 as the reference point, not an upstream claim.
        if int(reference['batch_sz']) != 256:
            raise RuntimeError('OPS-SAT adapter batch changed; re-audit the profiling sweep')
        return [256, 1024, 4096, 8192]
    original=int(reference['batch_sz'])
    return [max(1,original//2),original,original*2]


def prepare_train_batches(data,mode,bs,device,limit):
    """Exact same windows/index order across modes; includes prep cost."""
    import torch
    from torch.utils.data import DataLoader,Subset
    start=time.perf_counter()
    n=len(data) if limit==0 else min(len(data),limit)
    if n<1:raise ValueError('No train windows')
    if mode in ('ram','vram'):
        if isinstance(data,torch.Tensor):
            windows=data[:n].to(dtype=torch.float32).contiguous()
        elif hasattr(data,'windows'):
            windows=data.windows[:n].to(dtype=torch.float32).contiguous()
        else:
            windows=(data.big_track.unfold(0,int(data.window_size),1).permute(0,2,1)[:n]
                     .to(dtype=torch.float32).contiguous())
        if mode=='vram':
            if str(device).startswith('cuda'):
                free,total=torch.cuda.mem_get_info(device)
                need=windows.numel()*4
                budget=min(int(float(os.environ.get('TRANADPLUS_GPU_CACHE_MAX_GIB','16'))*2**30),int(free*.5))
                if need>budget:raise MemoryError(f'VRAM materialization requires {need/2**30:.2f} GiB, budget {budget/2**30:.2f} GiB')
            storage=windows.to(device)
            batches=[storage[i:i+bs] for i in range(0,n,bs)]
        else:
            storage=windows.pin_memory() if str(device).startswith('cuda') else windows
            batches=[storage[i:i+bs] for i in range(0,n,bs)]
    else:
        storage=None
        batches=DataLoader(Subset(data,range(n)),batch_size=bs,shuffle=False,num_workers=0,
                           pin_memory=mode=='pinned' and str(device).startswith('cuda'))
    sync(device)
    return batches,storage,time.perf_counter()-start,n


def create_source(ds,mode,bs,device,limit):
    """Compatibility alias for audited profiling data-path regression tests."""
    return prepare_train_batches(ds,mode,bs,device,limit)

def run_one(profile,data,eval_sources,*,device,bs,mode,schedule,epochs,limit,eval_limit,seed,progress,alpha_count):
    import torch
    from tqdm.auto import tqdm
    from TranADPlus.src.models import TranAD
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    train_batches,storage,prep,n=prepare_train_batches(data,mode,bs,device,limit)
    f=int(data.windows.shape[-1]) if hasattr(data,'windows') else int(data.big_track.shape[-1])
    model=TranAD(n_feats=f,window_sz=profile['window_sz'],batch_sz=bs,
                 dim_feedforward=profile['dim_feedforward'],num_encoder_layers=profile['num_layers'],
                 num_decoder_layers=profile['num_layers']).float().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=profile['lr'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,5,gamma=.9)
    sync(device);overall=time.perf_counter()
    history=[];total_train=total_eval=0.
    iterator=tqdm(range(epochs),desc=f'{profile["dataset"]} {mode} bs{bs} {schedule}',disable=not progress)
    for epoch in iterator:
        model.train();sync(device);start=time.perf_counter()
        for item in train_batches:
            windows=item.to(device,dtype=torch.float32,
                            non_blocking=mode in ('pinned','ram')).permute(1,0,2)
            target=windows[-1:]
            z1,z2=model(windows,target)
            w=1/(epoch+1)
            loss=(w*(z1-target).square()+(1-w)*(z2-target).square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step();sync(device)
        training_elapsed=time.perf_counter()-start
        total_train+=training_elapsed
        item={'epoch':epoch+1,'train_seconds':training_elapsed,'train_last_batch_loss':float(loss.detach().item())}
        if schedule=='each' or epoch==epochs-1:
            model.eval();sync(device);t=time.perf_counter()
            tr,te,groups,inventory=eval_sources
            evaluation=evaluate_full(model,profile,tr,te,groups,mode,bs,device,
                                     max_windows=eval_limit,alpha_count=alpha_count)
            sync(device);evaluation_seconds=time.perf_counter()-t
            total_eval+=evaluation_seconds
            item['eval_seconds']=evaluation_seconds
            item['evaluation']=evaluation
            item['pot_f1_timing_only']=evaluation['metrics']['pot']
            item['bf200_f1_timing_only']=evaluation['metrics']['best_f1_original']
            item['dense_f1_timing_only']=evaluation['metrics']['best_f1_dense_oracle']
        history.append(item)
        if progress:iterator.set_postfix(train_s=f'{total_train:.1f}',eval_s=f'{total_eval:.1f}')
    sync(device);overall_s=time.perf_counter()-overall
    pot_failures=sum(h['evaluation']['metrics']['pot_failed_alpha_count'] for h in history if 'evaluation' in h)
    result={'dataset':profile['dataset'],'mode':mode,'batch_size':bs,'schedule':schedule,
            'baseline_source':profile['profiling_baseline_source'],
            'model_window':profile['window_sz'],'model_feedforward':profile['dim_feedforward'],
            'model_lr':profile['lr'],'pot_q':profile['pot_q'],
            'epochs':epochs,'train_windows':n,'full_training':limit==0 or limit>=len(data),
            'train_batches_per_epoch':len(train_batches),'evaluation_passes':len([r for r in history if 'evaluation' in r]),
            'alphas_per_evaluation':alpha_count,
            'all_test_windows_evaluated':eval_limit==0 or eval_limit>=len(eval_sources[1]),
            'training_seconds':round(total_train,4),'evaluation_seconds':round(total_eval,4),
            'data_preparation_seconds':round(prep,4),
            'total_seconds_including_preparation':round(overall_s+prep,4),
            'gpu_peak_gib':round(torch.cuda.max_memory_allocated(device)/2**30,4) if str(device).startswith('cuda') else None,
            'history':history,'status':'complete' if not pot_failures else 'completed_with_pot_failures',
            'pot_failed_alpha_attempts':int(pot_failures)}
    del model,optimizer,scheduler,train_batches,storage
    gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['OPS-SAT','SMAP','MSL'],required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--max-windows',type=int,default=0,help='0 = ALL training windows; >0 subsets for smoke tests')
    p.add_argument('--max-eval-windows',type=int,default=0,help='0 = ALL official train+test windows; >0 NOT full metric benchmark')
    p.add_argument('--batch-sizes',default='',help='Comma-separated override; defaults OPS-SAT=256,1024,4096,8192 and SMAP/MSL=4096,8192,16384')
    p.add_argument('--modes',default='ordinary,pinned,ram,vram')
    p.add_argument('--schedules',default='each,after')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--alphas',type=int,default=21,help='21 faithfully reproduces full alpha grid; smaller for TEST SMOKE ONLY')
    p.add_argument('--progress',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--cache-max-gib',type=float,default=16.)
    p.add_argument('--continue-on-error',action='store_true',default=True)
    args=p.parse_args(argv)
    if args.epochs<1 or args.max_windows<0 or args.max_eval_windows<0 or args.repeats<1 or args.alphas<1:
        p.error('Invalid epochs/repeats/window limits/alphas')
    if not (0<args.cache_max_gib<=32):p.error('--cache-max-gib must lie within (0,32]')
    os.environ['TRANADPLUS_GPU_CACHE_MAX_GIB']=str(args.cache_max_gib)
    if args.data_root is not None:os.environ['TRANADPLUS_DATA_ROOT']=str(args.data_root.resolve())
    import torch
    from project_paths import OUTPUT_ROOT
    if args.device.startswith('cuda') and not torch.cuda.is_available():p.error('CUDA unavailable')
    profile=baseline_profile(args.dataset)
    bsizes=([int(v) for v in args.batch_sizes.split(',')] if args.batch_sizes else
            batch_choices(args.dataset,profile))
    if any(b<1 for b in bsizes) or len(bsizes)!=len(set(bsizes)):p.error('Invalid batch sizes')
    modes=[x.strip() for x in args.modes.split(',')]
    schedules=[x.strip() for x in args.schedules.split(',')]
    if any(x not in ('ordinary','pinned','ram','vram') for x in modes):p.error('Invalid modes')
    if any(x not in ('each','after') for x in schedules):p.error('Invalid schedules')
    print('[BASELINE]',json.dumps({k:profile[k] for k in ('dataset','window_sz','dim_feedforward','batch_sz','lr','pot_q','profiling_baseline_source')}),flush=True)
    data,train_inventory,data_load_seconds=build_data(args.dataset,profile)
    t=time.perf_counter()
    eval_sources=make_eval_data(args.dataset,profile)
    eval_data_load_seconds=time.perf_counter()-t
    output=args.output or OUTPUT_ROOT/'v12'/'profiling'/'five_epochs_faithful'/f'{args.dataset.lower().replace("-", "")}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'purpose':'runtime profiling only; NEVER scientific metric comparison',
            'dataset':args.dataset,'baseline':{k:profile[k] for k in ('window_sz','dim_feedforward','batch_sz','lr','pot_q','profiling_baseline_source')},
            'device':args.device,'gpu_name':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
            'batch_sizes':bsizes,'modes':modes,'schedules':schedules,
            'epochs':args.epochs,'repeats':args.repeats,'alphas':args.alphas,
            'max_windows':args.max_windows,'max_eval_windows':args.max_eval_windows,
            'train_dataset_initial_load_seconds':data_load_seconds,
            'evaluation_dataset_initial_load_seconds':eval_data_load_seconds,
            'train_inventory':train_inventory,'evaluation_inventory':eval_sources[3],
            'caution':'best-F1 alpha and thresholds use official test labels: oracle. Full metrics are timed, not scientific claims.',
            'runs':[]}
    print(f'[PLAN] distinct={len(bsizes)*len(modes)*len(schedules)}, runs={len(bsizes)*len(modes)*len(schedules)*args.repeats}; train={len(data)} test={len(eval_sources[1])}',flush=True)
    for repeat in range(1,args.repeats+1):
        for bs in bsizes:
            for mode in modes:
                for schedule in schedules:
                    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
                    try:
                        row=run_one(profile,data,eval_sources,device=args.device,bs=bs,mode=mode,
                                    schedule=schedule,epochs=args.epochs,limit=args.max_windows,
                                    eval_limit=args.max_eval_windows,seed=2026,progress=args.progress,
                                    alpha_count=args.alphas)
                    except Exception as exc:
                        row={'dataset':args.dataset,'mode':mode,'batch_size':bs,'schedule':schedule,
                             'status':'error','error':f'{type(exc).__name__}: {exc}',
                             'traceback':traceback.format_exc(limit=6)}
                        gc.collect()
                        if args.device.startswith('cuda'):torch.cuda.empty_cache()
                    row['repeat']=repeat
                    if 'total_seconds_including_preparation' in row:
                        # This is a projection of independent-process startup time;
                        # actual dataset loading is amortized within this sweep.
                        row['total_if_dataset_reloaded_per_run_seconds']=round(
                            row['total_seconds_including_preparation']+data_load_seconds+eval_data_load_seconds,4)
                    report['runs'].append(row)
                    digest={k:v for k,v in row.items() if k not in ('history','traceback')}
                    print('[RESULT]',json.dumps(digest,default=str),flush=True)
                    output.write_text(json.dumps(report,indent=2,default=str)+'\n')
                    headers=['dataset','mode','batch_size','schedule','repeat','status','train_windows',
                             'train_batches_per_epoch','evaluation_passes','training_seconds','evaluation_seconds',
                             'data_preparation_seconds','total_seconds_including_preparation','total_if_dataset_reloaded_per_run_seconds',
                             'pot_failed_alpha_attempts','gpu_peak_gib','error']
                    with output.with_suffix('.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=headers,extrasaction='ignore')
                        writer.writeheader();writer.writerows(report['runs'])
    print('[DONE]',output,flush=True)
    return report

if __name__=='__main__':main()
