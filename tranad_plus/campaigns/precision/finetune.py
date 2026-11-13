"""Matched FP32/QAT trajectories, retained milestones and exact budget extension."""
from pathlib import Path
import json
import time

import numpy as np
import torch

from tranad_plus.core.common import append_json, save_torch, save_npz, atomic_destination,now
from tranad_plus.core.models import build_model
from tranad_plus.core.training.engine import seed_all, rng_state, restore_rng, TrainingInterrupted
from tranad_plus.core.training.objectives import reconstruction_loss
from .quantization import apply_qat, recipes
from .progress import bar, record
from .resources import process_resources,analytic_operations
from gpu_batches import training_batches


def reconcile(folder):
    path, recovery = Path(folder)/'epochs.jsonl', Path(folder)/'checkpoints/resume.ckpt'
    if not path.exists():return
    count = torch.load(recovery, map_location='cpu', weights_only=True)['completed_epochs'] if recovery.exists() else 0
    rows = {}
    for line in path.read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError:continue
        if row['local_epoch']<=count:rows[row['local_epoch']]=row
    if set(rows)!=set(range(1,count+1)):raise RuntimeError('Missing committed fine-tuning telemetry')
    with atomic_destination(path) as stream:
        for epoch in sorted(rows):stream.write(json.dumps(rows[epoch])+'\n')


def fit(profile, parent_state, bundle, folder, signature, device, settings, bits=None, stop_after_epoch=None,target_epoch=None):
    folder=Path(folder); final=folder/'checkpoints/final.ckpt'; recovery=folder/'checkpoints/resume.ckpt'
    seed_all(profile['seed'])
    model=build_model(profile)
    model.load_state_dict(parent_state, strict=True)
    scales=recipes(model,bits) if bits else {}
    if bits:apply_qat(model,scales,bits)
    model.to(device)
    total=settings['epochs']
    target=target_epoch or total
    if not 1<=target<=total:raise ValueError('Requested fine-tuning budget outside declared horizon')
    milestones=set(settings.get('milestones',[total]))|{target}
    optimizer=torch.optim.Adam(model.parameters(),lr=profile['lr']*settings['lr_multiplier'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,settings['step_size'],gamma=settings['gamma'])
    start=0
    if final.exists() or recovery.exists():
        saved=torch.load(final if final.exists() else recovery,map_location='cpu',weights_only=True)
        if saved['signature']!=signature or saved['profile']!=profile or saved['scales']!=scales or saved['bits']!=bits:
            raise RuntimeError('Fine-tuning recovery differs in source/profile/quantizer')
        model.load_state_dict(saved['model_state'],strict=True)
        optimizer.load_state_dict(saved['optimizer_state']);scheduler.load_state_dict(saved['scheduler_state'])
        restore_rng(saved['rng_state']);start=saved['completed_epochs']
        if scheduler.last_epoch!=start:raise RuntimeError('Fine-tuning scheduler/epoch mismatch')
        reconcile(folder)
        if final.exists() and start!=total:raise RuntimeError('Incomplete fine-tuning final checkpoint')
        if start in milestones and not (folder/f'milestones/epoch_{start:03d}/checkpoint.ckpt').exists():
            save_torch(folder/f'milestones/epoch_{start:03d}/checkpoint.ckpt',{**saved,'kind':'milestone'})
        if start>=target:
            if start==total and not final.exists():save_torch(final,{**saved,'kind':'final'})
            return model
    else:reconcile(folder)
    batches,transfer=training_batches(bundle.datasets['train'],batch_size=profile['batch_sz'],shuffle=False,device=device,num_workers=0)
    label='QAT INT'+str(bits) if bits else 'FP32 CONTROL'
    payload=saved if start else None
    times=[];cuda=torch.device(device).type=='cuda'
    with bar(total=target,initial=start,desc=f'{label} | epochs',position=1,leave=False,unit='epoch') as epochs:
        for epoch in range(start,target):
            if cuda:torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            begun=time.monotonic();cpu_begun=time.process_time();model.train();loss_sum=torch.zeros((),device=device,dtype=torch.float64);count=0
            batch_losses=[];batch_sizes=[]
            absolute=profile['epochs']+epoch+1
            with bar(batches,total=len(batches),desc=f'{label} | ep {epoch+1} abs {absolute}',
                     position=2,leave=False,unit='batch') as loop:
                for batch in loop:
                    x=batch[0] if isinstance(batch,(tuple,list)) else batch
                    x=x.to(device,non_blocking=transfer=='cpu_pinned').permute(1,0,2);element=x[-1:]
                    loss=reconstruction_loss(model(x,element),element,absolute,
                         implementation='square' if profile['dataset']=='OPS-SAT' else 'mse_loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                    loss_sum.add_(loss.detach().double());count+=1
                    batch_losses.append(loss.detach());batch_sizes.append(x.shape[1])
            if not count:raise RuntimeError('No fine-tuning batches')
            value=float((loss_sum/count).item())
            if not np.isfinite(value):raise FloatingPointError('Nonfinite fine-tuning loss')
            scheduler.step()
            if cuda:torch.cuda.synchronize(device)
            seconds=time.monotonic()-begun;times.append(seconds)
            save_npz(folder/f'batch_losses/epoch_{epoch+1:03d}.npz',
                     loss=torch.stack(batch_losses).cpu().numpy(),windows=np.asarray(batch_sizes,dtype=np.int32))
            row={'local_epoch':epoch+1,'absolute_epoch':absolute,'loss':value,'seconds':seconds,
                 'cpu_process_seconds':time.process_time()-cpu_begun,
                 'lr':optimizer.param_groups[0]['lr'],'batches':count,'training_windows':len(bundle.datasets['train']),
                 'architecture_forward_MACs':analytic_operations(profile)['matmul_MACs_per_window']*len(bundle.datasets['train']),
                 **process_resources(device)}
            append_json(folder/'epochs.jsonl',row)
            payload={'schema_version':1,'kind':'resume','signature':signature,'profile':profile,'bits':bits,
                     'completed_epochs':epoch+1,'scales':scales,'model_state':model.state_dict(),
                     'optimizer_state':optimizer.state_dict(),'scheduler_state':scheduler.state_dict(),'rng_state':rng_state()}
            io_start=time.monotonic();save_torch(recovery,payload)
            snapshot=None
            if epoch+1 in milestones or (stop_after_epoch and epoch+1>=stop_after_epoch):
                snapshot=folder/f'milestones/epoch_{epoch+1:03d}/checkpoint.ckpt'
                save_torch(snapshot,{**payload,'kind':'milestone'})
            append_json(folder/'checkpoint_io.jsonl',{'epoch':epoch+1,'timestamp':now(),
                        'seconds':time.monotonic()-io_start,'resume_file_bytes':recovery.stat().st_size,
                        'milestone_file_bytes':snapshot.stat().st_size if snapshot else None})
            epochs.set_postfix(L=f'{value:.4g}',s=f'{seconds:.1f}',refresh=False);epochs.update(1)
            record(folder,'fine_tuning',kind=label,epoch=epoch+1,epochs=target,maximum_epochs=total,absolute_epoch=absolute,loss=value,
                   remaining_seconds_estimate=sum(times[-5:])/len(times[-5:])*(target-epoch-1))
            if stop_after_epoch and epoch+1>=stop_after_epoch:raise TrainingInterrupted('Fine-tuning checkpoint saved')
    if target==total:
        payload['kind']='final';save_torch(final,payload)
    return model
