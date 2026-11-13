"""Explicit analytic counts, raw timing samples, profiler coverage and telemetry."""
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
from contextlib import contextmanager

import numpy as np
import torch

from tranad_plus.core.common import now,save_npz,write_json


def process_resources(device=None):
    r=resource.getrusage(resource.RUSAGE_SELF)
    result={'timestamp':now(),'pid':os.getpid(),'cpu_user_seconds':r.ru_utime,'cpu_system_seconds':r.ru_stime,
            'process_peak_rss_bytes':int(r.ru_maxrss*(1 if platform.system()=='Darwin' else 1024))}
    if device is not None and torch.device(device).type=='cuda':
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def gpu_snapshot():
    if not shutil.which('nvidia-smi'):return {'status':'not_available','reason':'nvidia-smi not installed'}
    fields='index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu'
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],
                         text=True,capture_output=True,timeout=10,check=True)
        return {'status':'measured','timestamp':now(),'fields':fields.split(','),
                'rows':[line.split(', ') for line in p.stdout.strip().splitlines()],
                'units':{'memory.total':'MiB','memory.used':'MiB','utilization.gpu':'percent','power.draw':'W','temperature.gpu':'C'},
                'scope':'instantaneous GPU snapshot, not integrated experiment energy'}
    except (OSError,subprocess.SubprocessError) as exc:return {'status':'not_available','reason':str(exc)}


def analytic_operations(profile):
    f,w,h,layers=(profile[k] for k in ('features','window_sz','dim_feedforward','num_layers'))
    d=2*f
    encoder=layers*(4*w*d*d+2*w*w*d+2*w*d*h)
    decoder=layers*(4*d*d+2*d+(2+2*w)*d*d+2*w*d+2*d*h)
    head=d*f
    passes=2 if profile['variant'] in ('reference','shared_decoder') else 1
    one=encoder+head if profile['variant']=='encoder_only' else encoder+decoder+head
    macs=passes*one
    return {'matmul_MACs_per_window':int(macs),'matmul_FLOPs_per_window':int(2*macs),
            'encoder_MACs_per_pass':int(encoder),'decoder_MACs_per_pass':0 if profile['variant']=='encoder_only' else int(decoder),
            'head_MACs_per_pass':int(head),'passes':passes,
            'scope':'analytic projections, QK/AV, FFN and output head; excludes biases, activations, softmax, fake-quant, optimizer and backward',
            'precision_does_not_change_MAC_count':True}


@contextmanager
def native_profiler_log(path):
    # Profiler C++ diagnostics bypass redirect_stderr. Keep them out of tmux's
    # three progress rows while retaining them verbatim alongside the result.
    original=os.dup(2)
    try:
        with Path(path).open('a') as log:
            os.dup2(log.fileno(),2)
            yield
    finally:
        os.dup2(original,2);os.close(original)


def operator_profile(model,x,target,device,path):
    try:
        activities=[torch.profiler.ProfilerActivity.CPU]
        if torch.device(device).type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        with native_profiler_log(Path(path).with_suffix('.log')),torch.inference_mode(),torch.profiler.profile(activities=activities,record_shapes=True,
                                                           profile_memory=True,with_flops=True) as prof:
            model(x,target)
            if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
        rows=[]
        for event in prof.key_averages(group_by_input_shape=True):
            rows.append({'operator':event.key,'calls':event.count,'input_shapes':str(event.input_shapes),
                         'FLOPs_recognized':event.flops or 0,'cpu_time_us':event.cpu_time_total,
                         'self_cpu_time_us':event.self_cpu_time_total,
                         'device_time_us':getattr(event,'device_time_total',getattr(event,'cuda_time_total',0)),
                         'cpu_memory_bytes':event.cpu_memory_usage,
                         'device_memory_bytes':getattr(event,'device_memory_usage',getattr(event,'cuda_memory_usage',0))})
        result={'status':'measured','operators':rows,'recognized_FLOPs':sum(r['FLOPs_recognized'] for r in rows),
                'operator_calls':sum(r['calls'] for r in rows),
                'coverage':'partial: PyTorch recognizes FLOPs only for supported operators; fused attention can be omitted',
                'not_training_backward_or_optimizer':True}
    except Exception as exc:
        result={'status':'not_available','reason':f'{type(exc).__name__}: {exc}'}
    write_json(path,result)
    return {k:v for k,v in result.items() if k!='operators'}


def measure(model,dataset,device,settings,execution,profile,folder):
    folder=Path(folder);sample=dataset[0];sample=sample[0] if isinstance(sample,(tuple,list)) else sample
    x=sample.to(device).unsqueeze(1);target=x[-1:];model.eval();cuda=torch.device(device).type=='cuda'
    times=[];cpu_times=[]
    with torch.inference_mode():
        for _ in range(settings['latency_warmup']):model(x,target)
        if cuda:torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        before=process_resources(device)
        for _ in range(settings['latency_iterations']):
            if cuda:torch.cuda.synchronize(device)
            wall=time.perf_counter();cpu=time.process_time();model(x,target)
            if cuda:torch.cuda.synchronize(device)
            times.append(1000*(time.perf_counter()-wall));cpu_times.append(1000*(time.process_time()-cpu))
        after=process_resources(device)
    save_npz(folder/'latency_samples.npz',wall_ms=np.asarray(times),process_cpu_ms=np.asarray(cpu_times))
    operations=analytic_operations(profile)
    profiler=operator_profile(model,x,target,device,folder/'operators.json') if settings.get('profile_operators',True) else {'status':'disabled'}
    return {'mean_ms':float(np.mean(times)),'std_ms':float(np.std(times,ddof=1)) if len(times)>1 else 0.,
            'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
            'min_ms':float(min(times)),'max_ms':float(max(times)),
            'iterations':len(times),'warmup':settings['latency_warmup'],'batch_size':1,'execution':execution,
            'unique_parameters':sum(p.numel() for p in model.parameters()),
            'parameter_bytes_resident':sum(p.numel()*p.element_size() for p in model.parameters()),
            'buffer_bytes_resident':sum(b.numel()*b.element_size() for b in model.buffers()),
            'parameters_by_tensor':[{'name':n,'shape':list(p.shape),'elements':p.numel(),'dtype':str(p.dtype),'bytes':p.numel()*p.element_size()} for n,p in model.named_parameters()],
            'process_before':before,'process_after':after,'analytic_operations':operations,'profiler':profiler,
            'scope':'forward including FP16 wrapper casts and GPU synchronization; excludes data loading and profiler overhead',
            'not_integer_kernel_latency':execution=='dequantized_fp32_simulation'}
