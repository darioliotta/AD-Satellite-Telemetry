"""V17: true integer GEMMs on all four architectures and all three datasets."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from tranad_plus.core.bootstrap import PROJECT_ROOT
from tranad_plus.core.common import digest, hash_file, now, write_json, run_lock, source_manifest
from .recovery import batch_policy, evaluation_settings, inspect_existing, rebind_incomplete


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path(__file__).with_name('plan.json'))
    p.add_argument('--parent-root',type=Path,default=PROJECT_ROOT/'results/v13/architectures')
    p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'results/v17/integer')
    p.add_argument('--data-root',type=Path,default=PROJECT_ROOT/'data')
    p.add_argument('--dataset',choices=('OPS-SAT','SMAP','MSL'),action='append')
    p.add_argument('--variant',choices=('reference','shared_decoder','single_pass','encoder_only'),action='append')
    p.add_argument('--configuration',choices=('reference','compact','window_only','ff_only'),action='append')
    p.add_argument('--seed',type=int,action='append')
    p.add_argument('--only',help='Exact job path from --dry-run')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-threads',type=int,default=8)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--check',action='store_true')
    p.add_argument('--status',action='store_true',help='Show saved progress without loading torch/CUDA')
    p.add_argument('--verify',action='store_true',help='Verify all saved artifact hashes without CUDA')
    p.add_argument('--no-progress',action='store_true')
    args=p.parse_args(argv)
    if args.cpu_threads<1:p.error('--cpu-threads must be positive')
    return args


def plan(spec,args):
    if spec.get('name')!='v17_integer' or spec.get('schema_version')!=1:raise ValueError('Unknown plan')
    if spec['formats']!=['int8','int4']:raise ValueError('Only INT8 and INT4; FP32 is the paired reference')
    for key, expected in [('datasets',{'OPS-SAT','SMAP','MSL'}),('variants',{'reference','shared_decoder','single_pass','encoder_only'}),('configurations',{'reference','compact','window_only','ff_only'})]:
        if set(spec[key])!=expected or len(spec[key])!=len(expected):raise ValueError('Incomplete/duplicate '+key)
    evaluation=spec['evaluation']
    if evaluation['alpha_steps']!=21 or evaluation['paper_threshold_steps']!=200:raise ValueError('Use paper BF200 and alpha grids')
    if evaluation['save_pointwise_arrays'] or evaluation['save_reconstructions']:raise ValueError('Use compact results')
    if spec['quantization']['floating_gemm_fallback']:raise ValueError('Floating-point GEMM fallback forbidden')
    if not spec['seeds'] or len(set(spec['seeds']))!=len(spec['seeds']):raise ValueError('Duplicate/no seed')
    if args.seed and not set(args.seed)<=set(spec['seeds']):raise ValueError('Unknown parent seed')
    jobs=[str(Path(ds)/v/c/f'seed_{s}') for ds in spec['datasets'] for c in spec['configurations']
          for v in spec['variants'] for s in spec['seeds']
          if (not args.dataset or ds in args.dataset) and (not args.variant or v in args.variant)
          and (not args.configuration or c in args.configuration) and (not args.seed or s in args.seed)]
    jobs=[j for j in jobs if not args.only or args.only==j]
    if not jobs:raise ValueError('No selected jobs')
    return jobs


def validate_output(args):
    out=args.output_root.expanduser().resolve()
    protected=[args.data_root.resolve(),args.parent_root.resolve(),PROJECT_ROOT/'tranad_plus',PROJECT_ROOT/'tesi',PROJECT_ROOT/'docs']
    protected += [PROJECT_ROOT/'results'/v for v in ('legacy','v10','v12','v13','v14','v15','v16')]
    if any(out==p or p in out.parents or out in p.parents for p in protected):raise ValueError('Output overlaps protected data/code/results')
    return out


def verified_complete(folder,signature):
    status=folder/'status.json'
    if not status.exists():return False
    r=json.loads(status.read_text())
    if r['signature']!=signature:raise RuntimeError('Existing experiment has different code/data/plan/hardware')
    if r['status']!='complete':return False
    for name, expected in r['artifacts'].items():
        if hash_file(folder/name)!=expected:raise RuntimeError('Changed result artifact: '+name)
    return True


def main(argv=None):
    args=arguments(argv);spec=json.loads(args.plan.read_text());jobs=plan(spec,args);out=validate_output(args)
    print(f"V17 INTEGER | {len(jobs)} parent models | {2*len(jobs)} integer evaluations + {len(jobs)} paired FP32 references",flush=True)
    print('No training | BF200/POT + PA + AUC-PA | no FP16 | W4A4 uses INT8 integer instructions, not native INT4.',flush=True)
    print('Resume enabled: completed models are hash-verified and skipped.',flush=True)
    if args.status or args.verify:
        complete, missing = [], []
        for job in jobs:
            path=out/job/'status.json'
            if not path.exists():missing.append(job);continue
            state=json.loads(path.read_text())
            if state['status']!='complete':missing.append(job);continue
            if args.verify:verified_complete(path.parent,state['signature'])
            complete.append(job)
        print(f'Completed: {len(complete)}/{len(jobs)} | pending/interrupted: {len(missing)}')
        campaign=out/'campaign.json'
        if campaign.exists():
            saved=json.loads(campaign.read_text());print('Campaign status: '+saved['status'])
            current=saved.get('current_job');print('Last/current model: '+str(current))
            progress=out/current/'progress.json' if current else None
            if progress and progress.exists():print(progress.read_text())
        if args.verify and missing:
            print('Missing configurations: '+', '.join(missing));return 1
        return 0
    if args.dry_run:
        for i,job in enumerate(jobs,1):print(f'{i:03d}/{len(jobs):03d} {job}')
        return 0
    os.environ.update(TRANADPLUS_CUDA_PRECISION='faithful',TRANADPLUS_ADAM_IMPL='standard',TRANADPLUS_GPU_CACHE='auto',
                      TRANADPLUS_DATA_ROOT=str(args.data_root.resolve()),V16_NO_PROGRESS='1' if args.no_progress else '0',TOAST_NO_TQDM='1',TOAST_FORCE_TQDM='0')
    from tranad_plus.core.bootstrap import activate_baseline
    activate_baseline()
    import numpy as np
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from gpu_settings import configure
    from tranad_plus.core.data import inventory, load_data
    from tranad_plus.core.models import build_model
    from tranad_plus.core.training.engine import load_checkpoint
    from tranad_plus.campaigns.exploration.run import read_parent
    from tranad_plus.campaigns.precision.quantization import export_model
    from tranad_plus.campaigns.precision.evaluation import evaluate
    from tranad_plus.campaigns.precision.progress import bar, announce, logged, record
    from tranad_plus.campaigns.precision.resources import gpu_snapshot, process_resources
    from tranad_plus.campaigns.precision.storage import check_output_budget
    from .kernel import preflight, kernel_audit
    from .runtime import convert, export, load
    from .summarize import summarize
    class NoFloatGemm(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if any(op in str(func) for op in ('aten.mm.','aten.bmm.','aten.addmm.')):
                if any(isinstance(a,torch.Tensor) and a.is_floating_point() for a in args):
                    raise RuntimeError('Floating-point matrix product in integer inference: '+str(func))
            return func(*args,**(kwargs or {}))
    torch.set_num_threads(args.cpu_threads);device=torch.device(args.device)
    if device.type!='cuda':raise ValueError('Use a CUDA device; the CPU backend is for unit tests only')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; no CPU or floating-point fallback is permitted')
    torch.cuda.set_device(device)
    print('PREFLIGHT | compile integer kernels, exact arithmetic checks and PTX instruction inspection',flush=True)
    hardware=preflight(device)
    # Full architecture tests on device before any data campaign starts.
    import unittest
    from .test_integer import IntegerTests
    suite=unittest.TestSuite([IntegerTests('test_cuda_exact_and_all_architectures')])
    if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():raise RuntimeError('CUDA architecture preflight failed')
    configure()
    sources=source_manifest()
    # Pin the complete current source tree used by these campaign tools.
    code={str(p.relative_to(PROJECT_ROOT/'tranad_plus')):hash_file(p)
          for p in sorted((PROJECT_ROOT/'tranad_plus').rglob('*.py')) if '__pycache__' not in p.parts}
    execution={'torch':str(torch.__version__),'python':platform.python_version(),'cuda':torch.version.cuda,
               'device':str(device),'gpu':torch.cuda.get_device_name(device),'triton':hardware['triton_version'],
               'capability':list(torch.cuda.get_device_capability(device)),'cpu_threads':args.cpu_threads,'TF32':False}
    campaign={'name':spec['name'],'started':now(),'status':'running','execution':execution,'jobs':[],
              'recovery_revision':'fp32_parent_batch_v1',
              'evaluation_policy_note':'Completed jobs retain original source signatures and score batches. Pending jobs use the parent FP32 batch and unchanged integer batches.'}
    if not args.check:
        out.mkdir(parents=True,exist_ok=True);write_json(out/'kernel_preflight.json',hardware)
    bundles={}
    with bar(jobs,desc='V17 INTEGER | models',position=0,unit='model') as progress:
      for index, job in enumerate(progress,1):
        folder=out/job;parent=args.parent_root.resolve()/job;started=time.perf_counter()
        item={'id':job,'index':index,'status':'running'};campaign['jobs'].append(item);campaign['current_job']=job
        try:
            profile=json.loads((parent/'config.json').read_text())
            if tuple(Path(job).parts)!=(profile['dataset'],profile['variant'],profile['configuration'],f"seed_{profile['seed']}"):
                raise RuntimeError('Parent path/profile mismatch')
            files=inventory(profile,args.data_root);old,manifest=read_parent(parent,sources,files)
            saved=load_checkpoint(parent/'checkpoints/final.ckpt',profile,old['signature'],final=True)
            binding={'profile':profile,'files':files,'sources':code,'relocation_sha256':hash_file(PROJECT_ROOT/'tranad_plus/relocation.json'),
                     'plan':spec,'execution':execution,'parent_signature':old['signature'],'parent_artifacts':old['artifacts'],
                     'evaluation_policy':batch_policy(spec,profile,old)}
            signature=digest(binding)
            if args.check:
                decision=inspect_existing(folder,binding)
                item.update(status='checked',recovery=decision);continue
            with run_lock(folder):
                path=folder/'manifest.json'
                decision=inspect_existing(folder,binding)
                if decision['action'] in ('skip_complete','preserve_complete'):
                    item.update(status='skipped_verified_complete',recovery=decision)
                    announce(f"[{index:03d}/{len(jobs):03d}] VERIFIED / SKIP | {job}")
                    continue
                if decision['action']=='rebind_incomplete':
                    rebind_incomplete(folder,binding)
                    item['recovery']=decision
                    announce('    RECOVER | previous incomplete attempt archived; completed jobs remain unchanged')
                elif decision['action']=='finish_rebind_status':
                    write_json(folder/'status.json',{'status':'running','signature':signature,'updated':now()})
                    item['recovery']=decision
                check_output_budget(out,spec['storage']['max_output_gib'])
                write_json(out/'campaign.json',campaign)
                if not path.exists():write_json(path,{'signature':signature,'binding':binding,'created':now()})
                write_json(folder/'config.json',profile);write_json(folder/'status.json',{'status':'running','signature':signature,'updated':now()})
                announce(f"\n[{index:03d}/{len(jobs):03d}] {job} | W={profile['window_sz']} FF={profile['dim_feedforward']} | FP32 -> W8A8 -> W4A4")
                # W/FF do not change the scaler or official split; retain one bundle per dataset/window.
                cachekey=(profile['dataset'],profile['window_sz'])
                if cachekey not in bundles:
                    bundles.clear();bundles[cachekey]=load_data(profile,args.data_root,files)
                bundle=bundles[cachekey]
                with np.load(parent/'scores.npz',allow_pickle=False) as f:
                    scores=f['test_scores'].copy();np.testing.assert_array_equal(f['test_labels'],bundle.labels['test'])
                formats={}
                with logged(folder):
                  for name in ['fp32']+spec['formats']:
                    destination=folder/'evaluation'/name;packed=folder/'models'/(name+'.npz')
                    record(folder,'format',format=name,model=index,models_total=len(jobs))
                    existing=destination/'result.json'
                    if existing.exists():
                        value=json.loads(existing.read_text());storage=value['storage']
                        expected_path=packed if name!='fp32' else parent/'checkpoints/final.ckpt'
                        if hash_file(expected_path)!=storage['packed_model_sha256']:raise RuntimeError('Changed source/exported model')
                        for artifact,expected in value['artifacts'].items():
                            if hash_file(destination/artifact)!=expected:raise RuntimeError('Changed evaluation artifact')
                        formats[name]={**value,'status':'measured'};announce('    REUSE '+name.upper()+' | verified');continue
                    preparation_started=time.perf_counter()
                    model=build_model(profile);model.load_state_dict(saved['model_state'],strict=True);model.eval()
                    if name=='fp32':
                        storage=export_model(model,packed,profile,'fp32',measure_only=True)
                        storage.update(packed_model_sha256=hash_file(parent/'checkpoints/final.ckpt'),
                                       model_reference=str(parent/'checkpoints/final.ckpt'),check_parent_parity=True,
                                       fp32_model_duplicated=False,
                                       artifact_file_bytes_is_reference_export_estimate=True,
                                       reference_checkpoint_file_bytes=(parent/'checkpoints/final.ckpt').stat().st_size,
                                       archive_bytes_added_for_fp32_model=0)
                        model=model.to(device)
                    else:
                        bits=int(name[3:]);count=sum(p.numel() for p in model.parameters())
                        model=convert(model,bits);storage=export(model,profile,bits,packed)
                        storage.update(unique_learned_parameters_before_conversion=count,packed_model_sha256=hash_file(packed))
                        del model;model=load(packed,device)
                    torch.cuda.synchronize(device)
                    storage['conversion_export_reload_seconds']=time.perf_counter()-preparation_started
                    audit_context=NoFloatGemm() if name!='fp32' else __import__('contextlib').nullcontext()
                    # Audit separately so Python dispatch inspection does not bias latency measurements.
                    sample=bundle.datasets['test'][0]
                    sample=sample[0] if isinstance(sample,(tuple,list)) else sample
                    x=sample.to(device).unsqueeze(1)
                    with torch.inference_mode(), audit_context:model(x,x[-1:])
                    effective_settings=evaluation_settings(spec,profile,old,name)
                    storage['evaluation_policy']=binding['evaluation_policy']
                    announce(f"    {name.upper()} | score batch={effective_settings['score_batch_size_by_dataset'][profile['dataset']]}")
                    value=evaluate(model,bundle,profile,destination,device,effective_settings,old,scores,args.data_root,storage,progress_root=folder)
                    if name!='fp32':
                        operations=value['resources']['analytic_operations']
                        operations['integer_MACs_per_window']=operations['matmul_MACs_per_window']
                        operations['integer_arithmetic_operations_per_window']=2*operations['matmul_MACs_per_window']
                        operations['FP32_equivalent_matmul_FLOPs_per_window']=operations.pop('matmul_FLOPs_per_window')
                        operations['integer_count_scope']='Mathematical matrix operations; excludes padding and quantization overhead, not a hardware instruction count'
                        write_json(destination/'result.json',value)
                    formats[name]={**value,'status':'measured'}
                    if name!='fp32':write_json(destination/'integer_kernel_audit.json',{'instructions':kernel_audit(),'no_float_gemm_guard':True,'native_INT4':False})
                    del model
                    check_output_budget(out,spec['storage']['max_output_gib'])
                report={'signature':signature,'profile':profile,'stage':'integer','formats':formats,'execution':execution,
                        'total_seconds':time.perf_counter()-started,'resources_end':process_resources(device),'gpu_snapshot':gpu_snapshot()}
                write_json(folder/'metrics.json',report);record(folder,'complete',seconds=report['total_seconds'])
                artifacts={str(p.relative_to(folder)):hash_file(p) for p in sorted(folder.rglob('*'))
                           if p.is_file() and p not in (folder/'status.json',folder/'.lock')}
                write_json(folder/'status.json',{'status':'complete','signature':signature,'updated':now(),'artifacts':artifacts})
                item.update(status='complete',seconds=report['total_seconds']);campaign['tables']=summarize(out)
                del saved,scores
        except KeyboardInterrupt:
            item['status']='interrupted';campaign['status']='interrupted';return 130
        except Exception as exc:
            item.update(status='failed',reason=f'{type(exc).__name__}: {exc}');campaign['status']='failed'
            existing_status=folder/'status.json'
            completed=existing_status.exists() and json.loads(existing_status.read_text()).get('status')=='complete'
            if folder.exists() and not args.check and not completed:
                write_json(folder/'error.json',{'reason':item['reason'],'traceback':traceback.format_exc(),'updated':now()})
            traceback.print_exc();return 1
        finally:
            if not args.check:
                campaign['updated']=now();write_json(out/'campaign.json',campaign)
    if args.check:
        completed=sum(j['recovery']['action'] in ('skip_complete','preserve_complete') for j in campaign['jobs'])
        print(f'PREFLIGHT OK | {len(jobs)} verified parents | {completed} completed models verified / skip | {len(jobs)-completed} pending | no results written',flush=True)
    else:
        campaign['status']='complete';campaign['finished']=now();campaign['tables']=summarize(out);write_json(out/'campaign.json',campaign)
        print('V17 COMPLETE | '+str(out),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
