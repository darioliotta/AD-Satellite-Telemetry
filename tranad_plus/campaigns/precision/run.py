#!/usr/bin/env python3
"""V16 compact: PTQ for every parent -> two original-author SMD replicas."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'toast-diagnostics-mpl'))
if __package__ in (None,''):__package__='tranad_plus.campaigns.precision'

from tranad_plus.core.common import now,digest,hash_file,write_json,append_json,run_lock,source_manifest


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','ptq','smd'),default='all')
    p.add_argument('--plan',type=Path,default=Path(__file__).parent/'plan.json')
    p.add_argument('--parent-root',type=Path,default=PROJECT/'results/v13/architectures')
    p.add_argument('--data-root',type=Path,default=PROJECT/'data')
    p.add_argument('--output-root',type=Path,default=PROJECT/'results/v16')
    p.add_argument('--dataset',choices=('OPS-SAT','SMAP','MSL','SMD'),action='append')
    p.add_argument('--variant',choices=('reference','shared_decoder','single_pass','encoder_only'),action='append')
    p.add_argument('--seed',type=int,action='append',help='Select parent model seeds; SMD uses its predeclared seeds')
    p.add_argument('--only',help='Exact job identifier from --dry-run')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-threads',type=int,default=8)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--check',action='store_true')
    p.add_argument('--no-progress',action='store_true');p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--stop-after-epoch',type=int,help='Save and stop after this SMD epoch (recovery test)')
    args=p.parse_args(argv)
    if args.cpu_threads<1 or (args.stop_after_epoch is not None and args.stop_after_epoch<1):p.error('Invalid positive count')
    return args


def plan(spec,args):
    if spec.get('schema_version')!=1 or spec.get('name')!='v16_precision':raise ValueError('Unsupported V16 plan')
    if spec.get('campaigns')!=['ptq','smd']:raise ValueError('Compact V16 runs PTQ followed by SMD, without QAT')
    if spec.get('storage',{}).get('policy')!='compact':raise ValueError('Use compact output storage')
    if spec['evaluation'].get('archive_policy')!='compact_summary_only' or spec['evaluation'].get('save_pointwise_arrays') is not False or spec['evaluation'].get('save_reconstructions') is not False:
        raise ValueError('Heavy evaluation arrays are disabled in compact V16')
    if set(spec['variants'])!={'reference','shared_decoder','single_pass','encoder_only'} or len(spec['variants'])!=4:
        raise ValueError('All four architectures are required')
    if set(spec['configurations'])!={'reference','window_only','ff_only','compact'} or len(spec['configurations'])!=4:
        raise ValueError('All four window/feed-forward controls are required')
    if spec['evaluation']['alpha_steps']!=21 or spec['evaluation']['paper_threshold_steps']!=200:raise ValueError('Use original alpha/BF200 grids')
    if not spec['ptq_formats'] or not set(spec['ptq_formats'])<={'fp16','int8','int4'}:raise ValueError('Unsupported precision')
    if len(set(spec['ptq_formats']))!=len(spec['ptq_formats']):raise ValueError('Duplicate precision cases')
    if spec['quantization']['scheme']!='symmetric_per_tensor_fixed_parent_absmax' or spec['quantization']['execution']!='dequantized_float32_simulation':
        raise ValueError('This implementation uses fixed parent per-tensor scales and FP32 numerical simulation')
    expected=json.loads((Path(__file__).parent/'author_smd.json').read_text())['effective_profile']
    for key,value in expected.items():
        if spec['smd'].get(key)!=value:raise ValueError(f'SMD differs from author checkpoint: {key}={value}')
    seeds=args.seed or spec['seeds']
    if not set(seeds)<=set(spec['seeds']) or len(seeds)!=len(set(seeds)):raise ValueError('Unknown/duplicate parent seeds')
    parents=[Path(ds)/v/c/f'seed_{s}' for ds in spec['datasets'] for c in spec['configurations']
             for v in spec['variants'] for s in seeds
             if (not args.dataset or ds in args.dataset) and (not args.variant or v in args.variant)]
    jobs=[{'stage':'ptq','kind':'ptq','parent':str(p),'id':'ptq/'+str(p)} for p in parents]
    if (not args.dataset or 'SMD' in args.dataset) and (not args.variant or 'reference' in args.variant):
        for seed in spec['smd_seeds']:
            profile={**spec['smd'],'seed':seed,'identifier':f'v16_SMD_author_pot_w20_ff76_l5_bs8192_e100_s{seed}'}
            jobs.append({'stage':'smd','kind':'smd','profile':profile,'id':f'smd/author_pot/seed_{seed}'})
    jobs=[j for j in jobs if (args.stage=='all' or j['stage']==args.stage) and (not args.only or j['id']==args.only)]
    if not jobs:raise ValueError('No selected jobs')
    if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate jobs')
    return jobs


def validate_output(args):
    output=args.output_root.expanduser().resolve()
    protected=[args.parent_root.resolve(),args.data_root.resolve()]
    protected += [PROJECT/n for n in ('tranad_plus','tranad_plus.core','tranad_plus.campaigns.exploration','tranad_plus.campaigns.temporal','tranad_plus.campaigns.precision','tesi','docs')]
    protected += [PROJECT/'results'/n for n in ('legacy','v10','v12','v13','v14','v15')]
    if any(output==p or p in output.parents or output in p.parents for p in protected):raise ValueError('Output overlaps a protected tree')
    return output


def complete(folder,signature,required_epoch=0):
    path=folder/'status.json'
    if not path.exists():return False
    r=json.loads(path.read_text())
    if r['signature']!=signature:raise RuntimeError('Existing V16 experiment has different code/data/profile/environment')
    if r['status']!='complete':return False
    for name,expected in r['artifacts'].items():
        if hash_file(folder/name)!=expected:raise RuntimeError('Changed result artifact: '+str(folder/name))
    return r.get('completed_epochs',0)>=required_epoch


def check_input_arrays(files,data_root):
    """Check finite native inputs in bounded chunks, without building windows."""
    import numpy as np
    for entry in files['files']:
        path=Path(data_root)/entry['file']
        if path.suffix!='.npy':continue
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        for first in range(0,len(values),16384):
            if not np.isfinite(values[first:first+16384]).all():
                raise ValueError('Nonfinite native input: '+str(path))


def main(argv=None):
    args=arguments(argv);spec=json.loads(args.plan.read_text());jobs=plan(spec,args);output=validate_output(args)
    counts={kind:sum(j['kind']==kind for j in jobs) for kind in ('ptq','smd')}
    mode='V16 PREFLIGHT' if args.check else 'V16 COMPACT'
    print(f"{mode} | {len(jobs)} jobs | PTQ={counts['ptq']} | SMD={counts['smd']} | QAT=disabled",flush=True)
    if args.check:print('Read-only checks: sources, data, parent checkpoints and compact archive estimate. No training/evaluation.',flush=True)
    print('Order: PTQ (FP32 baseline, FP16, INT8, INT4) -> SMD seed 2 train/evaluate -> SMD seed 3 train/evaluate.',flush=True)
    print('INT8/INT4: packed storage; numerical inference simulated in FP32. Output: '+str(output),flush=True)
    print(f"Compact storage: models + scalar diagnostics; no full scores/reconstructions. Output budget: {spec['storage']['max_output_gib']} GiB.",flush=True)
    if args.dry_run:
        for i,j in enumerate(jobs,1):print(f"{i:03d}/{len(jobs):03d} {j['id']}")
        return 0
    os.environ.update(TRANADPLUS_CUDA_PRECISION='faithful',TRANADPLUS_ADAM_IMPL='standard',TRANADPLUS_GPU_CACHE='auto',
                      TRANADPLUS_DATA_ROOT=str(args.data_root.resolve()),V16_NO_PROGRESS='1' if args.no_progress else '0',
                      TOAST_NO_TQDM='1',TOAST_FORCE_TQDM='0')
    from tranad_plus.core.bootstrap import activate_baseline
    activate_baseline()
    import torch
    import numpy as np
    from gpu_settings import configure
    from tranad_plus.core.data import inventory,load_data
    from tranad_plus.core.models import build_model
    from tranad_plus.core.training.engine import load_checkpoint,train,TrainingInterrupted,seed_all
    from tranad_plus.campaigns.exploration.run import read_parent
    from tranad_plus.campaigns.temporal.run import reconcile_epochs
    from .quantization import export_model,load_export,recipes
    from .evaluation import evaluate
    from .progress import bar,announce,logged,native_training,record,NAMES
    from .resources import process_resources,gpu_snapshot
    from .storage import model_archive_estimate,storage_estimate,check_output_budget,output_bytes,StorageBudgetError
    from .summarize import summarize
    sources=source_manifest()
    own={p.name:hash_file(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}
    own.update({'plan.json':hash_file(args.plan),'author_smd.json':hash_file(Path(__file__).parent/'author_smd.json')})
    for package in ('tranad_plus.campaigns.exploration','tranad_plus.campaigns.temporal'):
        own.update({package+'/'+p.name:hash_file(p) for p in sorted((PROJECT/package.replace('.', '/')).glob('*.py')) if not p.name.startswith('test_')})
    torch.set_num_threads(args.cpu_threads);device=torch.device(args.device)
    if device.type not in ('cpu','cuda'):raise ValueError('Use CPU or CUDA')
    if not args.check:
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU must be explicit')
        configure()
    execution={'torch':str(torch.__version__),'python':platform.python_version(),'cuda':torch.version.cuda,'device':str(device),
               'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' and torch.cuda.is_available() else None,
               'cpu_threads':args.cpu_threads,'precision':'fp32_tf32_disabled','optimizer':'Adam_standard',
               **{p:version(p) for p in ('numpy','pandas','scipy','scikit-learn')}}
    report={'name':spec['name'],'started':now(),'status':'running','counts':counts,'planned_jobs':len(jobs),
            'archive_policy':'compact_summary_only',
            'planned_ids':[j['id'] for j in jobs],'runs':[]}
    report_path=output/'_campaign_logs'/('v16_'+now().replace(':','-')+'.json')
    parents_cache={};data_cache={};storage_parts=[];checked_inputs=set()
    def save_report():
        if not args.check:
            write_json(report_path,report);write_json(output/'campaign_latest.json',report)
    def bundle_for(profile,files):
        key=digest({'dataset':profile['dataset'],'window':profile['window_sz'],'files':files})
        if key not in data_cache:
            data_cache.clear();data_cache[key]=load_data(profile,args.data_root,files)
        return data_cache[key]
    save_report()
    if not args.check:
        environment={'execution':execution,'platform':platform.platform(),'gpu_snapshot':gpu_snapshot(),'process':process_resources(device)}
        environment_path=report_path.with_name(report_path.stem+'_environment.json')
        write_json(environment_path,environment);write_json(output/'environment.json',environment)
        report['environment_file']=str(environment_path);save_report()
    with bar(jobs,total=len(jobs),desc='V16 CHECK | parents' if args.check else 'V16 RUN | total jobs',position=0,unit='job',
             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed}]') as overall:
        for index,job in enumerate(overall,1):
            folder=output/job['id'];begun=time.monotonic();item={'id':job['id'],'stage':job['stage'],'kind':job['kind'],'started':now()}
            report['current_job']=job['id'];save_report()
            try:
                parent=None;parent_report=None
                if 'parent' in job:
                    parent=args.parent_root.resolve()/job['parent']
                    if job['parent'] not in parents_cache:
                        profile=json.loads((parent/'config.json').read_text())
                        expected=(profile['dataset'],profile['variant'],profile['configuration'],f"seed_{profile['seed']}")
                        if tuple(Path(job['parent']).parts)!=expected:raise RuntimeError('Parent folder/profile mismatch')
                        files=inventory(profile,args.data_root)
                        old,_=read_parent(parent,sources,files)
                        parents_cache[job['parent']]=(profile,files,old)
                    profile,files,parent_report=parents_cache[job['parent']]
                else:profile=job['profile'];files=inventory(profile,args.data_root)
                if args.check:
                    if profile['dataset'] not in checked_inputs:
                        check_input_arrays(files,args.data_root);checked_inputs.add(profile['dataset'])
                    if parent:
                        saved=load_checkpoint(parent/'checkpoints/final.ckpt',profile,parent_report['signature'],final=True)
                    else:
                        model=build_model(profile)
                        if model.transformer_encoder.layers[0].linear1.out_features!=76 or len(model.transformer_encoder.layers)!=5:
                            raise RuntimeError('SMD model is not the author FF76/L5 configuration')
                        del model
                    storage_parts.append(model_archive_estimate(profile,spec['ptq_formats'] if parent else ['fp32'],
                                                                checkpoint=saved if parent else None,smd=not bool(parent)))
                    item['status']='checked';continue
                binding={'job':job,'profile':profile,'files':files,'sources':sources,'v16_sources':own,'execution':execution,'plan':spec,
                         'parent_signature':parent_report['signature'] if parent_report else None,
                         'parent_artifacts':parent_report['artifacts'] if parent_report else None}
                signature=digest(binding)
                with run_lock(folder):
                    manifest=folder/'manifest.json'
                    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:raise RuntimeError('Refusing to overwrite a different V16 experiment')
                    if complete(folder,signature):item['status']='skipped_verified_complete';continue
                    check_output_budget(output,spec['storage']['max_output_gib'])
                    if not manifest.exists():
                        write_json(manifest,{'signature':signature,'created':now(),'binding':binding});write_json(folder/'config.json',profile)
                    write_json(folder/'status.json',{'status':'running','signature':signature,'updated':now()})
                    append_json(folder/'hardware_samples.jsonl',{'event':'attempt_start','resources':process_resources(device),
                                                               'gpu':gpu_snapshot() if device.type=='cuda' else None})
                    label=job['kind'].upper()
                    header=(f"[{index:03d}/{len(jobs):03d}] {label} | {profile['dataset']} | {NAMES[profile['variant']]} | "
                            f"{profile['configuration']} | s{profile['seed']}\n"
                            f"  W={profile['window_sz']} FF={profile['dim_feedforward']} L={profile['num_layers']} batch={profile['batch_sz']}")
                    if job['stage']=='smd':header+=f" | train {profile['epochs']} epochs | lr={profile['lr']} q={profile['pot_q']}"
                    announce('\n'+header)
                    bundle=bundle_for(profile,files);write_json(folder/'data_inventory.json',bundle.inventory)
                    with logged(folder):
                        print(header,flush=True)
                        original=None;parent_scores=None
                        if parent:
                            seed_all(profile['seed'])
                            saved=load_checkpoint(parent/'checkpoints/final.ckpt',profile,parent_report['signature'],final=True)
                            parent_state=saved['model_state'];del saved
                            original=build_model(profile);original.load_state_dict(parent_state,strict=True)
                            with np.load(parent/'scores.npz',allow_pickle=False) as f:
                                parent_scores=f['test_scores'].copy();np.testing.assert_array_equal(f['test_labels'],bundle.labels['test'])
                        if job['kind']=='ptq':
                            model=original;formats=['fp32']+spec['ptq_formats']
                        else:
                            # Uses the audited native objective/Adam/scheduler and 100 author epochs.
                            reconcile_epochs(folder)
                            with native_training(folder,profile,device):train(profile,bundle,folder,signature,device,stop_after_epoch=args.stop_after_epoch)
                            saved=load_checkpoint(folder/'checkpoints/final.ckpt',profile,signature,final=True)
                            model=build_model(profile);model.load_state_dict(saved['model_state'],strict=True);del saved
                            formats=['fp32']
                        def evaluate_formats(destination):
                          results={}
                          for name in formats:
                            announce('    FORMAT '+name.upper()+' | export/reload')
                            scales=recipes(original,int(name[3:])) if name.startswith('int') and original is not None else None
                            packed=destination/'models'/(name+'.npz')
                            cached_path=destination/'evaluation'/name/'result.json'
                            if cached_path.exists():
                                cached=json.loads(cached_path.read_text())
                                if hash_file(packed)!=cached['storage']['packed_model_sha256']:raise RuntimeError('Changed packed model: '+str(packed))
                                for artifact,expected in cached['artifacts'].items():
                                    if hash_file(cached_path.parent/artifact)!=expected:raise RuntimeError('Changed format artifact')
                                results[name]={**cached,'status':'measured'}
                                announce(f'    REUSE {name.upper()} | verified')
                                continue
                            export_start=time.perf_counter()
                            storage=export_model(model,packed,profile,name,scales,
                                      provenance={'stage':job['stage'],'kind':job['kind'],
                                                  'parent_signature':parent_report['signature'] if parent_report else None,
                                                  'completed_training_epochs':profile['epochs']})
                            storage['export_seconds']=time.perf_counter()-export_start
                            storage['packed_model_sha256']=hash_file(packed)
                            storage['check_parent_parity']=job['kind']=='ptq' and name=='fp32'
                            runtime=None
                            try:
                                load_start=time.perf_counter()
                                runtime=load_export(packed,device)
                                storage['load_and_dequantize_seconds']=time.perf_counter()-load_start
                                results[name]=evaluate(runtime,bundle,profile,destination/'evaluation'/name,device,spec['evaluation'],
                                                       parent_report,parent_scores,args.data_root,storage,progress_root=folder)
                                results[name]['status']='measured'
                            except (FloatingPointError,RuntimeError) as exc:
                                unsupported=isinstance(exc,RuntimeError) and any(t in str(exc).lower() for t in ('not implemented','not supported','unsupported dtype'))
                                if name=='fp32' or (not isinstance(exc,FloatingPointError) and not unsupported):raise
                                results[name]={'status':'not_available','reason':f'{type(exc).__name__}: {exc}','storage':storage}
                                write_json(destination/'evaluation'/name/'unavailable.json',results[name])
                                announce('    UNAVAILABLE '+name.upper()+': '+str(exc))
                            finally:
                                if runtime is not None:del runtime
                          return results
                        report_job={'signature':signature,'profile':profile,'stage':job['stage'],'kind':job['kind'],
                                    'parent_training':parent_report.get('training') if parent_report else None}
                        report_job['formats']=evaluate_formats(folder)
                        if job['stage']=='smd':
                            report_job['author_reference']=json.loads((Path(__file__).parent/'author_smd.json').read_text())
                        history_path=folder/'epochs.jsonl'
                        if history_path.exists():
                            history=[json.loads(line) for line in history_path.read_text().splitlines()]
                            report_job['training_telemetry']={'epochs_logged':len(history),
                               'epoch_compute_seconds':sum(r.get('seconds',r.get('elapsed_seconds',0)) for r in history),
                               'phase_wall_seconds_before_report':time.monotonic()-begun}
                        report_job['process_end']=process_resources(device)
                        write_json(folder/'metrics.json',report_job)
                        del model
                    record(folder,'complete',seconds=time.monotonic()-begun)
                    append_json(folder/'hardware_samples.jsonl',{'event':'attempt_end','resources':process_resources(device),
                                                               'gpu':gpu_snapshot() if device.type=='cuda' else None})
                    artifacts={str(p.relative_to(folder)):hash_file(p) for p in sorted(folder.rglob('*'))
                               if p.is_file() and p.suffix in ('.json','.npz','.ckpt','.jsonl') and p.name!='status.json'}
                    write_json(folder/'status.json',{'status':'complete','signature':signature,'updated':now(),'artifacts':artifacts,
                                                    'completed_epochs':profile['epochs'] if job['stage']=='smd' else 0})
                    check_output_budget(output,spec['storage']['max_output_gib'])
                    item['status']='complete';report['tables']=summarize(output)
                    announce(f"    JOB COMPLETE | {time.monotonic()-begun:.1f} s")
            except TrainingInterrupted as exc:
                item.update(status='interrupted_after_checkpoint',reason=str(exc));report['status']='interrupted_after_checkpoint';return 75
            except KeyboardInterrupt:
                item['status']='interrupted';report['status']='interrupted';return 130
            except Exception as exc:
                item.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
                announce('    FAILED '+job['id']+': '+item['reason']);traceback.print_exc()
                if folder.exists():write_json(folder/'error.json',{'reason':item['reason'],'traceback':traceback.format_exc(),'updated':now()})
                if args.fail_fast or isinstance(exc,StorageBudgetError):report['status']='failed';return 1
            finally:
                item.update(finished=now(),seconds=time.monotonic()-begun);report['runs'].append(item);save_report()
            if device.type=='cuda':torch.cuda.empty_cache()
    failures=sum(r['status']=='failed' for r in report['runs'])
    if args.check:
        report['storage_estimate']=storage_estimate(storage_parts,spec['storage']['max_output_gib'])
        announce('Storage preflight: '+json.dumps(report['storage_estimate']))
    if not args.check:report['tables']=summarize(output)
    if not args.check:report['output_storage']={'bytes':output_bytes(output),'budget_GiB':spec['storage']['max_output_gib']}
    report.update(status='completed_with_failures' if failures else ('checked' if args.check else 'complete'),
                  failures=failures,finished=now(),current_job=None);save_report()
    unavailable=report.get('tables',{}).get('unavailable_formats',0)
    announce(f"V16 {report['status']} | jobs={len(jobs)} | failures={failures} | unavailable formats={unavailable}")
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
