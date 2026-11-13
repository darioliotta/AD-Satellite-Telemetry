"""V18: BF200/POT without PA for the 240 V17 models, all three precisions."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
from tranad_plus.core.bootstrap import PROJECT_ROOT,BASELINE_ROOT
from tranad_plus.core.common import digest,hash_file,now,write_json,run_lock


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path(__file__).with_name('plan.json'))
    p.add_argument('--source-root',type=Path,default=PROJECT_ROOT/'results/v17/integer')
    p.add_argument('--parent-root',type=Path,default=PROJECT_ROOT/'results/v13/architectures')
    p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'results/v18/no_pa')
    p.add_argument('--data-root',type=Path,default=PROJECT_ROOT/'data')
    p.add_argument('--dataset',choices=('OPS-SAT','SMAP','MSL'),action='append')
    p.add_argument('--variant',choices=('reference','shared_decoder','single_pass','encoder_only'),action='append')
    p.add_argument('--configuration',choices=('reference','compact','window_only','ff_only'),action='append')
    p.add_argument('--seed',type=int,action='append')
    p.add_argument('--format',dest='formats',choices=('fp32','int8','int4'),action='append')
    p.add_argument('--only',help='Exact case path from --dry-run')
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-threads',type=int,default=8)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--check',action='store_true')
    p.add_argument('--status',action='store_true');p.add_argument('--verify',action='store_true')
    p.add_argument('--no-progress',action='store_true')
    a=p.parse_args(argv)
    if a.cpu_threads<1:p.error('Positive --cpu-threads required')
    return a


def plan(spec,args):
    if spec.get('schema_version')!=1 or spec.get('name')!='v18_no_pa':raise ValueError('Unknown plan')
    for key,required in [('datasets',{'OPS-SAT','SMAP','MSL'}),('variants',{'reference','shared_decoder','single_pass','encoder_only'}),
                         ('configurations',{'reference','window_only','ff_only','compact'}),('formats',{'fp32','int8','int4'})]:
        if set(spec[key])!=required or len(spec[key])!=len(required):raise ValueError('Incomplete/duplicate '+key)
    e=spec['evaluation']
    if e['point_adjustment_during_selection'] or e['alpha_steps']!=21 or e['bf_threshold_steps']!=200:
        raise ValueError('Use original BF200/alpha grid, selected without PA')
    if e['pot_threshold_source']!='v17_per_alpha_SPOT_thresholds' or e['pot_q_unchanged'] is not True:
        raise ValueError('Keep V17 per-alpha POT calibration')
    if e['score_parity_rtol']!=5e-4 or e['score_parity_atol']!=1e-7:raise ValueError('Original score tolerance required')
    if any(spec['storage'][k] for k in ('save_models','save_pointwise_arrays','save_reconstructions')):
        raise ValueError('Use compact report-only storage')
    if len(set(spec['seeds']))!=len(spec['seeds']) or not spec['seeds']:raise ValueError('Invalid seeds')
    if args.seed and not set(args.seed)<=set(spec['seeds']):raise ValueError('Unknown model seed')
    cases=[str(Path(d)/v/c/f'seed_{s}'/f) for d in spec['datasets'] for c in spec['configurations']
           for v in spec['variants'] for s in spec['seeds'] for f in spec['formats']
           if (not args.dataset or d in args.dataset) and (not args.variant or v in args.variant)
           and (not args.configuration or c in args.configuration) and (not args.seed or s in args.seed)
           and (not args.formats or f in args.formats)]
    cases=[c for c in cases if not args.only or c==args.only]
    if not cases:raise ValueError('No selected cases')
    return cases


def validate_output(args):
    out=args.output_root.expanduser().resolve()
    protected=[args.source_root.resolve(),args.parent_root.resolve(),args.data_root.resolve(),BASELINE_ROOT,PROJECT_ROOT/'tesi',PROJECT_ROOT/'docs']
    protected += [PROJECT_ROOT/'results'/v for v in ('legacy','v10','v12','v13','v14','v15','v16','v17')]
    if any(out==p or p in out.parents or out in p.parents for p in protected):raise ValueError('Output overlaps protected sources/data/results')
    return out


def complete(folder,signature=None):
    folder=Path(folder);status=folder/'status.json'
    if not status.exists():return False
    r=json.loads(status.read_text())
    if signature is not None and r['signature']!=signature:raise RuntimeError('Existing V18 binding differs')
    if r['status']!='complete':return False
    for name,expected in r['artifacts'].items():
        path=folder/name
        if not path.resolve().is_relative_to(folder.resolve()) or hash_file(path)!=expected:raise RuntimeError('Changed V18 artifact: '+name)
    manifest=json.loads((folder/'manifest.json').read_text())
    if digest(manifest['binding'])!=r['signature']:raise RuntimeError('V18 manifest signature differs')
    return True


def status(args,jobs,out):
    finished=[];pending=[]
    for job in jobs:
        p=out/job/'status.json'
        if p.exists() and json.loads(p.read_text()).get('status')=='complete':
            if args.verify:complete(p.parent)
            finished.append(job)
        else:pending.append(job)
    print(f'Completed evaluations: {len(finished)}/{len(jobs)} | pending/interrupted: {len(pending)}')
    p=out/'campaign.json'
    if p.exists():
        r=json.loads(p.read_text());print('Campaign status: '+r['status']);print('Last/current case: '+str(r.get('current_case')))
        if r.get('current_case'):
            live=out/r['current_case']/'progress.json'
            if live.exists():print(live.read_text())
    return 1 if args.verify and pending else 0


def main(argv=None):
    args=arguments(argv);spec=json.loads(args.plan.read_text());jobs=plan(spec,args);out=validate_output(args)
    print(f'V18 NO-PA | {len(jobs)} evaluations | BF200 + POT | original alpha grid | no training',flush=True)
    print('POT: reuse each V17 SPOT threshold; choose alpha by pointwise F1. BF200: choose alpha and threshold by pointwise F1.',flush=True)
    if args.dry_run:
        for i,j in enumerate(jobs,1):print(f'{i:03d}/{len(jobs):03d} {j}')
        return 0
    if args.status or args.verify:return status(args,jobs,out)
    os.environ.update(TRANADPLUS_CUDA_PRECISION='faithful',TRANADPLUS_GPU_CACHE='auto',
                      TRANADPLUS_DATA_ROOT=str(args.data_root.resolve()),V16_NO_PROGRESS='1' if args.no_progress else '0',TOAST_NO_TQDM='1')
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'toast-diagnostics-mpl'))
    from tranad_plus.core.bootstrap import activate_baseline
    activate_baseline()
    import numpy as np
    import torch
    from importlib.metadata import version
    from gpu_settings import configure
    from tranad_plus.core.data import inventory,load_data
    from tranad_plus.core.models import build_model
    from tranad_plus.core.training.engine import load_checkpoint
    from tranad_plus.core.evaluation.scoring import collect_outputs,reconstruction_scores
    from tranad_plus.campaigns.precision.progress import bar,announce,logged,scoring,threshold_search,record
    from tranad_plus.campaigns.precision.resources import process_resources
    from tranad_plus.campaigns.precision.storage import check_output_budget
    from tranad_plus.campaigns.integer.runtime import load as load_integer
    from tranad_plus.campaigns.integer.kernel import preflight,kernel_audit
    from .provenance import verify_source,numerical_sources,assert_score_statistics
    from .metrics import comparison
    from .summarize import summarize
    torch.set_num_threads(args.cpu_threads);device=torch.device(args.device)
    execution=None;hardware=None
    if not args.check:
        if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Use the CUDA server for V18; no floating-point substitute for integer inference')
        torch.cuda.set_device(device);configure();hardware=preflight(device)
        execution={'torch':str(torch.__version__),'python':platform.python_version(),'cuda':torch.version.cuda,
                   'device':str(device),'gpu':torch.cuda.get_device_name(device),'triton':hardware['triton_version'],
                   'capability':list(torch.cuda.get_device_capability(device)),'cpu_threads':args.cpu_threads,'TF32':False,
                   **{k:version(k) for k in ('numpy','pandas','scipy','scikit-learn')}}
    own={p.name:hash_file(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}
    own['progress_support']=hash_file(BASELINE_ROOT/'campaigns/precision/progress.py')
    campaign={'name':spec['name'],'started':now(),'status':'running','execution':execution,'cases':[]}
    verified={};bundles={}
    if not args.check:
        out.mkdir(parents=True,exist_ok=True);write_json(out/'kernel_preflight.json',hardware)
    with bar(jobs,desc='V18 | evaluations',position=0,unit='eval') as progress:
      for index,job in enumerate(progress,1):
        relative=Path(job);identity=relative.parent;fmt=relative.name;folder=out/relative;source=args.source_root.resolve()/identity
        item={'id':job,'index':index,'status':'running'};campaign['cases'].append(item);campaign['current_case']=job
        started=time.perf_counter()
        try:
            if str(identity) not in verified:
                old,manifest=verify_source(source);profile=old['profile']
                if tuple(identity.parts)!=(profile['dataset'],profile['variant'],profile['configuration'],f"seed_{profile['seed']}"):
                    raise RuntimeError('V17 folder/profile mismatch')
                files=inventory(profile,args.data_root)
                if files!=manifest['binding']['files']:raise RuntimeError('Input hashes/order differ from V17')
                numeric=numerical_sources(manifest['binding']['sources'])
                verified[str(identity)]=(old,manifest,files,numeric)
            old,manifest,files,numeric=verified[str(identity)];profile=old['profile'];reference=old['formats'][fmt]
            records=reference['oracle']['alpha_grid']
            alphas=[r['alpha'] for r in records]
            expected=np.linspace(0,1,21).tolist() if profile['variant'] in ('reference','shared_decoder') else [None]
            if alphas!=expected:raise RuntimeError('V17 alpha grid differs')
            if any(r.get('pot_threshold') is None or not np.isfinite(r['pot_threshold']) for r in records):raise RuntimeError('Missing/nonfinite V17 POT threshold')
            model_path=(args.parent_root.resolve()/identity/'checkpoints/final.ckpt') if fmt=='fp32' else source/'models'/(fmt+'.npz')
            if hash_file(model_path)!=reference['storage']['packed_model_sha256']:raise RuntimeError('Source checkpoint/export hash differs')
            batch=reference['inference_by_split']['test']['batch_size']
            if args.check:
                item['status']='checked';continue
            for key,value in old['execution'].items():
                if execution.get(key)!=value:raise RuntimeError(f'Execution environment differs from V17: {key}: {execution.get(key)} vs {value}')
            # V13 pins NumPy as V17 used the same server environment; scalar
            # promotion changes can otherwise change BF200 grid endpoints.
            parent_manifest=json.loads((args.parent_root.resolve()/identity/'manifest.json').read_text())
            for key in ('numpy','pandas','scipy'):
                if execution[key]!=parent_manifest['execution'][key]:raise RuntimeError('Library version differs from source parents: '+key)
            binding={'profile':profile,'format':fmt,'files':files,'numeric_sources':numeric,'campaign_sources':own,
                     'plan':spec,'execution':execution,'source_signature':old['signature'],
                     'source_metrics_sha256':hash_file(source/'metrics.json'),'model_sha256':hash_file(model_path),
                     'score_batch_size':batch,'pot_thresholds':records}
            signature=digest(binding)
            with run_lock(folder):
                path=folder/'manifest.json'
                if path.exists() and json.loads(path.read_text())['signature']!=signature:raise RuntimeError('Refusing to overwrite a different V18 experiment')
                if complete(folder,signature):
                    item['status']='skipped_verified_complete';announce(f'[{index:03d}/{len(jobs):03d}] VERIFIED / SKIP | {job}');continue
                check_output_budget(out,spec['storage']['max_output_gib'])
                write_json(out/'campaign.json',campaign)
                if not path.exists():write_json(path,{'signature':signature,'binding':binding,'created':now()})
                write_json(folder/'config.json',profile);write_json(folder/'status.json',{'status':'running','signature':signature,'updated':now()})
                title=f"[{index:03d}/{len(jobs):03d}] {job} | W={profile['window_sz']} FF={profile['dim_feedforward']} | batch={batch}"
                announce('\n'+title)
                data_fields=('dataset','features','window_sz','validation_fraction','split_seed','channels','max_train_fragments_per_channel','max_test_fragments_per_channel','padding','downsample')
                key=digest({k:profile.get(k) for k in data_fields})
                if key not in bundles:
                    bundles.clear();bundles[key]=load_data(profile,args.data_root,files)
                bundle=bundles[key]
                with logged(folder):
                    record(folder,'loading_model',case=index,cases_total=len(jobs),format=fmt)
                    if fmt=='fp32':
                        saved=load_checkpoint(model_path,profile,manifest['binding']['parent_signature'],final=True)
                        model=build_model(profile);model.load_state_dict(saved['model_state'],strict=True);del saved
                        model=model.to(device).eval()
                    else:model=load_integer(model_path,device)
                    record(folder,'test_inference',format=fmt,batch_size=batch)
                    with scoring('V18 '+fmt.upper()):
                        outputs=collect_outputs(model,bundle.datasets['test'],device,batch,'V18 '+fmt.upper()+' | test batches',folder/'nonfinite.json')
                    grids=np.stack([reconstruction_scores(outputs,a) for a in alphas]);inference=outputs['performance']
                    del outputs,model
                    parity=assert_score_statistics(grids,records,reference,spec['evaluation']['score_parity_rtol'],spec['evaluation']['score_parity_atol'])
                    with threshold_search(folder,len(alphas),'V18 '+fmt.upper()+' | no-PA alpha search') as advance:
                        result=comparison(grids,bundle.labels['test'],records,reference,bundle.groups.get('test'),advance)
                    warn=[k for k,v in result['v17_endpoint_parity'].items() if not v['raw_confusion_counts_match_v17'] or not v['PA_confusion_counts_match_v17']]
                    if warn:announce('    NOTE | threshold-boundary count changes vs V17: '+', '.join(warn)+'; both current comparison views are saved')
                    report={'signature':signature,'profile':profile,'format':fmt,'execution':execution,
                            'source_v17':str(source),'inference':inference,'score_parity':parity,'evaluation':result,
                            'timing_seconds':{'total_before_write':time.perf_counter()-started},'process_end':process_resources(device),
                            'storage_policy':'summary_only; models reused; score/reconstruction arrays not archived'}
                    write_json(folder/'metrics.json',report)
                    if fmt!='fp32':write_json(folder/'integer_kernel_audit.json',{'instructions':kernel_audit(),'native_INT4':False})
                    del grids
                    record(folder,'complete',seconds=time.perf_counter()-started)
                    announce(f"    DONE {fmt.upper()} | no-PA BF={result['protocols']['no_pa_optimized']['bf']['f1']:.4f} POT={result['protocols']['no_pa_optimized']['pot']['f1']:.4f}")
                artifacts={str(p.relative_to(folder)):hash_file(p) for p in sorted(folder.rglob('*'))
                           if p.is_file() and p not in (folder/'status.json',folder/'.lock')}
                write_json(folder/'status.json',{'status':'complete','signature':signature,'updated':now(),'artifacts':artifacts})
                item.update(status='complete',seconds=time.perf_counter()-started)
                # Rebuild tables every 12 cases and at completion; per-case JSON
                # remains durable even if interrupted between table rebuilds.
                if index%12==0:campaign['tables']=summarize(out)
                check_output_budget(out,spec['storage']['max_output_gib'])
        except KeyboardInterrupt:
            item['status']='interrupted';campaign['status']='interrupted';return 130
        except Exception as exc:
            item.update(status='failed',reason=f'{type(exc).__name__}: {exc}');campaign['status']='failed'
            state=folder/'status.json';finished=state.exists() and json.loads(state.read_text()).get('status')=='complete'
            if folder.exists() and not args.check and not finished:write_json(folder/'error.json',{'reason':item['reason'],'traceback':traceback.format_exc(),'updated':now()})
            traceback.print_exc();return 1
        finally:
            if not args.check:campaign['updated']=now();write_json(out/'campaign.json',campaign)
    if args.check:print(f'V18 PREFLIGHT OK | {len(verified)} V17 source models | {len(jobs)} format evaluations | sources/data/model hashes verified | no writes',flush=True)
    else:
        campaign.update(status='complete',finished=now(),tables=summarize(out));write_json(out/'campaign.json',campaign)
        print('V18 COMPLETE | '+str(out),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
