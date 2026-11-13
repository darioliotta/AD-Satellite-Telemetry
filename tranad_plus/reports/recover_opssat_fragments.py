"""Recover OPS-SAT fragment metrics from immutable archives; never train.

Primary endpoint: fixed alpha=.5 (or single output), p95 within each fragment,
BF200 threshold selected on the original validation fragments, frozen on test.
Historical test-label oracles stay separate. CPU reconstructions are labelled
as such and checked against the historical numerical summaries before reuse.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys, time, traceback
from collections import Counter, defaultdict
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/toast-matplotlib')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tranad_plus'))
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

OUT = ROOT / 'results/analysis_opssat_fragments'
DATA_HASH = None
SCHEMA = 1
WINDOW_CACHE = {}
DEVICE = 'cpu'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)

def csvfile(path, rows):
    if not rows: return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader(); w.writerows(rows)

def verified(path, expected=None):
    got = digest(path)
    if expected is not None and got != expected:
        raise RuntimeError('Changed historical artifact: ' + str(path))
    return {'file': str(Path(path).relative_to(ROOT)), 'sha256': got}

def counts(y, p):
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool)
    tp, fp, tn, fn = (int(np.sum(a)) for a in (y&p, ~y&p, ~y&~p, y&~p))
    return {'TP':tp, 'FP':fp, 'TN':tn, 'FN':fn,
            'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'AUC_bin':.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None}

def bf200(s, y):
    s = np.asarray(s); y = np.asarray(y).astype(bool)
    grid = np.linspace(float(s.min()), float(s.mean()+3*s.std()), 200)
    tp = int(y.sum())-np.searchsorted(np.sort(s[y]), grid, side='right')
    fp = int((~y).sum())-np.searchsorted(np.sort(s[~y]), grid, side='right')
    den = int(y.sum())+tp+fp
    f = np.divide(2*tp, den, out=np.zeros(200), where=den>0)
    k = int(np.argmax(f))
    return float(grid[k]), k

def aggregate(s, y, ids):
    s, y, ids = np.asarray(s), np.asarray(y), np.asarray(ids)
    if not (s.ndim == y.ndim == ids.ndim == 1 and len(s) == len(y) == len(ids)):
        raise ValueError('Unaligned point scores/labels/fragment IDs')
    if not np.isfinite(s).all(): raise ValueError('Nonfinite original scores')
    edges = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1, len(ids)]
    if len(np.unique(ids)) != len(edges)-1: raise ValueError('Noncontiguous fragment')
    result = {'ids':[], 'labels':[], 'p95':[], 'max':[], 'lengths':[]}
    for a,b in zip(edges[:-1], edges[1:]):
        if not np.all(y[a:b] == y[a]): raise ValueError('Mixed labels within fragment')
        result['ids'].append(int(ids[a])); result['labels'].append(int(y[a]))
        result['p95'].append(float(np.quantile(s[a:b], .95)))
        result['max'].append(float(s[a:b].max())); result['lengths'].append(int(b-a))
    order = np.argsort(result['ids'])
    return {k:np.asarray(v)[order] for k,v in result.items()}

def metric(s, y, threshold):
    return {**counts(y, np.asarray(s)>threshold), 'threshold':float(threshold)}

def finish(identity, profile, splits, sources, recovery, context=None):
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    path = OUT/'cases'/f'{key}.json'
    result = {'schema':SCHEMA, 'identity':identity, 'profile':profile, 'sources':sources,
              'recovery':recovery, 'data_sha256':DATA_HASH, 'point_adjustment':False,
              'counting_unit':'one_fragment_one_vote', 'metrics':{},
              'historical_context':context or {}, 'split_counts':{}}
    arrays = {}
    for split, a in splits.items():
        result['split_counts'][split] = {'fragments':len(a['ids']), 'anomalous':int(a['labels'].sum())}
        arrays.update({f'{split}_{k}':v for k,v in a.items()})
    for agg in ('p95', 'max'):
        if agg not in splits['test']: continue
        s, y = splits['test'][agg], splits['test']['labels']
        t, index = bf200(s,y)
        m = {'AUC_score':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,
             'AP_score':float(average_precision_score(y,s)) if len(np.unique(y))==2 else None,
             'all_positive':counts(y,np.ones(len(y))),
             'test_oracle_BF200':{**metric(s,y,t), 'threshold_index':index,
                                 'selection_uses_test_labels':True}}
        if 'validation' in splits and agg in splits['validation']:
            va = splits['validation']; threshold, vi = bf200(va[agg],va['labels'])
            m['validation_BF200'] = {**metric(s,y,threshold), 'threshold_index':vi,
                                     'validation':metric(va[agg],va['labels'],threshold),
                                     'selection_uses_test_labels':False}
        if 'train' in splits and agg in splits['train']:
            tr = splits['train']; nominal=tr[agg][tr['labels']==0]
            threshold=float(np.quantile(nominal,.995))
            m['train_nominal_quantile_995']={**metric(s,y,threshold),'selection_uses_test_labels':False}
        result['metrics'][agg]=m
    result['primary_available']='validation_BF200' in result['metrics']['p95']
    if not result['primary_available']:
        result['primary_unavailable_reason']='No original held-out validation scores; test-oracle/threshold-free diagnostics only.'
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    result['fragment_arrays_sha256']=digest(path.with_suffix('.npz'))
    write(path,result)
    return result

def cached(identity):
    key=hashlib.sha256(identity.encode()).hexdigest()[:16]
    path=OUT/'cases'/f'{key}.json'
    if not path.exists():return None
    r=read(path)
    if r['schema']!=SCHEMA or r['data_sha256']!=DATA_HASH:return None
    for source in r['sources']:
        verified(ROOT/source['file'],source['sha256'])
    verified(path.with_suffix('.npz'),r['fragment_arrays_sha256'])
    return r

def row_of(z, split):
    values=z[split+'_scores']
    if values.ndim==1:return values
    if len(values)==1:return values[0]
    alphas=z['alpha_values']; idx=np.flatnonzero(np.isclose(alphas,.5,rtol=0,atol=1e-12))
    if len(idx)!=1:raise ValueError('Missing fixed-alpha .5 row')
    return values[int(idx[0])]

def standardized_archive(path):
    identity='archive:'+str(path.relative_to(ROOT))
    if cached(identity):return
    report_path=path.parent/'metrics.json'; report=read(report_path)
    profile={**report['profile']}
    expected=report.get('artifacts',{}).get(path.name)
    # V15 records its compact artifacts in status, rather than metrics.
    status=path.parent/'status.json'
    if status.exists():expected=read(status).get('artifacts',{}).get(path.name,expected)
    sources=[verified(path,expected),verified(report_path)]
    splits={}
    with np.load(path,allow_pickle=False) as z:
        for split in ('train','validation','test'):
            if split+'_scores' not in z:continue
            idkey=split+'_segment_ids'
            if idkey not in z:
                metadata=path.parent/'scores_matched.npz'
                if not metadata.exists():raise ValueError('Missing fragment IDs '+str(path))
                with np.load(metadata,allow_pickle=False) as meta:ids=meta[idkey].copy()
                sources.append(verified(metadata))
            else:ids=z[idkey]
            splits[split]=aggregate(row_of(z,split),z[split+'_labels'],ids)
        if path.name=='scores_ordered_test.npz':
            # Same model; validation remains the training-matched evaluation.
            parent_case=standard_case_data(path.parent/'scores_matched.npz')
            splits['validation']=parent_case['validation']
    finish(identity,profile,splits,sources,{'mode':'original_stored_scores','alpha':.5 if profile['variant'] in ('reference','shared_decoder') else None,
           'input_condition':profile.get('input_condition','original'),'archive':str(path.relative_to(ROOT))})

def standard_case_data(path):
    splits={}
    with np.load(path,allow_pickle=False) as z:
        for split in ('train','validation','test'):
            if split+'_scores'in z and split+'_segment_ids'in z:
                splits[split]=aggregate(row_of(z,split),z[split+'_labels'],z[split+'_segment_ids'])
    return splits

def v20_existing(path):
    identity='v20:'+str(path.relative_to(ROOT))
    if cached(identity):return
    folder=path.parent;r=read(path);status=read(folder/'status.json');manifest=read(folder/'manifest.json')
    sources=[verified(path,status['artifacts']['result.json']),verified(folder/'manifest.json',status['artifacts']['manifest.json'])]
    p=folder/'fragment_scores.npz';sources.append(verified(p,status['artifacts'][p.name]))
    splits={}
    with np.load(p,allow_pickle=False) as z:
        for split in ('test','validation'):
            order=np.argsort(z[split+'_ids'])
            splits[split]={'ids':z[split+'_ids'][order], 'labels':z[split+'_labels'][order],
                           'p95':z[split+'_scores'][order]}
    profile=manifest['binding']['profile'];profile={**profile,'format':r['identity'].get('format','fp32')}
    result=finish(identity,profile,splits,sources,{'mode':'original_v20_fragment_arrays','alpha':r.get('fixed_alpha'),
                  'family':r['identity']['family'],'stage':r['identity']['stage'],
                  'identity':r['identity']},r.get('historical_paper_endpoints_context_only'))
    expected=r['metrics']['fragment_classification']
    np.testing.assert_allclose(result['metrics']['p95']['validation_BF200']['threshold'],expected['threshold'],rtol=0,atol=0)
    for k in ('TP','FP','TN','FN','f1','precision','recall'):
        assert abs(result['metrics']['p95']['validation_BF200'][k]-expected['test'][k])<1e-12

def bundle_for(profile):
    from tranad_plus.core.data import load_data
    fields=('window_sz','validation_fraction','split_seed','seed','channels',
            'max_train_fragments_per_channel','max_test_fragments_per_channel')
    key=json.dumps({k:profile.get(k) for k in fields},sort_keys=True)
    if key not in WINDOW_CACHE:
        if len(WINDOW_CACHE)>8:WINDOW_CACHE.clear()
        WINDOW_CACHE[key]=load_data(profile,ROOT/'data')
    return WINDOW_CACHE[key]

def infer(model,dataset,alpha,batch=1024):
    import torch
    model.eval();outputs=[[],[]];targets=[];started=time.perf_counter()
    with torch.inference_mode():
        for a in range(0,len(dataset),batch):
            # OPS-SAT FragmentWindows exposes the original normalized windows.
            x=dataset.windows[a:a+batch].to(DEVICE).permute(1,0,2);y=x[-1:]
            values=model(x,y);values=values if isinstance(values,tuple) else (values,)
            for i,v in enumerate(values):outputs[i].append(v[0].cpu().numpy())
            targets.append(y[0].cpu().numpy())
    outputs=[np.concatenate(v) for v in outputs if v];target=np.concatenate(targets)
    def score(a):
        rec=outputs[0] if len(outputs)==1 else float(a)*outputs[0]+(1-float(a))*outputs[1]
        return np.mean((rec-target)**2,axis=1)
    values=score(alpha)
    if not np.isfinite(values).all():raise ValueError('Nonfinite replay')
    return values,score,{'seconds':time.perf_counter()-started,'batch':batch,'points':len(dataset)}

def replay_legacy(report_path):
    identity='legacy:'+str(report_path.relative_to(ROOT))
    if cached(identity):return
    import torch
    from tranad_plus.core.models import build_model
    r=read(report_path);profile={**r['profile'],'features':1,'variant':'reference','format':'fp32'}
    profile['segments_csv']=str(ROOT/'data/raw/OPSSAT/segments.csv')
    path=report_path.parent/r['raw_pointwise_evaluation']['scores_file']
    checkpoint=report_path.parent/r['checkpoint']['file']
    sources=[verified(report_path),verified(path,r['raw_pointwise_evaluation'].get('scores_sha256')),
             verified(checkpoint,r['checkpoint']['sha256'])]
    with np.load(path,allow_pickle=False) as z:
        splits={'test':aggregate(z['scores'],z['labels'],z['segment_ids']),
                'train':aggregate(z['train_scores'],np.zeros(len(z['train_scores']),dtype=np.int8),z['train_segment_ids'])}
        for k in ('p95','max'):
            np.testing.assert_allclose(splits['test'][k],z['segment_scores_'+k],rtol=1e-6,atol=1e-10)
        original_scores=z['scores'].copy();original_ids=z['segment_ids'].copy();original_labels=z['labels'].copy()
    parity={'stored_test_fragment_aggregation_verified':True}
    if float(profile.get('validation_fraction',0)):
        bundle=bundle_for(profile)
        np.testing.assert_array_equal(bundle.datasets['test'].segment_ids,original_ids)
        np.testing.assert_array_equal(bundle.labels['test'],original_labels)
        model=build_model(profile);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        original_inventory=state.get('opssat_inventory',{})
        if original_inventory.get('source_sha256'):
            assert original_inventory['source_sha256']==DATA_HASH
        for name in ('train_fragment_ids','test_fragment_ids','validation_fragment_ids'):
            if name in original_inventory:
                assert original_inventory[name]==bundle.inventory['split'][name],name
        model.load_state_dict(state['model_states'],strict=True);model.to(DEVICE).eval()
        # Verify real, deterministic points throughout the stream before replaying validation.
        selected=np.unique(np.linspace(0,len(original_scores)-1,256,dtype=int))
        class Probe:
            windows=bundle.datasets['test'].windows[selected]
            def __len__(self):return len(self.windows)
        scores,_,timing=infer(model,Probe(),.5,batch=256)
        np.testing.assert_allclose(scores,original_scores[selected],rtol=5e-4,atol=1e-7)
        va,_,timing=infer(model,bundle.datasets['validation'],.5)
        splits['validation']=aggregate(va,bundle.labels['validation'],bundle.datasets['validation'].segment_ids)
        parity.update({'CPU_replay_probe_points':len(selected),'probe_max_abs_drift':float(np.abs(scores-original_scores[selected]).max()),
                       'validation_inference':timing})
    finish(identity,profile,splits,sources,{'mode':'stored_test_scores_and_CPU_validation_replay' if 'validation'in splits else 'stored_scores_without_original_validation',
           'alpha':.5,'parity':parity},r['original_paper_evaluation'])

def replay_quant(folder,format_name,kind):
    identity='quant:'+str(folder.relative_to(ROOT))+':'+format_name
    if cached(identity):return
    from tranad_plus.campaigns.precision.quantization import load_export
    from tranad_plus.campaigns.integer.runtime import load as load_integer
    report_path=folder/'metrics.json';r=read(report_path);status=read(folder/'status.json')
    profile={**r['profile'],'format':format_name};export=folder/'models'/f'{format_name}.npz'
    sources=[verified(report_path,status['artifacts']['metrics.json']),
             verified(export,status['artifacts'][str(export.relative_to(folder))])]
    if format_name=='fp32':model=load_export(export,DEVICE)
    elif kind=='weight_only':model=load_export(export,DEVICE)
    else:model=load_integer(export,DEVICE,backend='reference_cpu' if DEVICE=='cpu' else 'cuda_triton')
    alpha=.5 if model.output_count==2 else None
    bundle=bundle_for(profile);splits={};parity={};timings={}
    parent=ROOT/f"results/v13/architectures/OPS-SAT/{profile['variant']}/{profile['configuration']}/seed_{profile['seed']}/scores.npz"
    with np.load(parent,allow_pickle=False) as z:
        for split in ('train','validation','test'):
            np.testing.assert_array_equal(bundle.datasets[split].segment_ids,z[split+'_segment_ids'])
            np.testing.assert_array_equal(bundle.labels[split],z[split+'_labels'])
    for split in ('test','validation'):
        scores,score,timing=infer(model,bundle.datasets[split],alpha)
        index=10 if alpha is not None else 0
        expected=r['formats'][format_name]['score_statistics_by_split'][split][index]
        summary={'mean':float(scores.mean()),'std':float(scores.std()),'min':float(scores.min()),'max':float(scores.max())}
        np.testing.assert_allclose([summary[k] for k in summary],[expected[k] for k in summary],rtol=5e-4,atol=1e-7)
        splits[split]=aggregate(scores,bundle.labels[split],bundle.datasets[split].segment_ids)
        timings[split]=timing
        parity[split]={'historical_score_summary_check_passed':True,'actual':summary,'historical':expected}
        if split=='test':
            # Check old PA counts with exact old alpha/threshold; retain boundary drift explicitly.
            from opssat_pipeline import adjust_per_fragment
            endpoint_checks={}
            for method in ('bf','pot'):
                e=r['formats'][format_name]['oracle'][method];oldscore=score(e['alpha']);thr=e[method+'_threshold']
                pa=adjust_per_fragment(oldscore,bundle.labels['test'],bundle.groups['test'],thr)
                c=counts(bundle.labels['test'],pa)
                delta={k:c[k]-int(e[method+'_'+k]) for k in ('TP','FP','TN','FN')}
                endpoint_checks[method]={'historical_confusion_delta':delta,'exact_match':not any(delta.values())}
            parity['old_PA_endpoints']=endpoint_checks
    # Compare recovered integer scores to independently archived V20 A100 fragments where available.
    v20=ROOT/f"results/v20/mechanisms/analysis/OPS-SAT/{profile['variant']}/{profile['configuration']}/seed_{profile['seed']}/{format_name}/fragment_scores.npz"
    if v20.exists() and kind=='integer':
        with np.load(v20,allow_pickle=False) as z:
            errors={}
            for split in ('validation','test'):
                order=np.argsort(z[split+'_ids']);expected=z[split+'_scores'][order]
                np.testing.assert_array_equal(splits[split]['ids'],z[split+'_ids'][order])
                actual=splits[split]['p95']
                close=np.isclose(actual,expected,rtol=5e-4,atol=1e-7)
                errors[split]={'max_absolute_drift':float(np.max(np.abs(actual-expected))),
                    'rtol':5e-4,'atol':1e-7,'outside_tolerance':int((~close).sum()),
                    'exact_values':bool(np.array_equal(actual,expected))}
            parity['v20_GPU_fragment_parity']=errors
    finish(identity,profile,splits,sources,{'mode':'CPU_export_replay' if DEVICE=='cpu' else 'CUDA_export_replay','quantization_kind':kind,'alpha':alpha,
           'execution':('integer_INT8_times_INT8_accumulate_INT32_reference_CPU' if DEVICE=='cpu' else 'integer_CUDA_triton') if kind=='integer' else 'dequantized_FP32_'+DEVICE,
           'not_a_GPU_timing_measurement':True,
           'platform_note':'CPU reference recovery is a separate evaluation, not claimed identical to the A100 campaign. Quantization-boundary drift is retained explicitly.' if kind=='integer' else 'CPU FP32 replay checked against historical GPU score summaries.',
           'historical_numerical_checks':parity,'timings':timings})

def parent_validation(profile,alpha):
    path=ROOT/f"results/v13/architectures/OPS-SAT/{profile['variant']}/{profile['configuration']}/seed_{profile['seed']}/scores.npz"
    with np.load(path,allow_pickle=False) as z:
        if alpha is None:row=0
        else:
            indices=np.flatnonzero(np.isclose(z['alpha_values'],alpha,rtol=0,atol=1e-12))
            assert len(indices)==1;row=int(indices[0])
        a=aggregate(z['validation_scores'][row],z['validation_labels'],z['validation_segment_ids'])
    return a,verified(path)

def context_archive(path):
    r=read(path.parent/'metrics.json');profile=r['profile'];alpha=r['alpha']
    va,source=parent_validation(profile,alpha)
    sources=[verified(path),verified(path.parent/'metrics.json'),source]
    with np.load(path,allow_pickle=False) as z:
        for i,condition in enumerate(z['conditions']):
            condition=str(condition);identity='context:'+str(path.relative_to(ROOT))+':'+condition
            if cached(identity):continue
            te=aggregate(z['test_scores'][i],z['test_labels'],z['test_segment_ids'])
            finish(identity,{**profile,'input_condition':condition},{'test':te,'validation':va},sources,
                   {'mode':'stored_context_scores','alpha':alpha,'validation':'unperturbed parent; frozen calibration for test-only perturbation'})

def frozen_archive(folder):
    r=read(folder/'metrics.json');profile=r['profile'];metadata=folder/'metadata.npz'
    with np.load(metadata,allow_pickle=False) as z:
        labels=z['test_labels'].copy();ids=z['test_segment_ids'].copy()
    for condition in r['conditions']:
        name=condition['condition'];path=folder/(name+'.npz')
        for method,e in condition['methods'].items():
            if method not in ('bf','pot'):continue
            identity='frozen:'+str(path.relative_to(ROOT))+':'+method
            if cached(identity):continue
            va,parent=parent_validation(profile,e['alpha'])
            with np.load(path,allow_pickle=False) as z:te=aggregate(z[method+'_scores'],labels,ids)
            finish(identity,{**profile,'input_condition':name},{'test':te,'validation':va},
                   [verified(path),verified(metadata),verified(folder/'metrics.json'),parent],
                   {'mode':'stored_frozen_endpoint_scores','alpha':e['alpha'],'historical_method':method,
                    'validation':'same frozen parent alpha; unperturbed parent validation, not fitted to perturbed test'})

def inventory():
    rows=[]
    for version in sorted((ROOT/'results').iterdir()):
        if not version.is_dir() or version.name=='analysis_opssat_fragments':continue
        files=[p for p in version.rglob('*') if p.is_file() and ('ops-sat'in str(p).lower() or 'opssat'in p.name.lower())]
        counts=Counter(p.suffix for p in files)
        rows.append({'archive':version.name,'files':len(files),'checkpoints':counts['.ckpt'],'npz':counts['.npz'],'json':counts['.json']})
    csvfile(OUT/'inventory.csv',rows)
    return rows

def summarize():
    cases=[read(p) for p in sorted((OUT/'cases').glob('*.json'))];rows=[]
    for r in cases:
        p=r['profile'];family=('ablation_grid' if r['identity'].startswith('legacy:results/v10/') or r['identity'].startswith('legacy:results/v12/') else
            'architecture' if r['identity'].startswith('archive:results/v13/') else
            r['recovery'].get('quantization_kind','other'))
        if r['identity'].startswith('v20:'):
            family='integer_original_GPU' if '/analysis/'in r['identity'] else 'v20_diagnostics'
        for agg,met in r['metrics'].items():
            for protocol in ('validation_BF200','test_oracle_BF200','train_nominal_quantile_995'):
                if protocol not in met:continue
                v=met[protocol]
                rows.append({'identity':r['identity'],'family':family,'variant':p.get('variant','reference'),
                    'configuration':p.get('configuration',''),'format':p.get('format','fp32'),
                    'window':p['window_sz'],'ff':p['dim_feedforward'],'seed':p.get('seed'),
                    'epochs':p.get('epochs'),'aggregation':agg,'protocol':protocol,
                    'recovery_mode':r['recovery']['mode'],'AUC_score':met['AUC_score'],'AP_score':met['AP_score'],
                    **{k:v[k] for k in ('precision','recall','f1','AUC_bin','TP','FP','TN','FN','threshold')}})
    csvfile(OUT/'all_fragment_metrics.csv',rows)
    failures=[read(p) for p in (OUT/'errors').glob('*.json')] if (OUT/'errors').exists() else []
    write(OUT/'coverage.json',{'cases_completed':len(cases),'primary_validation_cases':sum(r['primary_available'] for r in cases),
          'metric_rows':len(rows),'failures':failures,'training_runs_added':0,
          'source_campaigns_unchanged':True,'data_sha256':DATA_HASH})
    print('RECOVERY SUMMARY',len(cases),'cases;',len(failures),'unresolved evaluation cases',flush=True)

def main():
    global OUT,DATA_HASH
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['all','archives','replay','diagnostics','summary','inventory','integer_probe'],default='all')
    ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--limit',type=int)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    global DEVICE
    DEVICE=args.device
    DATA_HASH=digest(ROOT/'data/raw/OPSSAT/segments.csv')
    if args.phase=='inventory':inventory();return
    if args.phase=='summary':summarize();return
    import torch
    torch.set_num_threads(1)
    inventory();tasks=[]
    if args.phase in ('all','archives'):
        for pattern in ('results/v13/architectures/OPS-SAT/**/scores.npz','results/v14/continuation/OPS-SAT/**/scores.npz',
                        'results/v15/train/OPS-SAT/**/scores_matched.npz','results/v15/train/OPS-SAT/**/scores_ordered_test.npz'):
            tasks.extend((standardized_archive,(p,)) for p in sorted(ROOT.glob(pattern)))
        tasks.extend((v20_existing,(p,)) for p in sorted((ROOT/'results/v20/mechanisms').glob('**/OPS-SAT/**/result.json'))
                     if (p.parent/'fragment_scores.npz').exists())
    if args.phase in ('all','replay'):
        for v in ('v6','v8','v9','v10','v12'):
            tasks.extend((replay_legacy,(p,)) for p in sorted((ROOT/'results'/v).rglob('*opssat*integrated*.json')))
        for kind,root in [('weight_only',ROOT/'results/v16/ptq/OPS-SAT'),('integer',ROOT/'results/v17/integer/OPS-SAT')]:
            for p in sorted(root.glob('*/*/seed_*/metrics.json')):
                for fmt in ('int8','int4'):
                    # Use the saved A100 V20 p95 arrays for seeds2026-28; replay only missing seeds.
                    profile=read(p)['profile']
                    existing=ROOT/f"results/v20/mechanisms/analysis/OPS-SAT/{profile['variant']}/{profile['configuration']}/seed_{profile['seed']}/{fmt}/fragment_scores.npz"
                    if kind=='integer' and existing.exists():continue
                    tasks.append((replay_quant,(p.parent,fmt,kind)))
    if args.phase=='integer_probe':
        tasks=[(replay_quant,(ROOT/'results/v17/integer/OPS-SAT/reference/reference/seed_2026','int8','integer')),
               (replay_quant,(ROOT/'results/v17/integer/OPS-SAT/reference/reference/seed_2026','int4','integer'))]
    if args.phase in ('all','diagnostics'):
        tasks.extend((context_archive,(p,)) for p in sorted((ROOT/'results/v14/context/OPS-SAT').glob('**/scores.npz')))
        tasks.extend((frozen_archive,(p.parent,)) for p in sorted((ROOT/'results/v15/frozen/OPS-SAT').glob('**/metrics.json')))
    if args.limit:tasks=tasks[:args.limit]
    for function,params in tqdm(tasks,desc='OPS-SAT recovery | NO TRAINING',unit='case'):
        label=function.__name__+':'+':'.join(str(p) for p in params)
        tqdm.write(label)
        error_path=OUT/'errors'/(hashlib.sha256(label.encode()).hexdigest()[:16]+'.json')
        try:
            function(*params)
            if error_path.exists():error_path.unlink()
        except Exception as exc:
            write(error_path,{'case':label,'reason':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()})
            tqdm.write('NEEDS FOLLOW-UP: '+str(exc)[:180])
    summarize()

if __name__=='__main__':main()
