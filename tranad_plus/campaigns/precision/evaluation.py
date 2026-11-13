"""Shared BF200/POT evaluation, parent-locked controls, measured storage and probes."""
from pathlib import Path
import json
import time
import warnings

import numpy as np
import torch

from tranad_plus.core.common import save_npz, write_json, hash_file
from tranad_plus.core.evaluation.scoring import collect_outputs, reconstruction_scores
from tranad_plus.core.evaluation.metrics import fixed_metrics
from tranad_plus.campaigns.temporal.evaluation import oracle
from opssat_pipeline import point_metrics,adjust_per_fragment
from TranADPlus.src.omnianomaly import adjust_predicts
from .progress import announce, bar, record, scoring, metric_bars
from .resources import measure


def verify_parent_endpoint(scores,old,labels,threshold,method,groups,original,current):
    """Distinguish numerical boundary flips from a changed metric protocol."""
    np.testing.assert_allclose(scores,old,rtol=5e-4,atol=1e-7)
    reference=fixed_metrics(old,labels,threshold,method,groups)
    keys=('TP','TN','FP','FN')
    if any(reference[method+'_'+key]!=original[method+'_'+key] for key in keys):
        raise RuntimeError('Stored parent counts cannot be reproduced from stored scores: metric protocol mismatch')
    flips=(scores>threshold)!=(old>threshold)
    tolerance=1e-7+5e-4*np.abs(old)
    if np.any(np.abs(old[flips]-threshold)>tolerance[flips]):
        raise RuntimeError('FP32 alarm difference exceeds numerical score tolerance')
    return {'scores_within_tolerance':True,'rtol':5e-4,'atol':1e-7,
            'stored_parent_counts_reproduced':True,
            'exact_confusion_counts_match':all(current[method+'_'+k]==original[method+'_'+k] for k in keys),
            'direct_alarm_boundary_flips':int(flips.sum()),
            'max_absolute_score_drift':float(np.abs(scores-old).max()),
            'parent_counts':{k:original[method+'_'+k] for k in keys},
            'current_counts':{k:current[method+'_'+k] for k in keys},
            'comparison_baseline':'This campaign FP32; parent-locked boundary differences remain explicitly reported.'}


def probe_inputs(window):
    """Equal current/past mean and equal past RMS across nonconstant shapes."""
    n=window-1
    if n<2:raise ValueError('Controlled shape probe requires window>=3')
    ramp=np.linspace(-1,1,n)
    step=np.r_[-np.ones(n//2),np.ones(n-n//2)]
    pulse=np.zeros(n);pulse[n//2]=1
    templates={'ramp':ramp,'reverse_ramp':ramp[::-1],'step':step,'pulse':pulse}
    windows,rows=[],[]
    for current in (.2,.35,.5,.65,.8):
        windows.append(np.full((window,1),current,np.float32))
        rows.append({'current':current,'rms':0.,'shape':'constant'})
        for amplitude in (.025,.05):
            for name,shape in templates.items():
                centered=shape-shape.mean();centered=centered/np.sqrt(np.mean(centered**2))
                values=np.r_[current+amplitude*centered,current].astype(np.float32)
                windows.append(values[:,None]);rows.append({'current':current,'rms':amplitude,'shape':name})
    return np.stack(windows),rows


def probes(model, profile, device, parent_endpoints, path):
    values,rows=probe_inputs(profile['window_sz'])
    class Dataset(torch.utils.data.Dataset):
        windows=torch.from_numpy(values)
        def __len__(self):return len(self.windows)
        def __getitem__(self,i):return self.windows[i]
    outputs=collect_outputs(model,Dataset(),device,256,'OPS-SAT controlled windows',path.parent/'nonfinite.json')
    arrays={'windows':values,'current':np.array([r['current'] for r in rows]),
            'past_rms':np.array([r['rms'] for r in rows]),'shape':np.array([r['shape'] for r in rows])}
    for key in ('bf','pot'):
        arrays[key+'_scores']=reconstruction_scores(outputs,parent_endpoints[key]['alpha'])
    save_npz(path,**arrays)
    return {'cases':len(rows),'file':path.name,
            'meaning':'Controlled response to normalized synthetic shapes; no fault labels or physical-cause claim',
            'invariants':'Same current and past mean; equal past RMS across nonconstant shapes at each amplitude',
            'alpha':'frozen parent BF/POT selections','uses_test_labels_to_define_shapes':False}


def evaluate(model,bundle,profile,folder,device,settings,parent_report,parent_scores,data_root,storage,progress_root=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    progress_root=Path(progress_root) if progress_root is not None else folder.parents[1]
    done=folder/'result.json'
    if done.exists():
        previous=json.loads(done.read_text())
        for name,expected in previous['artifacts'].items():
            if hash_file(folder/name)!=expected:raise RuntimeError('Changed format artifact: '+str(folder/name))
        return previous
    format_name=storage['format']
    started=time.perf_counter()
    announce(f"    EVALUATE {format_name.upper()} | inference -> BF200/POT -> locked thresholds -> resources")
    if settings.get('save_pointwise_arrays',False) or settings.get('save_reconstructions',False):
        raise ValueError('V16 compact does not archive pointwise arrays or reconstructions')
    grids={};inference={};batch=settings['score_batch_size_by_dataset'][profile['dataset']]
    alphas=np.linspace(0,1,21).tolist() if model.output_count==2 else [None]
    # Native sets have no validation. OPS validation statistics are recorded;
    # outputs/targets and all-alpha grids stay in memory only, never on disk.
    with bar(total=len(bundle.datasets),desc=f'{format_name.upper()} | score splits',position=1,unit='split',leave=False) as phases:
        for split,ds in bundle.datasets.items():
            record(progress_root,'evaluating',format=format_name,split=split,
                   detail_folder=str(folder.relative_to(progress_root)))
            with scoring(format_name):
                value=collect_outputs(model,ds,device,batch,f'{format_name.upper()} | {split} batches',folder/'nonfinite.json')
            grids[split]=np.stack([reconstruction_scores(value,a) for a in alphas])
            inference[split]=value['performance']
            inference[split]['points_per_second']=len(ds)/value['performance']['seconds_including_transfer']
            del value
            phases.update(1)
    if format_name=='fp32' and storage.get('check_parent_parity'):
        np.testing.assert_allclose(grids['test'],parent_scores,rtol=5e-4,atol=1e-7)
    inference_done=time.perf_counter()
    warnings_list=[]
    with metric_bars(),warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        selected=oracle(grids['train'],grids['test'],bundle.labels['test'],alphas,profile['pot_q'],
                        bundle.groups.get('test'),folder,f'{format_name.upper()} | BF200 + POT')
        warnings_list=sorted(set(str(w.message) for w in recorded))
    search_done=time.perf_counter()
    labels=bundle.labels['test'];groups=bundle.groups.get('test');locked={};drift={};predictions={};verification={}
    def adjusted(scores,threshold):
        return (adjust_per_fragment(scores,labels,groups,threshold) if groups else
                adjust_predicts(scores,labels.copy(),threshold)).astype(np.uint8)
    for method in ('bf','pot'):
        endpoint=selected[method]
        if endpoint.get(method+'_f1') is not None:
            row=alphas.index(endpoint['alpha'])
            endpoint['raw_without_PA']=point_metrics(labels,grids['test'][row]>endpoint[method+'_threshold'],method)
            predictions['oracle_'+method+'_direct']=(grids['test'][row]>endpoint[method+'_threshold']).astype(np.uint8)
            predictions['oracle_'+method+'_adjusted']=adjusted(grids['test'][row],endpoint[method+'_threshold'])
        if parent_report:
            original=parent_report['paper_oracle'][method]
            if original.get(method+'_f1') is None:continue
            row=alphas.index(original['alpha']);scores=grids['test'][row]
            locked[method]={**fixed_metrics(scores,labels,original[method+'_threshold'],method,groups),
                            'alpha':original['alpha'],'locked_to_parent':True,
                            'raw_without_PA':point_metrics(labels,scores>original[method+'_threshold'],method)}
            old=parent_scores[row]
            predictions['locked_'+method+'_direct']=(scores>original[method+'_threshold']).astype(np.uint8)
            predictions['locked_'+method+'_adjusted']=adjusted(scores,original[method+'_threshold'])
            drift[method]={'mean_absolute_score_change':float(np.abs(scores-old).mean()),
                           'max_absolute_score_change':float(np.abs(scores-old).max()),
                           'direct_alarm_flip_fraction':float(np.mean((scores>original[method+'_threshold'])!=(old>original[method+'_threshold']))),
                           'PA_alarm_flip_fraction':float(np.mean(predictions['locked_'+method+'_adjusted']!=adjusted(old,original[method+'_threshold'])))}
            if format_name=='fp32' and storage.get('check_parent_parity'):
                verification[method]=verify_parent_endpoint(scores,old,labels,original[method+'_threshold'],
                                                             method,groups,original,locked[method])
                if not verification[method]['exact_confusion_counts_match']:
                    announce(f"    FP32 | {method.upper()} parent boundary drift: "
                             f"{verification[method]['direct_alarm_boundary_flips']} direct alarms; "
                             'stored counts verified; current FP32 is the paired baseline')
    # Confusion counts and scalar metrics retain predictions' useful summaries.
    # Full predictions are reproducible from the exported model and pinned data.
    del predictions
    record(progress_root,'resource_measurement',format=format_name)
    resources=measure(model,bundle.datasets['test'],device,settings,storage['execution'],profile,folder)
    physical=None
    if profile['dataset']=='OPS-SAT' and parent_report:
        record(progress_root,'controlled_windows',format=format_name,cases=45)
        with scoring('probes'):
            physical=probes(model,profile,device,parent_report['paper_oracle'],folder/'controlled_windows.npz')
    artifacts={p.name:hash_file(p) for p in folder.iterdir() if p.suffix=='.npz' or p.name=='operators.json'}
    result={'archive_policy':'compact_summary_only',
            'not_archived':['pointwise_scores','all_alpha_score_arrays','reconstructions','targets','predictions','duplicated_labels'],
            'regeneration':'Re-run evaluation from the saved exported model and the data/source manifest.',
            'oracle':selected,'parent_locked':locked,'score_drift':drift,'parent_verification':verification,'storage':storage,
            'resources':resources,'controlled_windows':physical,'fit_warnings':warnings_list,'artifacts':artifacts,
            'inference_by_split':inference,
            'score_statistics_by_split':{split:[{'alpha':a,'mean':float(s.mean()),'std':float(s.std()),
                   'min':float(s.min()),'max':float(s.max()),'quantiles_50_90_99_999':np.quantile(s,[.5,.9,.99,.999]).tolist()}
                    for a,s in zip(alphas,grid)] for split,grid in grids.items()},
            'phase_seconds':{'inference_scores_and_array_save':inference_done-started,
                             'BF200_POT_search':search_done-inference_done,
                             'post_search_controls_resources_probes':time.perf_counter()-search_done,
                             'total_before_report_write':time.perf_counter()-started}}
    write_json(done,result)
    announce(f"    DONE {format_name.upper()} | BF={selected['bf']['bf_f1']:.4f} | POT="+
             ('NA' if selected['pot'].get('pot_f1') is None else f"{selected['pot']['pot_f1']:.4f}"))
    return result
