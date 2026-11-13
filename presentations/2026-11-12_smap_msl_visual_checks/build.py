"""Real SMAP/MSL examples with original BF200/POT decisions, no new protocol."""
from pathlib import Path
from collections import defaultdict
import ast,csv,hashlib,json,os,sys,statistics as st
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/toast-matplotlib')
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tranad_plus'))
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tranad_plus.core.data import load_data
from tranad_plus.core.models import build_model
from TranADPlus.src.omnianomaly import adjust_predicts
torch.set_num_threads(1)
RED='#9B0014';INK='#202020';GRAY='#666666';FP='#D89428'
COLORS=[RED,'#286A8C','#986221','#626262'];STYLES=['-','--','-.',':']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
DS=('SMAP','MSL');SEEDS=(2026,2027,2028);ANCHOR=2026
WCONFIG=('reference','window_only','ff_only','compact')
ARCHS=('reference','shared_decoder','single_pass','encoder_only')
SPECS=[('reference',c) for c in WCONFIG]+[(v,'reference') for v in ARCHS[1:]]
metadata=list(csv.DictReader((ROOT/'data/raw/SMAP_MSL/labeled_anomalies.csv').open()))
catalog={};sources=[];label_checks=[];models={};bundles={};parities=[];all_examples=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def counts(y,p):
    y,p=np.asarray(y).astype(bool),np.asarray(p).astype(bool)
    tp,fp,tn,fn=[int(x.sum()) for x in (y&p,~y&p,~y&~p,y&~p)]
    return {'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            'FPR':fp/(fp+tn) if fp+tn else None}

def build_catalog(ds):
    base=ROOT/f'data/processed/tranad_plus/{ds}';tracks={};events=[]
    train=[np.load(p) for p in sorted((base/'Train/Tracks').glob('*.npy'))]
    scaler=MinMaxScaler().fit(np.concatenate(train))
    for p in sorted((base/'Test/Tracks').glob('*.npy')):
        raw=np.load(p);label=np.load(base/'Test/Labels'/p.name)
        y=(label.sum(axis=1)>0);matches=[r for r in metadata if r['chan_id']==p.stem and r['spacecraft']==ds]
        assert matches,p
        expected=np.zeros(len(raw),dtype=bool);spans=ast.literal_eval(matches[0]['anomaly_sequences'])
        classes=matches[0]['class'].strip('[]').replace(' ','').split(',')
        for start,end in spans:expected[start:end+1]=True
        np.testing.assert_array_equal(y,expected)
        assert np.all(label==label[:,0:1]),'Feature-label replication differs'
        norm=scaler.transform(raw)
        tracks[p.stem]={'raw':raw,'norm':norm,'labels':y,'path':str(p.relative_to(ROOT))}
        label_checks.append({'dataset':ds,'track':p.stem,'samples':len(y),'positive_samples':int(y.sum()),
           'matches_inclusive_metadata_intervals':True,'metadata_rows':len(matches),
           'metadata_policy':'first row, matching original preprocessing' if len(matches)>1 else 'unique row'})
        for j,(a,b) in enumerate(spans):
            events.append({'track':p.stem,'start':int(a),'stop':int(b)+1,'length':int(b-a+1),
                           'class':classes[j] if j<len(classes) else 'unknown'})
    catalog[ds]={'tracks':tracks,'events':events,'scaler':scaler}

def load_endpoint(ds,variant,config,seed):
    key=(ds,variant,config,seed)
    folder=ROOT/f'results/v13/architectures/{ds}/{variant}/{config}/seed_{seed}'
    r=json.loads((folder/'metrics.json').read_text());profile=r['profile']
    path=folder/'scores.npz';assert sha(path)==r['artifacts']['scores.npz']
    starts={};off=0
    for unit,t in catalog[ds]['tracks'].items():starts[unit]=off;off+=len(t['labels'])+profile['window_sz']-1
    with np.load(path,allow_pickle=False) as z:
        grid=z['test_scores'];labels=z['test_labels'].copy();alphas=z['alpha_values'].copy()
        assert np.array_equal(alphas,np.linspace(0,1,21)) if len(grid)>1 else len(alphas)==0
        real=np.concatenate([np.arange(a,a+len(catalog[ds]['tracks'][unit]['labels'])) for unit,a in starts.items()])
        expected=np.concatenate([t['labels'] for t in catalog[ds]['tracks'].values()])
        np.testing.assert_array_equal(labels[real],expected)
        endpoints={}
        for method in ('bf','pot'):
            e=r['paper_oracle'][method];alpha=e['alpha']
            row=0 if alpha is None else int(np.argmin(np.abs(alphas-alpha)))
            scores=grid[row].copy();threshold=float(e[method+'_threshold']);direct=scores>threshold
            pa=adjust_predicts(scores,labels.copy(),threshold).astype(bool)
            c=counts(labels,pa)
            for k in ('TP','FP','TN','FN'):assert c[k]==int(e[method+'_'+k]),(key,method,k)
            endpoints[method]={'scores':scores,'alpha':alpha,'threshold':threshold,'direct':direct,'PA':pa,
                'raw_metrics_full':counts(labels,direct),'PA_metrics_full':c,
                'raw_metrics_real':counts(labels[real],direct[real]),'PA_metrics_real':counts(labels[real],pa[real]),
                'AUC_score_full':float(roc_auc_score(labels,scores))}
        # Matched-alpha control: the POT threshold already recorded for the BF-selected alpha.
        v17=ROOT/f'results/v17/integer/{ds}/{variant}/{config}/seed_{seed}/metrics.json'
        original=json.loads(v17.read_text())['formats']['fp32']['oracle']
        bfalpha=endpoints['bf']['alpha'];records=original['alpha_grid']
        rec=next(x for x in records if x['alpha']==bfalpha)
        threshold=float(rec['pot_threshold']);scores=endpoints['bf']['scores']
        direct=scores>threshold;pa=adjust_predicts(scores,labels.copy(),threshold).astype(bool)
        cf={'alpha':bfalpha,'threshold':threshold,'direct':direct,'PA':pa,
            'raw_metrics_full':counts(labels,direct),'PA_metrics_full':counts(labels,pa)}
    value={'profile':profile,'report':r,'folder':folder,'starts':starts,'labels':labels,
           'endpoints':endpoints,'POT_at_BF_alpha':cf,'raw_indices':real}
    sources.append({'key':list(key),'report':str((folder/'metrics.json').relative_to(ROOT)),
        'report_sha256':sha(folder/'metrics.json'),'scores_sha256':r['artifacts']['scores.npz'],
        'matched_alpha_POT_source':str(v17.relative_to(ROOT)),'matched_alpha_POT_source_sha256':sha(v17)})
    return value

def clip(m,unit,a,b,method,kind='direct'):
    start=m['starts'][unit];return m['endpoints'][method][kind][start+a:start+b]

def select_examples(ds):
    method='bf' if ds=='SMAP' else 'pot';candidates=[]
    excluded={'G-7','E-4'} if ds=='SMAP' else {'P-11','M-6'}
    ref=lambda seed:models[ds,'reference','reference',seed]
    short=lambda seed:models[ds,'reference','window_only',seed]
    encoder=lambda seed:models[ds,'encoder_only','reference',seed]
    # Nominal blocks: both model families, median lexical candidate; preserve all candidates.
    for family,other in [('window',short),('architecture',encoder)]:
        found=[]
        for unit,t in catalog[ds]['tracks'].items():
            if unit in excluded or unit=='P-2':continue
            y=t['labels'];n=len(y)
            for a in range(0,n-159,160):
                b=a+160
                if y[a:b].any():continue
                differences=[int(clip(ref(s),unit,a,b,method).sum())-int(clip(other(s),unit,a,b,method).sum()) for s in SEEDS]
                hits=[int(clip(ref(s),unit,a,b,method).sum()) for s in SEEDS]
                if differences[0]>0 and hits[0]>0:
                    found.append({'dataset':ds,'family':family,'kind':'nominal_FP_reduction','track':unit,'start':a,'stop':b,
                        'length':160,'method':method,'votes':sum(v>0 for v in differences),
                        'anchor_FP_difference':differences[0],'paired_differences':differences})
        assert found,(ds,family,'no nominal FP controls')
        stable=[x for x in found if x['votes']>=2] or found
        stable.sort(key=lambda x:(x['track'],x['start']));chosen=stable[(len(stable)-1)//2].copy()
        chosen.update({'candidate_count':len(found),'stable_candidates':len(stable),'selection':'median lexical track/index among nominal FP-reduction blocks; anchor seed2026; prefer agreement in >=2/3 seeds'})
        candidates.extend(found);all_examples.append(chosen)
    # Positive intervals: window/direct-hit tradeoff and an architecture counterexample.
    event_kinds=['window_PA_same_raw_fewer','architecture_counterexample'] if ds=='SMAP' else ['BF_miss_POT_hit','architecture_counterexample']
    for kind in event_kinds:
        found=[]
        for event in catalog[ds]['events']:
            unit,a,b=event['track'],event['start'],event['stop']
            if unit in excluded or unit=='P-2' or not 15<=event['length']<=500:continue
            agreements=[];detail=[]
            for seed in SEEDS:
                if kind=='window_PA_same_raw_fewer':
                    first=clip(ref(seed),unit,a,b,method);second=clip(short(seed),unit,a,b,method)
                    qualifies=bool(first.any() and second.any() and first.sum()>second.sum())
                elif kind=='BF_miss_POT_hit':
                    first=clip(ref(seed),unit,a,b,'bf');second=clip(ref(seed),unit,a,b,'pot')
                    qualifies=bool(not first.any() and second.any())
                else:
                    first=clip(ref(seed),unit,a,b,method);second=clip(encoder(seed),unit,a,b,method)
                    # Keep an explicit counterexample: reference hits, encoder-only misses.
                    qualifies=bool(first.any() and not second.any())
                agreements.append(qualifies);detail.append([int(first.sum()),int(second.sum())])
            if agreements[0]:
                found.append({'dataset':ds,'family':'architecture' if kind=='architecture_counterexample' else 'window',
                    'kind':kind,**event,'method':method,'votes':sum(agreements),'paired_direct_TP':detail})
        if not found and kind=='architecture_counterexample':
            # If no full-event miss exists, compare fewer direct hits with unchanged event coverage.
            for event in catalog[ds]['events']:
                unit,a,b=event['track'],event['start'],event['stop']
                if unit in excluded or unit=='P-2' or not 15<=event['length']<=500:continue
                detail=[];votes=0
                for seed in SEEDS:
                    first=clip(ref(seed),unit,a,b,method);second=clip(encoder(seed),unit,a,b,method)
                    detail.append([int(first.sum()),int(second.sum())]);votes+=bool(first.any() and second.any() and first.sum()>second.sum())
                if detail[0][0]>detail[0][1]>0:
                    found.append({'dataset':ds,'family':'architecture','kind':'architecture_same_PA_fewer_direct',**event,
                        'method':method,'votes':votes,'paired_direct_TP':detail})
        assert found,(ds,kind,'no positive controls; inspect rather than fabricate')
        stable=[x for x in found if x['votes']>=2] or found
        stable.sort(key=lambda x:(x['length'],x['track'],x['start']));chosen=stable[(len(stable)-1)//2].copy()
        chosen.update({'candidate_count':len(found),'stable_candidates':len(stable),
            'selection':'median duration among declared outcome category; anchor seed2026; prefer agreement in >=2/3 seeds'})
        candidates.extend(found);all_examples.append(chosen)
    return candidates

def replay(m,idx,method):
    key=(m['profile']['dataset'],m['profile']['window_sz'])
    if key not in bundles:bundles[key]=load_data(m['profile'],ROOT/'data')
    bundle=bundles[key]
    np.testing.assert_array_equal(bundle.labels['test'],m['labels'])
    if 'model'not in m:
        p=m['folder']/'checkpoints/final.ckpt';assert sha(p)==m['report']['artifacts']['checkpoints/final.ckpt']
        saved=torch.load(p,map_location='cpu',weights_only=True);model=build_model(m['profile'])
        assert saved['profile']==m['profile'];model.load_state_dict(saved['model_state'],strict=True);m['model']=model.eval()
    targets=[];recs=[];alpha=m['endpoints'][method]['alpha']
    with torch.inference_mode():
        for a in range(0,len(idx),32):
            batch=[]
            for i in idx[a:a+32]:
                sample=bundle.datasets['test'][int(i)];batch.append(sample[0] if isinstance(sample,tuple) else sample)
            x=torch.stack(batch).permute(1,0,2);target=x[-1:];values=m['model'](x,target)
            reconstruction=values if not isinstance(values,tuple) else alpha*values[0]+(1-alpha)*values[1]
            recs.append(reconstruction[0].numpy());targets.append(target[0].numpy())
    rec,target=np.concatenate(recs),np.concatenate(targets);score=np.mean((rec-target)**2,axis=1)
    np.testing.assert_allclose(score,m['endpoints'][method]['scores'][idx],rtol=5e-4,atol=1e-7)
    parities.append({'profile':m['profile']['identifier'],'method':method,'points':len(idx),
                    'max_absolute_MSE_drift':float(np.max(np.abs(score-m['endpoints'][method]['scores'][idx])))})
    return rec,target

def plot_case(pdf,case,number,method,case_rows):
    ds,unit=case['dataset'],case['track'];t=catalog[ds]['tracks'][unit]
    a=max(0,case['start']-60);b=min(len(t['labels']),case['stop']+60)
    if case['kind']=='nominal_FP_reduction':a,b=case['start'],case['stop']
    specs=[('reference',c) for c in WCONFIG] if case['family']=='window' else [(v,'reference') for v in ARCHS]
    names=(['TranAD+','Smaller W','Smaller FF','Smaller W + FF'] if case['family']=='window' else ['TranAD+','One-decoder','One-pass','Encoder-only'])
    chosen=[models[ds,v,c,ANCHOR] for v,c in specs];xx=np.arange(a,b);y=t['labels'][a:b]
    recs=[];targets=[];eps=[];indices=[];local=[]
    arrays={'sample_index':xx,'labels':y}
    for m in chosen:
        idx=m['starts'][unit]+np.arange(a,b);rec,target=replay(m,idx,method)
        recs.append(rec);targets.append(target);eps.append(m['endpoints'][method]);indices.append(idx)
    for target in targets[1:]:np.testing.assert_array_equal(targets[0],target)
    target=targets[0];arrays['target_all_features']=target
    fig=plt.figure(figsize=(16,11.6));grid=fig.add_gridspec(3,1,left=.15,right=.90,top=.84,bottom=.14,
                      height_ratios=[2.7,3.1,2.6],hspace=.34)
    top=fig.add_subplot(grid[0]);sg=grid[1].subgridspec(4,1,hspace=.15)
    scores=[fig.add_subplot(sg[i],sharex=top) for i in range(4)];errors=fig.add_subplot(grid[2],sharex=top)
    methodname='BF200' if method=='bf' else 'POT'
    titles={'nominal_FP_reduction':'Nominal interval: false-alarm comparison',
            'window_PA_same_raw_fewer':'Anomalous interval: fewer direct hits can retain PA credit',
            'BF_miss_POT_hit':'Anomalous interval: threshold selection changes event coverage',
            'architecture_counterexample':'Counterexample: simpler architecture can miss an event',
            'architecture_same_PA_fewer_direct':'Anomalous interval: architecture changes direct coverage'}
    fig.text(.025,.97,f'{number}. {ds} / {unit} / {methodname}',color=RED,fontsize=20,va='top')
    fig.text(.025,.932,titles[case['kind']],fontsize=13)
    fig.text(.025,.903,f"Original saved alpha and threshold for each model | seed2026 | selection category agrees in {case['votes']}/3 seeds ({case['method'].upper()})",fontsize=9,color=GRAY)
    top.plot(xx,target[:,0],color=INK,lw=1.6,zorder=6)
    handles=[Line2D([0],[0],color=INK,lw=1.8,label='Observed telemetry')]
    for j,(m,rec,c,style,name) in enumerate(zip(chosen,recs,COLORS,STYLES,names)):
        top.plot(xx,rec[:,0],color=c,ls=style,lw=1.2)
        p=m['profile'];handles.append(Line2D([0],[0],color=c,ls=style,label=f"{name} (W{p['window_sz']}/FF{p['dim_feedforward']})"))
    fig.legend(handles=handles,ncol=5,fontsize=8,loc='upper left',bbox_to_anchor=(.025,.884),frameon=False)
    top.set_title('1  Telemetry coordinate 0 and reconstruction',loc='left',fontsize=11)
    top.set_ylabel('Normalized value')
    matrix=[];labels=[];shares=[]
    values=np.concatenate([ep['scores'][idx] for ep,idx in zip(eps,indices)]+[np.array([e['threshold'] for e in eps])])
    positive=values[values>0];lo=float(positive.min()*.5);hi=float(positive.max()*2)
    for j,(m,rec,ep,idx,ax,name) in enumerate(zip(chosen,recs,eps,indices,scores,names)):
        ax.plot(xx,np.maximum(ep['scores'][idx],lo),color=COLORS[j],lw=1.1)
        ax.axhline(ep['threshold'],color=INK,ls='--',lw=1)
        ax.set_yscale('log');ax.set_ylim(lo,hi);ax.set_ylabel(name,rotation=0,ha='right',va='center',fontsize=8,labelpad=12)
        ax.tick_params(axis='y',labelsize=7);ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10,numticks=3))
        alpha='n/a' if ep['alpha'] is None else f"{ep['alpha']:.2f}"
        ax.text(.995,.95,f"alpha {alpha} | threshold {ep['threshold']:.4g}",transform=ax.transAxes,ha='right',va='top',fontsize=7.5,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1})
        for kind in ['direct','PA']:
            pred=ep[kind][idx];error=np.zeros(len(y),dtype=np.int8);error[~y&pred]=1;error[y&~pred]=2
            matrix.append(error);labels.append(name+(' / no PA' if kind=='direct' else ' / PA'))
        raw,pa=counts(y,ep['direct'][idx]),counts(y,ep['PA'][idx]);err=(rec-target)**2
        share=float(err[:,0].sum()/err.sum()) if err.sum() else 0;shares.append(share)
        item={'example':number,'dataset':ds,'track':unit,'method':method,'family':case['family'],
             'model':name,'profile':m['profile']['identifier'],'alpha':ep['alpha'],'threshold':ep['threshold'],
             'display_start':a,'display_stop':b,'annotated_event_start':case['start'],'annotated_event_stop':case['stop'],
             'positive_points_in_crop':int(y.sum()),'first_coordinate_MSE_fraction':share,
             **{'raw_'+k:v for k,v in raw.items()},**{'PA_'+k:v for k,v in pa.items()}}
        local.append(item);case_rows.append(item)
        arrays.update({f'model{j}_reconstruction':rec,f'model{j}_score':ep['scores'][idx],
                       f'model{j}_direct':ep['direct'][idx],f'model{j}_PA':ep['PA'][idx]})
    spans=np.diff(np.r_[False,y,False].astype(int))
    for ax in [top]+scores:
        for l,r in zip(np.flatnonzero(spans==1),np.flatnonzero(spans==-1)):
            ax.axvspan(a+l-.5,a+r-.5,color='#F5E8EA',lw=0,zorder=0)
        ax.set_xlim(a-.5,b-.5);ax.grid(axis='y',color='#E5E5E5',lw=.5);ax.tick_params(labelbottom=False)
    scores[0].set_title('2  Mean squared error over all input features; dashed line = selected threshold',loc='left',fontsize=11,pad=10)
    errors.imshow(np.array(matrix),origin='upper',aspect='auto',interpolation='nearest',vmin=0,vmax=2,
                  cmap=ListedColormap(['white',FP,RED]),extent=(a-.5,b-.5,7.5,-.5))
    errors.set_yticks(range(8),labels,fontsize=8);errors.set_xlim(a-.5,b-.5)
    for r in (1.5,3.5,5.5):errors.axhline(r,color='#BBBBBB',lw=.8)
    errors.set_title('3  Prediction errors before and after PA',loc='left',fontsize=11,pad=10)
    for row in range(8):
        item=local[row//2];kind='raw' if row%2==0 else 'PA'
        errors.text(1.01,row,f"FP {item[kind+'_FP']:3d}  FN {item[kind+'_FN']:3d}",
              transform=matplotlib.transforms.blended_transform_factory(errors.transAxes,errors.transData),va='center',fontsize=8,family='monospace')
    errors.set_xlabel('Sample index within the original track')
    errors.legend(handles=[Patch(facecolor=FP,label='False positive'),Patch(facecolor=RED,label='False negative'),
                           Patch(facecolor='white',edgecolor='#BBBBBB',label='Correct prediction')],
                  ncol=3,loc='upper left',bbox_to_anchor=(-.16,-.19),fontsize=9,frameon=False)
    fig.text(.025,.060,'Pale shading: supplied positive labels. PA is applied on the full original stream before cropping; false positives remain unchanged.',fontsize=9)
    fig.text(.025,.038,'Top shows one telemetry coordinate; score uses all inputs. Coordinate-0 share of displayed MSE: '+', '.join(f'{100*x:.1f}%' for x in shares)+'.',fontsize=9,color=GRAY)
    fig.text(.025,.017,'Illustrative, outcome-selected examples; candidate lists and selection rules are saved. Anonymous channel IDs do not establish a physical fault cause.',fontsize=8,color=GRAY)
    stem=f'{number:02}_{ds.lower()}_{unit}_{method}'
    fig.savefig(OUT/(stem+'.pdf'));fig.savefig(OUT/(stem+'.png'),dpi=180);pdf.savefig(fig);plt.close(fig)
    np.savez_compressed(OUT/(stem+'.npz'),**arrays)

def main():
    for ds in DS:build_catalog(ds)
    for ds in DS:
        for seed in SEEDS:
            for variant,config in SPECS:
                print('SCAN',ds,variant,config,seed,flush=True)
                models[ds,variant,config,seed]=load_endpoint(ds,variant,config,seed)
    candidates=[]
    for ds in DS:candidates.extend(select_examples(ds))
    (OUT/'selection.json').write_text(json.dumps({'examples':all_examples,'candidates':candidates},indent=2)+'\n')
    write('label_checks.csv',label_checks)
    scalar=[];threshold_controls=[]
    for key,m in models.items():
        ds,variant,config,seed=key
        for method,ep in m['endpoints'].items():
            row={'dataset':ds,'variant':variant,'configuration':config,'seed':seed,'method':method,
                 'alpha':ep['alpha'],'threshold':ep['threshold'],'window':m['profile']['window_sz'],'ff':m['profile']['dim_feedforward'],
                 'AUC_score_full':ep['AUC_score_full']}
            row.update({'raw_'+k:v for k,v in ep['raw_metrics_full'].items()})
            row.update({'PA_'+k:v for k,v in ep['PA_metrics_full'].items()})
            scalar.append(row)
        if ds=='MSL':
            bf=m['endpoints']['bf'];cf=m['POT_at_BF_alpha'];s=bf['scores'];grid=np.linspace(float(s.min()),float(s.mean()+3*s.std()),200)
            for name,e in [('original_BF200',bf),('POT_threshold_at_same_BF_alpha',cf)]:
                threshold_controls.append({'variant':variant,'configuration':config,'seed':seed,'endpoint':name,
                    'alpha':e['alpha'],'threshold':e['threshold'],'BF_grid_step':float(grid[1]-grid[0]),
                    'score_median':float(np.median(s)),'score_q99':float(np.quantile(s,.99)),'score_max':float(s.max()),
                    'AUC_score_same_scores':bf['AUC_score_full'],
                    **{'raw_'+k:v for k,v in e['raw_metrics_full'].items()},**{'PA_'+k:v for k,v in e['PA_metrics_full'].items()}})
    write('original_endpoint_metrics.csv',scalar);write('MSL_matched_alpha_threshold_control.csv',threshold_controls)
    case_rows=[]
    with PdfPages(OUT/'SMAP_MSL_visual_checks.pdf') as pdf:
        for number,case in enumerate(all_examples,1):
            for method in ('bf','pot'):
                print('PLOT',number,case['dataset'],case['track'],case['kind'],method,flush=True)
                plot_case(pdf,case,number,method,case_rows)
    write('example_counts.csv',case_rows)
    (OUT/'provenance.json').write_text(json.dumps({'sources':sources,'label_checks':len(label_checks),
        'reconstruction_parity_checks':parities,'anchor_seed':ANCHOR,'seeds_used_for_selection':list(SEEDS),
        'new_training':0,'original_alpha_and_threshold_preserved':True,'new_threshold_protocol':False,
        'score_uses_all_features':True,'first_coordinate_only_in_telemetry_panel':True,
        'source_metadata_sha256':sha(ROOT/'data/raw/SMAP_MSL/labeled_anomalies.csv'),
        'script_sha256':sha(__file__),'original_results_changed':False},indent=2)+'\n')
    print('COMPLETE',OUT/'SMAP_MSL_visual_checks.pdf',flush=True)

if __name__=='__main__':main()
