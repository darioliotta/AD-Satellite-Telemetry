"""Controlled visual W/FF comparison: fixed alpha, original BF200/POT rules."""
from pathlib import Path
import csv, hashlib, json, os, sys
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/toast-matplotlib')
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tranad_plus'))
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from core.data import load_data
from core.models import build_model
from core.evaluation.metrics import fixed_metrics, bounded_pot
from paper_metrics import _stats
from core.evaluation.scoring import reconstruction_scores
from opssat_pipeline import adjust_per_fragment
from TranADPlus.src.omnianomaly import adjust_predicts
torch.set_num_threads(1)
RED='#9B0014';GRAY='#777777';BLACK='#202020'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

cache={};sources=[];results=[];candidate_notes=[];parities=[]
def model_data(ds,variant,configuration,seed):
    key=(ds,variant,configuration,seed)
    if key in cache:return cache[key]
    folder=ROOT/f'results/v13/architectures/{ds}/{variant}/{configuration}/seed_{seed}'
    report=json.loads((folder/'metrics.json').read_text());profile=report['profile']
    for name in ('scores.npz','checkpoints/final.ckpt','config.json','manifest.json'):
        assert sha(folder/name)==report['artifacts'][name]
    bundle=load_data(profile,ROOT/'data')
    with np.load(folder/'scores.npz') as z:
        score=z['test_scores'].copy();train_score=z['train_scores'][10].copy();labels=z['test_labels'].copy();alphas=z['alpha_values'].copy()
    assert np.array_equal(labels,bundle.labels['test'])
    state=torch.load(folder/'checkpoints/final.ckpt',map_location='cpu',weights_only=True)
    assert state['profile']==profile
    model=build_model(profile);model.load_state_dict(state['model_state'],strict=True);model.eval()
    starts={};offset=0
    if ds=='OPS-SAT':
        for group in bundle.groups['test']:
            starts[str(group['id'])]=(offset,len(group['values']))
            offset+=len(group['values'])
    else:
        for path in sorted((ROOT/f'data/processed/tranad_plus/{ds}/Test/Tracks').glob('*.npy')):
            n=len(np.load(path,mmap_mode='r'));starts[path.stem]=(offset,n)
            offset+=n+profile['window_sz']-1
    value={'report':report,'profile':profile,'model':model,'bundle':bundle,
           'scores':score,'train_scores':train_score,'labels':labels,'alphas':alphas,'starts':starts,'folder':folder,'endpoints':{}}
    cache[key]=value
    sources.append({'model':list(key),'metrics_file':str((folder/'metrics.json').relative_to(ROOT)),
                    'metrics_sha256':sha(folder/'metrics.json'),'checkpoint_sha256':sha(folder/'checkpoints/final.ckpt'),
                    'scores_sha256':sha(folder/'scores.npz'),'input_inventory':bundle.inventory})
    return value

def endpoint(value,method):
    if method in value['endpoints']:return value['endpoints'][method]
    alpha=.5;index=int(np.argmin(np.abs(value['alphas']-alpha)))
    assert index==10 and value['alphas'][index]==.5
    scores=value['scores'][index];y=value['labels'];groups=value['bundle'].groups.get('test')
    if method=='bf':
        thresholds=np.linspace(float(scores.min()),float(scores.mean()+3*scores.std()),200)
        boundaries=np.cumsum([len(g['values']) for g in groups])[:-1] if groups else None
        stats=_stats(scores,y,thresholds,boundaries=boundaries,preserve_upstream_index_zero=groups is None)
        threshold=float(thresholds[int(np.argmax(stats['f1']))])
        chosen=fixed_metrics(scores,y,threshold,'bf',groups)
    else:
        np.random.seed(2026)
        chosen=bounded_pot(value['train_scores'],scores,y,value['profile']['pot_q'])
        assert chosen['status']=='measured',chosen
        threshold=chosen['pot_threshold']
        chosen=fixed_metrics(scores,y,threshold,'pot',groups)
    saved=value['report']['paper_oracle']['alpha_grid'][index]
    assert abs(chosen[method+'_f1']-saved[method+'_f1'])<1e-6,(value['profile']['identifier'],method,chosen,saved)
    direct=scores>threshold
    pa=(adjust_per_fragment(scores,y,groups,threshold) if groups else adjust_predicts(scores,y.copy(),threshold)).astype(bool)
    value['endpoints'][method]={'scores':scores,'threshold':threshold,'alpha':alpha,'direct':direct,'pa':pa,'chosen':chosen}
    return value['endpoints'][method]

def replay(value,indices,method):
    dataset=value['bundle'].datasets['test'];rec=[];target=[]
    ep=endpoint(value,method)
    with torch.inference_mode():
        for start in range(0,len(indices),32):
            batch=[]
            for i in indices[start:start+32]:
                sample=dataset[int(i)]
                batch.append(sample[0] if isinstance(sample,tuple) else sample)
            x=torch.stack(batch).permute(1,0,2);y=x[-1:]
            outputs=value['model'](x,y);outputs=outputs if isinstance(outputs,tuple) else (outputs,)
            mixed=outputs[0] if ep['alpha'] is None else ep['alpha']*outputs[0]+(1-ep['alpha'])*outputs[1]
            rec.append(mixed[0].numpy());target.append(y[0].numpy())
    rec=np.concatenate(rec);target=np.concatenate(target)
    actual=np.mean((rec-target)**2,axis=1);saved=ep['scores'][indices]
    np.testing.assert_allclose(actual,saved,rtol=5e-4,atol=1e-7)
    parities.append({'profile':value['profile']['identifier'],'method':method,'points':len(indices),
                     'max_absolute_score_drift':float(np.max(np.abs(actual-saved)))})
    return rec,target

def raw_aligned(value):
    # Drop only synthetic target rows; return real samples in raw track order.
    idx=np.concatenate([np.arange(a,a+n) for a,n in value['starts'].values()])
    return idx,value['labels'][idx]

# All examples retain the same real intervals as the preceding visual review.
# Their prior selection is documented; they are not a representative sample.
CASES=[('OPS-SAT','239',0,445,'Anomalous fragment'),
       ('OPS-SAT','1560',0,38,'Anomalous fragment'),
       ('OPS-SAT','1281',0,48,'Nominal fragment'),
       ('SMAP','G-7',7480,7756,'Labelled anomalous interval'),
       ('SMAP','E-4',700,800,'Nominal interval'),
       ('MSL','P-11',1158,1425,'Labelled anomalous interval'),
       ('MSL','M-6',1763,1964,'Extreme-value diagnostic; not a typical case')]
CONFIGS=['reference','window_only','ff_only','compact']
COLORS=['#9B0014','#286A8C','#976224','#626262']
STYLES=['-','--','-.',':']
FP='#D89428';FN='#9B0014';SHADE='#F5E8EA'
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
all_rows=[];figures=[]

def counts(y,p):
    return {'TP':int(np.sum(y&p)),'FP':int(np.sum(~y&p)),
            'FN':int(np.sum(y&~p)),'TN':int(np.sum(~y&~p))}

with PdfPages(OUT/'Visual_checks.pdf') as pdf:
 for number,(ds,unit,a,b,description) in enumerate(CASES,1):
    print(f'[{number}/{len(CASES)}] {ds} {unit}: four reference-architecture models, seed 2026',flush=True)
    models=[model_data(ds,'reference',c,2026) for c in CONFIGS]
    # All fields except the declared window/FF/configuration/identifier match.
    excluded={'window_sz','dim_feedforward','configuration','identifier'}
    common={k:v for k,v in models[0]['profile'].items() if k not in excluded}
    for m in models:assert {k:v for k,v in m['profile'].items() if k not in excluded}==common
    recs=[];targets=[];indices=[];eps=[];names=[];shortnames=[]
    for j,m in enumerate(models):
        start,n=m['starts'][unit];assert 0<=a<b<=n
        idx=start+np.arange(a,b);indices.append(idx)
        rec,target=replay(m,idx,'bf');recs.append(rec);targets.append(target)
        eps.append({method:endpoint(m,method) for method in ('pot','bf')})
        p=m['profile'];shortnames.append(['Reference','Small W','Small FF','Small W + FF'][j])
        names.append(f"{shortnames[-1]}  (W{p['window_sz']}, FF{p['dim_feedforward']})")
    for target,m,idx in zip(targets,models,indices):
        np.testing.assert_array_equal(target,targets[0]);np.testing.assert_array_equal(m['labels'][idx],models[0]['labels'][indices[0]])
    target=targets[0];y=models[0]['labels'][indices[0]].astype(bool);xx=np.arange(a,b)
    stem=f'{number:02}_{ds.lower().replace("-","")}_{unit}'
    arrays={'sample_index':xx,'labels':y,'target_all_features':target}
    fig=plt.figure(figsize=(16,13))
    outer=fig.add_gridspec(3,1,left=.155,right=.985,top=.885,bottom=.125,
                           height_ratios=[2.6,3.1,3.8],hspace=.31)
    top=fig.add_subplot(outer[0]);scoregrid=outer[1].subgridspec(4,1,hspace=.13)
    scoreaxes=[fig.add_subplot(scoregrid[i],sharex=top) for i in range(4)]
    bottom=fig.add_subplot(outer[2],sharex=top)
    fig.text(.025,.972,f'{ds}  |  {unit}  |  {description}',fontsize=19,color=RED,va='top')
    fig.text(.025,.938,'TranAD+ architecture throughout  |  seed 2026  |  50 epochs  |  fixed alpha = 0.5',fontsize=11)
    handles=[Line2D([0],[0],color=BLACK,lw=1.8,label='Observed telemetry')]
    handles += [Line2D([0],[0],color=c,ls=s,lw=1.5,label=n) for c,s,n in zip(COLORS,STYLES,names)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.025,.927),ncol=5,fontsize=9,frameon=False,columnspacing=1.5)
    spans=np.diff(np.r_[False,y,False].astype(int))
    for axis in [top]+scoreaxes:
        for l,r in zip(np.flatnonzero(spans==1),np.flatnonzero(spans==-1)):
            axis.axvspan(a+l-.5,a+r-.5,color=SHADE,lw=0,zorder=0)
        axis.set_xlim(a-.5,b-.5);axis.grid(axis='y',color='#E4E4E4',lw=.6)
        axis.tick_params(labelbottom=False)
    top.plot(xx,target[:,0],color=BLACK,lw=1.65,zorder=5)
    for rec,c,s in zip(recs,COLORS,STYLES):top.plot(xx,rec[:,0],color=c,ls=s,lw=1.3)
    top.set_ylabel('Normalized telemetry\nand reconstruction',fontsize=10)
    top.set_title('1  Observed telemetry and four reconstructions',loc='left',fontsize=12,pad=9)
    scoreaxes[0].set_title('2  Anomaly scores and thresholds  |  solid black: POT; dashed black: BF200',loc='left',fontsize=12,pad=10)
    # Identical log-score limits within each example, across all four models.
    allvals=np.concatenate([e['bf']['scores'][idx] for e,idx in zip(eps,indices)]+
                           [np.array([e[t]['threshold'] for t in ('pot','bf')]) for e in eps])
    assert np.all(allvals>0),'Log-scale requires positive displayed scores and thresholds'
    lim=(float(allvals.min()*.65),float(allvals.max()*1.7))
    matrix=[];rowlabels=[];local_rows=[];contributions=[]
    for j,(m,idx,e,rec,ax) in enumerate(zip(models,indices,eps,recs,scoreaxes)):
        ax.plot(xx,e['bf']['scores'][idx],color=COLORS[j],lw=1.1)
        ax.set_yscale('log');ax.set_ylim(*lim)
        ax.set_ylabel(shortnames[j],rotation=0,ha='right',va='center',fontsize=9,labelpad=12,color=COLORS[j])
        ax.tick_params(axis='y',labelsize=7);ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10,numticks=3))
        for method,ls in [('pot','-'),('bf','--')]:
            ep=e[method];threshold=ep['threshold'];ax.axhline(threshold,color=BLACK,ls=ls,lw=1.1)
            arrays[f'{CONFIGS[j]}_{method}_threshold']=np.array(threshold)
            direct=ep['direct'][idx];pa=ep['pa'][idx]
            assert np.array_equal(direct[~y],pa[~y])
            assert not np.any(y&direct&~pa)
            cr=counts(y,direct);cp=counts(y,pa)
            for suffix,pred in [('raw',direct),('PA',pa)]:
                error=np.zeros(len(y),dtype=np.int8);error[~y&pred]=1;error[y&~pred]=2
                matrix.append(error);rowlabels.append(f"{shortnames[j]} · {'BF200' if method=='bf' else 'POT'} · {'no PA' if suffix=='raw' else 'PA'}")
                arrays[f'{CONFIGS[j]}_{method}_{suffix}_predictions']=pred
            item={'dataset':ds,'case':number,'unit':unit,'start':a,'stop':b,'configuration':CONFIGS[j],
                  'seed':2026,'alpha':.5,'window':m['profile']['window_sz'],'ff':m['profile']['dim_feedforward'],
                  'method':method,'threshold':threshold,'labelled_positive_points':int(y.sum()),
                  **{f'raw_{k}':v for k,v in cr.items()},**{f'PA_{k}':v for k,v in cp.items()},
                  'FN_removed_by_PA':cr['FN']-cp['FN']}
            all_rows.append(item);local_rows.append(item)
        error=(rec-target)**2;share=float(error[:,0].sum()/error.sum()) if error.sum() else 0
        contributions.append(share)
        arrays[f'{CONFIGS[j]}_reconstruction_all_features']=rec
        arrays[f'{CONFIGS[j]}_scores']=e['bf']['scores'][idx]
        ax.text(.995,.96,f"POT {e['pot']['threshold']:.3g}  |  BF200 {e['bf']['threshold']:.3g}",
                transform=ax.transAxes,ha='right',va='top',fontsize=7.5,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':1.5})
    matrix=np.array(matrix)
    bottom.imshow(matrix,aspect='auto',interpolation='nearest',origin='upper',
                  cmap=ListedColormap(['#FFFFFF',FP,FN]),vmin=0,vmax=2,
                  extent=(a-.5,b-.5,15.5,-.5))
    bottom.set_yticks(np.arange(16),rowlabels,fontsize=8)
    bottom.set_title('3  False positives and false negatives  |  same score and threshold before/after PA',loc='left',fontsize=12,pad=12)
    for k in np.arange(1.5,16,2):bottom.axhline(k,color='#D8D8D8',lw=.6)
    for k in (3.5,7.5,11.5):bottom.axhline(k,color='#AAAAAA',lw=1)
    # Keep numbers inside the figure, in a separate right-hand margin.
    bottom.set_position([.155,.125,.74,bottom.get_position().height])
    for r in range(16):
        item=local_rows[r//2];prefix='raw' if r%2==0 else 'PA'
        bottom.text(1.012,r,f"FP {item[prefix+'_FP']:3d}   FN {item[prefix+'_FN']:3d}",
                    transform=matplotlib.transforms.blended_transform_factory(bottom.transAxes,bottom.transData),
                    fontsize=7.5,va='center',family='monospace')
    # Match x alignment across all three panels (reserve the same count margin).
    for ax in [top]+scoreaxes:
        pos=ax.get_position();ax.set_position([.155,pos.y0,.74,pos.height])
    bottom.set_xlabel('Sample index within fragment' if ds=='OPS-SAT' else 'Sample index within original track')
    bottom.legend(handles=[Patch(facecolor=FP,label='False positive: nominal sample flagged'),
                           Patch(facecolor=FN,label='False negative: positive label missed'),
                           Patch(facecolor='white',edgecolor='#BBBBBB',label='Correct prediction')],
                  loc='upper left',bbox_to_anchor=(-.17,-.14),ncol=3,frameon=False,fontsize=9)
    fig.text(.025,.050,'PA can fill missed positive labels after an event is hit; it leaves false positives unchanged. White after PA does not mean every sample was directly detected.',fontsize=9)
    note=('OPS-SAT: the label applies to the whole fragment, not to an exact fault onset.' if ds=='OPS-SAT' else
          f'Top: first telemetry coordinate only. Score: mean squared error over all {target.shape[1]} inputs. Its first-coordinate share here: '+', '.join(f'{v:.0%}' for v in contributions)+'.')
    fig.text(.025,.030,note,fontsize=9,color=GRAY)
    fig.text(.025,.013,'Pale shading = positive labels. BF200 selected with PA on the full test set. POT follows the existing implementation. PA applied before cropping.',fontsize=8,color=GRAY)
    fig.savefig(OUT/(stem+'.png'),dpi=190);fig.savefig(OUT/(stem+'.pdf'));pdf.savefig(fig);plt.close(fig)
    np.savez_compressed(OUT/(stem+'.npz'),**arrays);figures.append(stem)
    print('  saved',stem,flush=True)
write('case_metrics.csv',all_rows)
(OUT/'provenance.json').write_text(json.dumps({'source_runs':sources,'replay_parity':parities,'figures':figures,
    'script_sha256':sha(__file__),'alpha':.5,'seed':2026,'epochs':50,'architecture':'reference',
    'evaluation':'Threshold-only recomputation at fixed alpha .5; BF200 maximizes PA F1. Raw uses exactly the same thresholds. No retuning for no-PA.',
    'selection':'Same seven real intervals as prior review. OPS inherited earlier outcome-based illustrative controls (previously seed2028 encoder-only); SMAP G-7 morphology-selected, E-4 outcome-selected; MSL P-11 morphology-selected, M-6 deliberately maximum-range diagnostic. No population-level claims.',
    'training_added':0,'source_data_or_results_changed':False},indent=2)+'\n')
(OUT/'README.md').write_text('''# Consistent W / FF visual comparison

Open Visual_checks.pdf. Seven real intervals, four TranAD+ models per interval: reference, smaller W, smaller FF, both smaller. All use seed2026, 50 epochs and the same other training settings within each dataset.

Three panels: observed telemetry and reconstructions; four aligned score lanes with POT (solid) and BF200 (dashed) thresholds; pointwise errors for both threshold methods, before and after PA. Orange = false positive, red = false negative, white = correct. Counts appear on the right. All panels share the same sample axis.

Alpha is fixed at .5 for every model. Thus the same reconstruction and score legitimately have both POT and BF200 thresholds. Thresholds were recalculated using full original stored scores; their PA F1 is checked against the saved alpha=.5 report. This is a controlled visual diagnostic, not the independently alpha-optimized headline benchmark. Raw predictions use the identical PA-selected threshold: no raw retuning. PA is applied to full original sequences / separate OPS fragments before cropping.

The selected cases illustrate behavior, not its frequency. Both favorable and unfavorable examples are retained. OPS cases239/1560/1281 come from prior outcome-based controls (then encoder-only, seed2028), now shown for all four reference-architecture configurations at seed2026. SMAP G-7 is a prior morphology-selected event; E-4 was selected for a prior window-related false-alarm difference. MSL P-11 is morphology-selected; M-6 deliberately illustrates the largest previously found out-of-output-range target. Physical fault causes are not inferred from these traces.

SMAP/MSL top panels show coordinate0; scores use all 25/55 inputs. OPS labels cover whole fragments, not precise physical fault locations. Shading always represents the same supplied positive labels. FP is invariant under PA; fewer FN after PA means event credit, not necessarily better time localization.

No training, checkpoint changes, original-result changes or thesis changes. Build script, compressed plotted arrays, exact counts and source hashes are included for reproducibility.
''')
print('COMPLETE',OUT/'Visual_checks.pdf',flush=True)

