#!/usr/bin/env python3
"""Read-only ablation preview: ordinary BF200/POT, PA, all 270 saved runs."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,os,statistics as st
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/toast-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'tesi/data/chapter5/runs.csv'
DS=('OPS-SAT','SMAP','MSL');RED='#9B0014';GRAY='#555555'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

with SOURCE.open() as f:rows=list(csv.DictReader(f))
assert len(rows)==270
seen=set();groups=defaultdict(list);provenance=[]
for r in rows:
 for k in ('window','feedforward','seed','epochs'):r[k]=int(r[k])
 for k in ('bf_f1','bf_precision','bf_recall','bf_auc','pot_f1','pot_precision','pot_recall','pot_auc','parameters','macs'):r[k]=float(r[k])
 ident=(r['dataset'],r['window'],r['feedforward'],r['seed']);assert ident not in seen;seen.add(ident)
 path=ROOT/r['report_file'];assert sha(path)==r['report_sha256']
 report=json.loads(path.read_text());p=report['profile']
 assert (p['window_sz'],p['dim_feedforward'],p['seed'],p['epochs'])==(r['window'],r['feedforward'],r['seed'],50)
 assert report['checkpoint']['sha256']==r['checkpoint_sha256']
 for method in ('bf','pot'):
  saved=report['original_paper_evaluation']['metrics'][method]
  for metric in ('f1','precision','recall'):assert saved[method+'_'+metric]==r[method+'_'+metric]
  assert saved[method+'_ROC/AUC']==r[method+'_auc']
 groups[(r['dataset'],r['window'],r['feedforward'])].append(r)
 provenance.append({'report':r['report_file'],'sha256':r['report_sha256'],'checkpoint_sha256':r['checkpoint_sha256'],'campaign':r['campaign']})
assert len(groups)==90
cells=[];lookup={};axes_values={}
for ds in DS:
 part=[r for r in rows if r['dataset']==ds];ws=sorted({r['window'] for r in part});hs=sorted({r['feedforward'] for r in part})
 assert len(ws)==6 and len(hs)==5 and len(part)==90
 axes_values[ds]=(ws,hs)
 for w in ws:
  for h in hs:
   g=groups[(ds,w,h)];assert len(g)==3 and {r['seed'] for r in g}=={2026,2027,2028}
   a={'dataset':ds,'window':w,'feedforward':h,'seeds':3,'epochs':50,'campaign':g[0]['campaign']}
   for field in ('bf_f1','bf_precision','bf_recall','bf_auc','pot_f1','pot_precision','pot_recall','pot_auc'):
    values=[100*r[field] for r in g];a[field+'_mean']=st.mean(values);a[field+'_sd']=st.stdev(values)
   cells.append(a);lookup[(ds,w,h)]=a
write('cell_summary.csv',cells)
write('source_runs.csv',rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
pdf=PdfPages(OUT/'Ablation_preview.pdf');produced=[];slices=[]

def save(fig,name):
 for ext in ('pdf','svg','png'):
  path=figdir/(name+'.'+ext);fig.savefig(path,dpi=220,bbox_inches='tight',pad_inches=.12);produced.append(str(path.relative_to(OUT)))
 pdf.savefig(fig,bbox_inches='tight',pad_inches=.12);plt.close(fig)

for direction in ('window','feedforward'):
 fig,axes=plt.subplots(1,3,figsize=(14.5,4.7));fig.subplots_adjust(left=.06,right=.985,bottom=.19,top=.76,wspace=.25)
 for ax,ds in zip(axes,DS):
  ws,hs=axes_values[ds];W,H=max(ws),max(hs)
  xs=ws if direction=='window' else hs
  selected=[lookup[(ds,x,H)] if direction=='window' else lookup[(ds,W,x)] for x in xs]
  for metric,color,label,marker in [('bf',RED,'Best F1 (200 thresholds)','o'),('pot',GRAY,'POT','s')]:
   mean=np.array([a[metric+'_f1_mean'] for a in selected]);sd=np.array([a[metric+'_f1_sd'] for a in selected])
   ax.plot(xs,mean,color=color,lw=2,marker=marker,ms=5,label=label)
   ax.fill_between(xs,mean-sd,mean+sd,color=color,alpha=.10,linewidth=0)
   ax.errorbar(xs,mean,yerr=sd,fmt='none',color=color,elinewidth=1,capsize=3)
   for a in selected:slices.append({'dataset':ds,'direction':direction,'window':a['window'],'feedforward':a['feedforward'],'method':metric,'mean_f1_pct':a[metric+'_f1_mean'],'sample_sd_pp':a[metric+'_f1_sd'],'seeds':3})
  ax.set_title(ds+'\n'+(f'FF fixed at {H}' if direction=='window' else f'W fixed at {W}'),loc='left',pad=10)
  ax.set_xlabel('Window W (samples)' if direction=='window' else 'Feed-forward width FF');ax.set_xticks(xs)
  limits={'OPS-SAT':(40,100),'SMAP':(65,95),'MSL':(15,100)}[ds];ax.set_ylim(*limits);ax.set_ylabel('F1 (%)');ax.grid(axis='y',color='#e5e5e5',lw=.7)
  ax.set_axisbelow(True);ax.tick_params(direction='out')
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.91),ncol=2,frameon=False)
 fig.suptitle('Window-size ablation' if direction=='window' else 'Feed-forward-width ablation',x=.06,y=1.01,ha='left',fontsize=17,color=RED)
 fig.text(.06,.025,'Mean ± sample SD over 3 seeds · 50 epochs · point adjustment',fontsize=10,color=GRAY)
 save(fig,'01_window' if direction=='window' else '02_feedforward')
write('slice_summary.csv',slices)

cmap=LinearSegmentedColormap.from_list('padova',['#fff7f7','#eac4cb','#c66173',RED,'#4a000a'])
for metric,title,name in [('bf','Best F1 (200 thresholds)','03_heatmap_bf200'),('pot','POT','04_heatmap_pot')]:
 fig,axes=plt.subplots(1,3,figsize=(14.5,5.4));fig.subplots_adjust(left=.055,right=.99,bottom=.29,top=.84,wspace=.30)
 for ax,ds in zip(axes,DS):
  ws,hs=axes_values[ds];values=np.array([[lookup[(ds,w,h)][metric+'_f1_mean'] for w in ws] for h in hs]);lo=np.floor(values.min());hi=np.ceil(values.max());hi=max(hi,lo+1)
  norm=Normalize(lo,hi)
  im=ax.imshow(values,origin='lower',aspect='auto',cmap=cmap,norm=norm,interpolation='none')
  ax.set_xticks(range(len(ws)),ws);ax.set_yticks(range(len(hs)),hs);ax.set_xlabel('Window W (samples)');ax.set_ylabel('Feed-forward width FF');ax.set_title(ds,loc='left',pad=9)
  for spine in ax.spines.values():spine.set_visible(False)
  ax.set_xticks(np.arange(-.5,len(ws),1),minor=True);ax.set_yticks(np.arange(-.5,len(hs),1),minor=True);ax.grid(which='minor',color='white',linewidth=1.4);ax.tick_params(which='both',length=0)
  for j,h in enumerate(hs):
   for i,w in enumerate(ws):
    rgba=cmap(norm(values[j,i]));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
    ax.text(i,j,f'{values[j,i]:.1f}',ha='center',va='center',color='white' if lum<.49 else '#242424',fontsize=11)
  ax.add_patch(Rectangle((len(ws)-1.5,len(hs)-1.5),1,1,fill=False,edgecolor='black',linewidth=1.6))
  bounds=ax.get_position();cax=fig.add_axes([bounds.x0,.115,bounds.width,.025]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_label('Mean F1 (%)',fontsize=10);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=9)
 fig.suptitle('Joint W / FF ablation · '+title,x=.055,y=.985,ha='left',fontsize=16,color=RED)
 fig.text(.055,.015,'Mean of 3 seeds · 50 epochs · PA · dataset-specific colour scales',fontsize=9.5,color=GRAY)
 save(fig,name)

# Supplement: seed variability for every BF200 cell, not only the 1D slices.
fig,axes=plt.subplots(1,3,figsize=(14.5,5.1));fig.subplots_adjust(left=.055,right=.94,bottom=.17,top=.84,wspace=.30)
max_sd=max(a['bf_f1_sd'] for a in cells);vmax=5*np.ceil(max_sd/5)
for ax,ds in zip(axes,DS):
 ws,hs=axes_values[ds];values=np.array([[lookup[(ds,w,h)]['bf_f1_sd'] for w in ws] for h in hs]);im=ax.imshow(values,origin='lower',aspect='auto',cmap='Greys',vmin=0,vmax=vmax,interpolation='none')
 ax.set_xticks(range(len(ws)),ws);ax.set_yticks(range(len(hs)),hs);ax.set_title(ds,loc='left');ax.set_xlabel('Window W (samples)');ax.set_ylabel('Feed-forward width FF')
 for j in range(len(hs)):
  for i in range(len(ws)):ax.text(i,j,f'{values[j,i]:.1f}',ha='center',va='center',color='white' if values[j,i]>vmax*.6 else 'black',fontsize=11)
fig.suptitle('Best F1 (200 thresholds) · seed variability',x=.055,y=.985,ha='left',fontsize=16,color=RED)
cax=fig.add_axes([.955,.20,.012,.58]);fig.colorbar(im,cax=cax,label='Sample SD (pp)');fig.text(.055,.045,'3 seeds per cell · 50 epochs · PA · common colour scale',fontsize=10,color=GRAY)
save(fig,'05_heatmap_seed_sd');pdf.close()

endpoints=[]
for ds in DS:
 ws,hs=axes_values[ds]
 for desc,w,h in [('reference',max(ws),max(hs)),('window_only',min(ws),max(hs)),('ff_only',max(ws),min(hs)),('joint',min(ws),min(hs))]:
  a=lookup[(ds,w,h)];endpoints.append({'dataset':ds,'setting':desc,'window':w,'feedforward':h,**{k:a[k] for k in ('bf_f1_mean','bf_f1_sd','pot_f1_mean','pot_f1_sd')}})
write('endpoint_comparisons.csv',endpoints)
manifest={'source_csv':str(SOURCE.relative_to(ROOT)),'source_sha256':sha(SOURCE),'raw_reports_verified':270,'grid_cells':90,'datasets':{ds:{'windows':axes_values[ds][0],'feedforwards':axes_values[ds][1]} for ds in DS},'seeds':[2026,2027,2028],'epochs':50,'metrics':'Ordinary BF200 and POT, both PA; saved original alpha/threshold choices. No dense metric used.','statistics':'Means and sample standard deviation over 3 seeds, no seed maxima.','one_dimensional_slices':'W varies at maximum FF; FF varies at maximum W, fixed before observing outcomes.','colour':'#9B0014','heatmap_scales':'Absolute mean F1, dataset-specific colour scales, numeric cells; outline is geometrical reference, not metric winner.','campaign_scope':'Disjoint v10/v12 cells; original reports unchanged. Source-revision differences are retained; this preview does not certify numerical equivalence of campaign implementations.','figures':produced,'reports':provenance}
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'README.md').write_text('''# Ablation preview — original BF200 and POT

Open **Ablation_preview.pdf** (5 pages). PNG previews, vector PDFs and SVGs are under `figures/`.

- 270 runs: 3 datasets × 6 windows × 5 feed-forward widths × 3 seeds; 50 epochs.
- Every plotted scalar is verified against its original report and report SHA-256.
- Main 1D slices fix the other hyperparameter at the largest width/window. Lines connect recorded means, not fitted trend models. Shading and error bars show sample SD across 3 seeds, not confidence intervals.
- Heatmaps show all 30 cells per dataset. Their colour scales are local to each dataset; numeric labels permit direct comparison. The outlined cell is the largest W/FF reference, not a selected best result. A fifth page displays BF200 seed variability on a common scale.
- Only original Best F1 with 200 thresholds and POT, both point-adjusted. No dense search. Original per-run test-selected alphas/thresholds are retained, not recalibrated.
- BF200 can be lower than POT because its finite threshold grid can miss useful operating points. The marked MSL variability must not automatically be attributed to reconstruction quality; score distribution and threshold resolution are plausible contributors. No new training or score evaluation is performed here.
- The combined grid uses disjoint V10/V12 cells. This is a descriptive presentation of saved measurements, not certification that their source revisions are numerically identical. Four endpoint comparisons (largest/smallest W and FF) are all V10.
- Native padding changes the number of duplicated targets with W; OPS-SAT retains its fixed test fragments. Cross-dataset raw point metrics are not interchangeable with independent fragment classification.

`source_runs.csv` preserves all 270 rows. `cell_summary.csv`, `slice_summary.csv`, and `endpoint_comparisons.csv` contain exact plotted statistics. `provenance.json` records source/report hashes and plot definitions.

Rebuild from the repository root:

```bash
python -B presentations/2026-11-11_ablation_preview/build.py
```

Existing thesis text, prior presentations, research results and training code are not changed.
''')
print('Verified 270 reports; 90 complete cells; 5 preview pages.')
for a in endpoints:print(a)
print('PDF:',OUT/'Ablation_preview.pdf')
