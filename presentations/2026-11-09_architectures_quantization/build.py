#!/usr/bin/env python3
"""Build Results II from verified, completed V14–V17 artifacts. V18 excluded."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics as st
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/toast-presentation-mpl')
if Path('/tmp/toast-presentation-deps').is_dir():sys.path.insert(0,'/tmp/toast-presentation-deps')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor

PROJECT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('toast_previous_deck',PROJECT/'presentations/2026-11-03_progress/build.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
RED,INK,GRAY,PALE,GRID=base.RED,base.INK,base.GRAY,base.PALE,base.GRID
W,H=base.W,base.H
DS=('OPS-SAT','SMAP','MSL')
VAR=('reference','shared_decoder','single_pass','encoder_only')
VN={'reference':'TranAD+','shared_decoder':'One-decoder','single_pass':'One-pass','encoder_only':'Encoder-only'}
SHORT=('TranAD+','One-dec.','One-pass','Enc-only')
CONFIG=('reference','window_only','ff_only','compact')
FORMATS=('fp32','int8','int4')
FC={'fp32':INK,'int8':'#D68A98','int4':RED}
SOURCES={};DISPLAY={};REPORTS={};WEIGHTS={};CONT=[];FROZEN=[];TRAIN=[]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_json(path,status=None,full=False):
    path=Path(path)
    if status:
        sp=path.parent/status;s=json.loads(sp.read_text())
        if s['status']!='complete':raise RuntimeError('Incomplete source '+str(path))
        expected=s['artifacts'].get(path.name)
        if expected is None or sha(path)!=expected:raise RuntimeError('Changed source report '+str(path))
        if full:
            for name,h in s['artifacts'].items():
                if sha(path.parent/name)!=h:raise RuntimeError('Changed source artifact '+str(path.parent/name))
        SOURCES[str(sp.relative_to(PROJECT))]={'sha256':sha(sp),'role':'completion and artifact provenance'}
    SOURCES[str(path.relative_to(PROJECT))]={'sha256':sha(path),'role':'numerical source'}
    return json.loads(path.read_text())


def write_csv(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def mean(a):return st.mean(a)
def sd(a):return st.stdev(a) if len(a)>1 else 0.
def ms(a,digits=2):return f'{mean(a):.{digits}f} ± {sd(a):.{digits}f}'

def group(d,v,c='reference',collection=None):
    source=REPORTS if collection is None else collection
    values=[source[(d,v,c,s)] for s in range(2026,2031)]
    assert len(values)==5
    return values


def f1_values(d,v,c='reference',fmt='fp32',method='bf',protocol='oracle'):
    return [100*r['formats'][fmt][protocol][method][method+'_f1'] for r in group(d,v,c)]


def endpoint_values(d,v,metric='f1',fmt='fp32',method='pot',c='reference'):
    return [100*r['formats'][fmt]['oracle'][method][method+'_'+metric] for r in group(d,v,c)]


def pooled(d,fmt,method,protocol='oracle',metric='f1',variant=None,collection=None):
    values=REPORTS if collection is None else collection
    return [100*r['formats'][fmt][protocol][method][method+'_'+metric]
            for (ds,v,c,s),r in values.items() if ds==d and (variant is None or v==variant)]


def resource_values(d,v,field,fmt='fp32'):
    a=[]
    for r in group(d,v):
        f=r['formats'][fmt]
        if field=='bytes':a.append(f['storage']['tensor_bytes_including_bias_scales_buffers'])
        elif field=='params':a.append(f['resources']['unique_parameters'])
        elif field=='macs':a.append(f['resources']['analytic_operations']['matmul_MACs_per_window'])
        elif field=='ms':a.append(f['resources']['median_ms'])
    return a


def load(project):
    metric_rows=[]
    for p in sorted((project/'results/v17/integer').glob('*/*/*/seed_*/metrics.json')):
        r=read_json(p,'status.json',full=True);q=r['profile'];key=tuple(q[k] for k in ('dataset','variant','configuration','seed'))
        if key in REPORTS:raise RuntimeError('Duplicate parent')
        REPORTS[key]=r
        for fmt in FORMATS:
            f=r['formats'][fmt]
            if f['status']!='measured':raise RuntimeError('Unavailable format')
            for method in ('bf','pot'):
                e=f['oracle'][method];locked=f['parent_locked'][method]
                assert e['point_adjustment'] in ('fragment_aware','upstream_global')
                auc=.5*(e[method+'_TP']/(e[method+'_TP']+e[method+'_FN'])+e[method+'_TN']/(e[method+'_TN']+e[method+'_FP']))
                assert abs(auc-e[method+'_ROC/AUC'])<1e-10
                metric_rows.append({**{k:q[k] for k in ('dataset','variant','configuration','seed','window_sz','dim_feedforward')},
                    'format':fmt,'method':method,'F1_PA_pct':100*e[method+'_f1'],'precision_PA_pct':100*e[method+'_precision'],
                    'recall_PA_pct':100*e[method+'_recall'],'AUC_PA_pct':100*e[method+'_ROC/AUC'],
                    'F1_raw_at_PA_selection_pct':100*e['raw_without_PA'][method+'_f1'],
                    'F1_parent_locked_PA_pct':100*locked[method+'_f1'],
                    'tensor_bytes':f['storage']['tensor_bytes_including_bias_scales_buffers'],'latency_median_ms':f['resources']['median_ms'],
                    'source':str(p.relative_to(project))})
    expected={(d,v,c,s) for d in DS for v in VAR for c in CONFIG for s in range(2026,2031)}
    assert set(REPORTS)==expected
    for p in sorted((project/'results/v16/ptq').glob('*/*/*/seed_*/metrics.json')):
        r=read_json(p,'status.json');q=r['profile'];key=tuple(q[k] for k in ('dataset','variant','configuration','seed'))
        assert all(r['formats'][f]['status']=='measured' for f in FORMATS)
        WEIGHTS[key]=r
    assert set(WEIGHTS)==expected
    for d in DS:
        for v in ('reference','encoder_only'):
            for seed in range(2026,2031):
                path=project/f'results/v14/continuation/{d}/{v}/reference/seed_{seed}/comparison.json'
                r=read_json(path,'diagnostic_status.json')
                assert r['parent_epochs']==50 and r['final_epochs']==100
                row=dict(dataset=d,variant=v,seed=seed,source=str(path.relative_to(project)))
                for method in ('bf','pot'):
                    for stage in ('before','after'):
                        row[stage+'_'+method]=100*r['steps'][stage]['paper_oracle'][method][method+'_f1']
                CONT.append(row)
    for d in DS:
        for v in VAR:
            for seed in range(2026,2031):
                path=project/f'results/v15/frozen/{d}/{v}/reference/seed_{seed}/metrics.json'
                r=read_json(path,'status.json')
                for condition in r['conditions']:
                    FROZEN.append(dict(dataset=d,variant=v,seed=seed,condition=condition['condition'],
                        pot_f1=100*condition['methods']['pot']['pot_f1'],bf_f1=100*condition['methods']['bf']['bf_f1'],source=str(path.relative_to(project))))
    for v in VAR:
        for condition in ('shuffle_1101','repeat_current','current_only'):
            config='no_history' if condition=='current_only' else 'reference'
            folder='original' if condition=='current_only' else condition
            for seed in range(2026,2031):
                path=project/f'results/v15/train/OPS-SAT/{v}/{config}/{folder}/seed_{seed}/metrics.json'
                r=read_json(path,'status.json');e=r['matched']['pot']
                TRAIN.append(dict(dataset='OPS-SAT',variant=v,seed=seed,condition=condition,
                    f1=None if e.get('pot_f1') is None else 100*e['pot_f1'],
                    raw_f1=None if e.get('raw_without_PA',{}).get('pot_f1') is None else 100*e['raw_without_PA']['pot_f1'],
                    pot_status=e.get('status','measured'),source=str(path.relative_to(project))))
    assert len(CONT)==30 and len(FROZEN)==360 and len(TRAIN)==60
    return metric_rows


def figure(subtitle=''):
    fig=base.canvas()
    if subtitle:fig.text(.055,.858,subtitle,fontsize=17.5,color=GRAY,va='top')
    return fig


def insight(fig,text):fig.text(.055,.072,text,fontsize=18,color=INK,va='bottom')


def panels(fig,top=.735,bottom=.23):
    axes=[fig.add_axes([.075+i*.315,bottom,.265,top-bottom]) for i in range(3)]
    for ax,d in zip(axes,DS):
        ax.set_title(d,fontsize=21,color=RED,weight='bold',pad=14)
        ax.grid(axis='y',color=GRID,lw=.7);ax.set_axisbelow(True)
        ax.tick_params(labelsize=14)
    return axes


def fmt_table(fig,headers,rows,widths,**kwargs):
    base.table(fig,headers,rows,widths,**kwargs)


def architecture_schema():
    fig=figure('One encoder module is reused across passes in the two-pass variants.')
    ax=base.plain_axis(fig)
    def box(x,y,w,h,text,accent=True):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.008',
                     facecolor=PALE if accent else '#F5F5F5',edgecolor=RED if accent else '#BBBBBB',lw=1.1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=20,color=RED if accent else INK)
    for y,v,note in zip([.68,.51,.34,.17],VAR,['2 passes · feedback','2 passes · feedback','1 pass','1 pass']):
        ax.text(.055,y+.074,VN[v],fontsize=22,color=INK,weight='bold',va='center')
        ax.text(.055,y+.019,note,fontsize=15,color=GRAY,va='center')
        box(.285,y,.17,.115,'Encoder')
        box(.725,y,.22,.115,'Linear + sigmoid')
        if v!='encoder_only':
            label='Decoder 1 / 2' if v=='reference' else 'Shared decoder' if v=='shared_decoder' else 'Decoder'
            box(.505,y,.17,.115,label)
            for a,b in [(.455,.505),(.675,.725)]:ax.add_patch(FancyArrowPatch((a,y+.0575),(b,y+.0575),arrowstyle='-|>',mutation_scale=14,color=GRAY,lw=1.2))
        else:
            ax.add_patch(FancyArrowPatch((.455,y+.0575),(.725,y+.0575),arrowstyle='-|>',mutation_scale=14,color=GRAY,lw=1.2))
            ax.text(.59,y+.09,'last encoded token',ha='center',fontsize=14,color=GRAY)
    return fig


def study_grid():
    fig=figure('3 datasets · 4 architectures · 4 W/FF settings · 5 seeds · 50 epochs')
    rows=[]
    for d in DS:
        p=[group(d,'reference',c)[0]['profile'] for c in CONFIG]
        rows.append([d,str(p[0]['features']),*[f"{q['window_sz']} / {q['dim_feedforward']}" for q in p]])
    fmt_table(fig,['Dataset','Features','Reference','Window only','FF only','Both'],rows,[.16,.12,.15,.17,.15,.14],
              top=.77,row_height=.091,fontsize=21)
    ax=base.plain_axis(fig)
    for x,title,lines in [(.055,'BF200',['200 thresholds · 21 alpha values','Select the largest test F1']),(.53,'POT',['SPOT threshold for each alpha','Select the largest test F1 across alpha'])]:
        ax.text(x,.337,title,fontsize=24,color=RED,weight='bold')
        for j,line in enumerate(lines):ax.text(x,.28-j*.052,line,fontsize=19,color=INK)
    ax.text(.055,.105,'W / FF in table · single-output models have no alpha sweep',fontsize=17,color=GRAY)
    ax.text(.055,.055,'AUC-PA: binary predictions after PA · (Recall + Specificity) / 2',fontsize=17,color=GRAY)
    DISPLAY['study_grid']=rows
    return fig


def pa_example():
    fig=figure('One hit inside a labelled anomaly interval fills that interval during evaluation.')
    ax=base.plain_axis(fig);truth=np.zeros(12,dtype=int);truth[2:8]=1
    a=np.zeros(12,dtype=int);a[[2,3,4,9]]=1
    b=np.zeros(12,dtype=int);b[6]=1
    ap=a.copy();ap[2:8]=1;bp=truth.copy()
    labels=['Labels','A · direct','A · after PA','B · direct','B · after PA']
    vectors=[truth,a,ap,b,bp];original=[truth,a,a,b,b]
    x0=.28;cw=.048;gap=.006;h=.038
    for i in range(12):ax.text(x0+i*(cw+gap)+cw/2,.805,str(i+1),ha='center',fontsize=12,color=GRAY)
    for row,(name,values,initial) in enumerate(zip(labels,vectors,original)):
        y=.75-row*.053
        ax.text(.055,y+h/2,name,va='center',fontsize=17,color=INK)
        for i,val in enumerate(values):
            color='white'
            if val:color=RED if truth[i] else INK
            if row in (2,4) and val and not initial[i]:color='#E5AEB8'
            ax.add_patch(Rectangle((x0+i*(cw+gap),y),cw,h,facecolor=color,edgecolor='#CFCFCF',lw=.6))
    ax.text(.28,.493,'Red: detected · pale red: filled by PA · dark: false alarm',fontsize=14,color=GRAY)
    rows=[['A','Direct','3','1','3','75.0','50.0','60.0'],['A','PA','6','1','0','85.7','100.0','92.3'],
          ['B','Direct','1','0','5','100.0','16.7','28.6'],['B','PA','6','0','0','100.0','100.0','100.0']]
    fmt_table(fig,['Model','Scoring','TP','FP','FN','Precision','Recall','F1'],rows,[.10,.12,.075,.075,.075,.16,.15,.135],
              top=.455,row_height=.061,fontsize=18,groups=2,left_columns=(0,1))
    insight(fig,'Intervals come from labels, not window size. False alarms outside remain unchanged.')
    DISPLAY['pa_example']={'toy_example':True,'labels':truth.tolist(),'A':a.tolist(),'B':b.tolist(),'table':rows}
    return fig


def architecture_cost():
    fig=figure('Reference W/FF · quantities relative to TranAD+ FP32')
    axes=panels(fig)
    data=[];x=np.arange(4)
    for ax,d in zip(axes,DS):
        params=[mean(resource_values(d,v,'params')) for v in VAR]
        macs=[mean(resource_values(d,v,'macs')) for v in VAR]
        for off,vals,color in [(-.17,params,RED),(.17,macs,'#B4B4B4')]:
            heights=100*np.array(vals)/vals[0];bars=ax.bar(x+off,heights,width=.30,color=color)
            for z,h in zip(bars,heights):ax.text(z.get_x()+z.get_width()/2,h+2,f'{h:.0f}',ha='center',fontsize=12)
        ax.set(ylim=(0,119),xticks=x,xticklabels=SHORT);ax.tick_params(axis='x',labelsize=12)
        data.append({'dataset':d,'parameters':params,'MACs':macs})
    axes[0].set_ylabel('% of TranAD+',fontsize=17)
    fig.legend(handles=[Rectangle((0,0),1,1,color=RED,label='Parameters'),Rectangle((0,0),1,1,color='#B4B4B4',label='MACs')],
               loc='upper center',bbox_to_anchor=(.52,.185),ncol=2,frameon=False,fontsize=17)
    insight(fig,'Sharing decoder weights saves parameters; removing a pass also saves computation.')
    DISPLAY['architecture_cost']=data
    return fig


def architecture_scores(method):
    fig=figure('Reference W/FF · FP32 · PA · mean ± sample SD over 5 model seeds')
    axes=panels(fig);data=[];x=np.arange(4)
    for ax,d in zip(axes,DS):
        values=[f1_values(d,v,method=method) for v in VAR]
        mu=[mean(a) for a in values];err=[sd(a) for a in values]
        ax.bar(x,mu,width=.60,color=[INK,'#D9A1AB','#BD5F72',RED],yerr=err,capsize=3,
               error_kw={'elinewidth':1,'ecolor':'#666666'})
        for i,(m,e) in enumerate(zip(mu,err)):ax.text(i,m+e+2,f'{m:.1f}',ha='center',fontsize=14)
        ax.set(ylim=(0,103),xticks=x,xticklabels=SHORT);ax.tick_params(axis='x',labelsize=12)
        data.append({'dataset':d,'mean':mu,'sample_sd':err})
    axes[0].set_ylabel('F1 (%)',fontsize=17)
    insight(fig,'Architectural changes have different effects across the three datasets.')
    DISPLAY['architecture_'+method]=data
    return fig


def operating_points():
    fig=figure('Reference W/FF · FP32 · POT + PA · mean over 5 model seeds')
    rows=[]
    for d in DS:
        for v in ('reference','encoder_only'):
            rows.append([d,VN[v],*[f'{mean(endpoint_values(d,v,k)):.2f}' for k in ('precision','recall','f1','ROC/AUC')]])
    fmt_table(fig,['Dataset','Model','Precision','Recall','F1','AUC-PA'],rows,[.15,.20,.145,.13,.13,.135],
              top=.78,row_height=.083,fontsize=21,groups=2,left_columns=(0,1))
    insight(fig,'Percentages at the selected operating point; metrics are not maximized separately.')
    DISPLAY['operating_points']=rows
    return fig


def window_heatmaps():
    fig=figure('POT + PA · mean over 5 seeds in every architecture/configuration cell')
    axes=[fig.add_axes([.13+i*.295,.28,.245,.425]) for i in range(3)];data=[]
    cmap=LinearSegmentedColormap.from_list('padova',['#FCF6F7','#DCABB5',RED])
    for ax,d in zip(axes,DS):
        values=np.array([[mean(f1_values(d,v,c,method='pot')) for c in CONFIG] for v in VAR])
        ax.imshow(values,vmin=40,vmax=95,cmap=cmap,aspect='auto')
        ax.set_title(d,fontsize=21,color=RED,weight='bold',pad=16)
        ax.set(xticks=range(4),xticklabels=['Ref.','W','FF','Both'],yticks=range(4),yticklabels=SHORT)
        ax.tick_params(axis='x',labelsize=16,bottom=False);ax.tick_params(axis='y',labelsize=13,left=False)
        if d!='OPS-SAT':ax.set_yticklabels([])
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=16,color='white' if values[i,j]>80 else INK)
        for sp in ax.spines.values():sp.set_visible(False)
        data.append({'dataset':d,'POT_F1_pct':values.tolist()})
    insight(fig,'Reference, reduced window, reduced FF, or both: no universal monotonic pattern.')
    DISPLAY['window_ff']=data
    return fig


def continuation():
    fig=figure('Reference W/FF · PA · mean F1 (%) over 5 seeds · continuation from epoch 50')
    rows=[]
    for d in DS:
        for v in ('reference','encoder_only'):
            a=[r for r in CONT if r['dataset']==d and r['variant']==v]
            assert len(a)==5
            rows.append([d,VN[v],*[f'{mean([r[k] for r in a]):.2f}' for k in ('before_bf','after_bf','before_pot','after_pot')]])
    fmt_table(fig,['Dataset','Model','BF 50 ep.','BF 100 ep.','POT 50 ep.','POT 100 ep.'],rows,[.15,.19,.135,.145,.145,.145],
              top=.78,row_height=.083,fontsize=20,groups=2,left_columns=(0,1))
    insight(fig,'An additional 50 epochs do not give a uniform improvement.')
    DISPLAY['continuation']=rows
    return fig


def history_schema():
    fig=figure('Only past samples change. The current sample and reconstruction target stay fixed.')
    ax=base.plain_axis(fig)
    labels=['Original','Shuffled past','Repeated current'];values=[[1,3,2,4,5,6],[4,1,5,2,3,6],[6,6,6,6,6,6]]
    for row,(label,vals) in enumerate(zip(labels,values)):
        y=.64-row*.18
        ax.text(.055,y+.045,label,fontsize=23,weight='bold',va='center')
        for i,value in enumerate(vals):
            x=.37+i*.083
            ax.add_patch(FancyBboxPatch((x,y),.066,.09,boxstyle='round,pad=0,rounding_size=.006',facecolor=RED if i==5 else PALE,edgecolor=RED,lw=1))
            ax.text(x+.033,y+.045,str(value),ha='center',va='center',fontsize=25,color='white' if i==5 else RED)
    ax.text(.37,.795,'Past samples',fontsize=18,color=GRAY)
    ax.text(.818,.795,'Current',fontsize=18,color=RED,ha='center')
    insight(fig,'Frozen inference tests sensitivity; matched retraining checks adaptation to the input condition.')
    DISPLAY['history_schema']={'illustration':True,'values':values}
    return fig


def frozen_effects():
    fig=figure('POT + PA · fixed parent alpha/threshold · paired changes over 5 model seeds')
    axes=panels(fig);x=np.arange(4);data=[]
    for ax,d in zip(axes,DS):
        shifts={};errors={}
        for condition in ('shuffle','repeat_current'):
            values=[]
            for v in VAR:
                byseed=[]
                for seed in range(2026,2031):
                    part=[r for r in FROZEN if r['dataset']==d and r['variant']==v and r['seed']==seed]
                    original=next(r['pot_f1'] for r in part if r['condition']=='original')
                    if condition=='shuffle':new=mean([r['pot_f1'] for r in part if r['condition'].startswith('shuffle_')])
                    else:new=next(r['pot_f1'] for r in part if r['condition']==condition)
                    byseed.append(new-original)
                values.append(byseed)
            shifts[condition]=[mean(a) for a in values];errors[condition]=[sd(a) for a in values]
        for off,key,color in [(-.17,'shuffle',RED),(.17,'repeat_current','#9B9B9B')]:
            ax.bar(x+off,shifts[key],width=.30,color=color,yerr=errors[key],capsize=3,error_kw={'elinewidth':.9,'ecolor':'#555555'})
        ax.axhline(0,color=INK,lw=.9);ax.set(ylim=(-46,17),xticks=x,xticklabels=SHORT);ax.tick_params(axis='x',labelsize=12)
        data.append({'dataset':d,'mean_delta_pp':shifts,'sample_sd':errors})
    axes[0].set_ylabel('F1 change (pp)',fontsize=17)
    fig.legend(handles=[Rectangle((0,0),1,1,color=RED,label='Shuffled past'),Rectangle((0,0),1,1,color='#9B9B9B',label='Repeated current')],
               loc='upper center',bbox_to_anchor=(.52,.185),ncol=2,frameon=False,fontsize=17)
    insight(fig,'Shuffle results average 3 permutations per model.')
    DISPLAY['history_frozen']=data
    return fig


def retraining_controls():
    fig=figure('OPS-SAT · POT F1 (%) · matched training/test inputs')
    rows=[]
    for v in VAR:
        row=[VN[v],f'{mean(f1_values("OPS-SAT",v,method="pot")):.2f}']
        for condition in ('shuffle_1101','repeat_current','current_only'):
            all_rows=[r for r in TRAIN if r['variant']==v and r['condition']==condition];assert len(all_rows)==5
            a=[r['f1'] for r in all_rows if r['f1'] is not None]
            row.append(f'{mean(a):.2f}')
        a=[r['raw_f1'] for r in TRAIN if r['variant']==v and r['condition']=='current_only' and r['raw_f1'] is not None]
        row.append(f'{mean(a):.2f}');rows.append(row)
    fmt_table(fig,['Model','Original\nPA','Shuffle\nPA','Repeat\nPA','W = 1\nPA','W = 1\nno PA'],rows,[.21,.135,.135,.135,.135,.16],
              top=.76,row_height=.105,fontsize=21,left_columns=(0,))
    fig.text(.055,.15,'No-PA column: same PA-selected operating point.',fontsize=17,color=GRAY)
    insight(fig,'FF = 8 · matched training/test inputs · W = 10 except the W = 1 control.')
    DISPLAY['retraining_controls']=rows
    return fig


def quantization_schema():
    fig=figure('INT8 and INT4 · two ways to use reduced precision')
    ax=base.plain_axis(fig)
    panels=[(.055,'Weights only',['Weights saved as INT8 or INT4.','Calculations run in FP32.']),
            (.525,'Weights + calculations',['Weights and inputs are integers.','Matrix products run with integers.'])]
    for x,label,rows in panels:
        ax.add_patch(FancyBboxPatch((x,.205),.42,.53,boxstyle='round,pad=0,rounding_size=.012',facecolor=PALE,edgecolor='none'))
        ax.text(x+.025,.655,label,fontsize=24,color=RED,weight='bold')
        for i,line in enumerate(rows):
            ax.text(x+.025,.51-i*.10,line,fontsize=21,color=INK)
    insight(fig,'Softmax, sigmoid and some other steps remain in FP32.')
    return fig


def quantization_scores(weight_only=False):
    collection=WEIGHTS if weight_only else REPORTS
    fig=figure('PA · mean over 80 parents per dataset: 4 architectures × 4 W/FF settings × 5 seeds')
    rows=[]
    for d in DS:
        for method in ('bf','pot'):
            rows.append([d,'BF200' if method=='bf' else 'POT',*[f'{mean(pooled(d,f,method,collection=collection)):.2f}' for f in FORMATS]])
    fmt_table(fig,['Dataset','Method','FP32 (%)','INT8 (%)','INT4 (%)'],rows,[.20,.18,.17,.17,.18],top=.78,row_height=.085,fontsize=23,groups=2,left_columns=(0,1))
    insight(fig,'Weight-only sensitivity; floating-point matrix products.' if weight_only else 'Integer matrix products; threshold and alpha reselected for each precision.')
    DISPLAY['weight_only' if weight_only else 'integer_quantization']=rows
    return fig


def int4_architecture():
    fig=figure('INT4 minus matched FP32 · PA · mean paired change over 4 configurations × 5 seeds')
    axes=[fig.add_axes([.15+i*.285,.25,.215,.455]) for i in range(3)];data=[]
    cmap=LinearSegmentedColormap.from_list('padova_delta',['#7B7B7B','#FFFFFF',RED])
    for ax,d in zip(axes,DS):
        a=np.array([[mean(pooled(d,'int4',m,variant=v))-mean(pooled(d,'fp32',m,variant=v)) for m in ('bf','pot')] for v in VAR])
        ax.imshow(a,cmap=cmap,norm=TwoSlopeNorm(vmin=-35,vcenter=0,vmax=35),aspect='auto')
        ax.set(title=d,xticks=[0,1],xticklabels=['BF200','POT'],yticks=range(4),yticklabels=SHORT)
        ax.set_title(d,fontsize=21,color=RED,weight='bold',pad=16);ax.tick_params(length=0,labelsize=15)
        if d!='OPS-SAT':ax.set_yticklabels([])
        for i in range(4):
            for j in range(2):ax.text(j,i,f'{a[i,j]:+.2f}',ha='center',va='center',fontsize=21,color='white' if a[i,j]>22 else INK)
        for sp in ax.spines.values():sp.set_visible(False)
        data.append({'dataset':d,'delta_BF_POT_pp':a.tolist()})
    insight(fig,'The gain depends on architecture and threshold selection; encoder-only is not always favoured.')
    DISPLAY['int4_architecture']=data
    return fig


def pa_influence():
    fig=figure('F1 (%) · same PA-selected alpha and threshold · mean over 80 parents per dataset')
    rows=[]
    for d in DS:
        for method in ('bf','pot'):
            raw=[]
            for fmt in ('fp32','int4'):
                raw.append(mean([100*r['formats'][fmt]['oracle'][method]['raw_without_PA'][method+'_f1'] for (ds,v,c,s),r in REPORTS.items() if ds==d]))
            rows.append([d,'BF200' if method=='bf' else 'POT',f'{mean(pooled(d,"fp32",method)):.2f}',f'{mean(pooled(d,"int4",method)):.2f}',*[f'{a:.2f}' for a in raw]])
    fmt_table(fig,['Dataset','Method','PA\nFP32','PA\nINT4','No PA\nFP32','No PA\nINT4'],rows,[.16,.14,.15,.15,.15,.15],
              top=.78,row_height=.085,fontsize=22,groups=2,left_columns=(0,1))
    insight(fig,'The no-PA columns are not independently optimized pointwise F1 results.')
    DISPLAY['pa_influence']=rows
    return fig


def frozen_calibration():
    fig=figure('Original FP32 alpha and threshold kept fixed · PA · mean paired F1 change (pp)')
    rows=[]
    for d in DS:
        values=[]
        for method,fmt in [('bf','int8'),('bf','int4'),('pot','int8'),('pot','int4')]:
            values.append(mean(pooled(d,fmt,method,'parent_locked'))-mean(pooled(d,'fp32',method,'parent_locked')))
        rows.append([d,*[f'{v:+.2f}' for v in values]])
    fmt_table(fig,['Dataset','BF INT8','BF INT4','POT INT8','POT INT4'],rows,[.20,.175,.175,.175,.175],top=.74,row_height=.12,fontsize=23)
    insight(fig,'INT4 can change score calibration substantially; a previous threshold need not transfer.')
    DISPLAY['frozen_calibration']=rows
    return fig


def storage():
    fig=figure('Reference W/FF · model tensor payload including biases, scales and positional buffers')
    rows=[]
    for d in DS:
        for v in ('reference','encoder_only'):
            rows.append([d,VN[v],*[f'{mean(resource_values(d,v,"bytes",f))/1024:.2f}' for f in FORMATS]])
    fmt_table(fig,['Dataset','Model','FP32 (KiB)','INT8 (KiB)','INT4 (KiB)'],rows,[.16,.23,.17,.17,.17],
              top=.78,row_height=.085,fontsize=22,groups=2,left_columns=(0,1))
    insight(fig,'Model payload reduction is not the same as total peak inference-memory reduction.')
    DISPLAY['storage']=rows
    return fig


def latency():
    fig=figure('A100 · batch 1 · 100 warm forward measurements · mean ± SD across 5 model seeds')
    axes=panels(fig);x=np.arange(4);data=[]
    for ax,d in zip(axes,DS):
        for fmt in FORMATS:
            a=[resource_values(d,v,'ms',fmt) for v in VAR]
            ax.errorbar(x,[mean(z) for z in a],yerr=[sd(z) for z in a],color=FC[fmt],marker='o',ms=6,lw=2,capsize=3)
            data.append({'dataset':d,'format':fmt,'mean_ms':[mean(z) for z in a],'sample_sd_ms':[sd(z) for z in a]})
        ax.set(ylim=(0,22.5),xticks=x,xticklabels=SHORT);ax.tick_params(axis='x',labelsize=12)
    axes[0].set_ylabel('Forward latency (ms)',fontsize=17)
    fig.legend(handles=[Line2D([],[],color=FC[f],marker='o',lw=2,label=f.upper()) for f in FORMATS],
               loc='upper center',bbox_to_anchor=(.52,.185),ncol=3,frameon=False,fontsize=17)
    insight(fig,'This integer backend saves model storage but is slower than matched FP32 in every measured case.')
    DISPLAY['latency']=data
    return fig


def takeaways():
    fig=figure()
    ax=base.plain_axis(fig)
    statements=[('Architecture','Fewer parameters and MACs; faster FP32 inference.'),
                ('INT8','Most PA metrics preserved, with smaller model payloads.'),
                ('INT4','Gains depend on PA, dataset and threshold recalibration.'),
                ('Execution','The current integer backend adds substantial A100 latency.')]
    for y,(label,text) in zip([.74,.57,.40,.23],statements):
        ax.text(.055,y,label,fontsize=25,color=RED,weight='bold')
        ax.text(.055,y-.06,text,fontsize=22,color=INK)
    return fig


def bf_spacing():
    r=REPORTS[('MSL','encoder_only','compact',2026)]['formats']['fp32']
    stats=r['score_statistics_by_split']['test'][0];bf=r['oracle']['bf'];pot=r['oracle']['pot']
    fig=figure('MSL · encoder-only · W = 25, FF = 55 · seed 2026 · FP32 + PA')
    ax=fig.add_axes([.10,.39,.80,.30]);ax.set_xscale('log')
    ax.axvspan(max(stats['min'],1e-8),stats['quantiles_50_90_99_999'][2],color=PALE)
    ax.axvline(pot['pot_threshold'],color=RED,lw=2)
    ax.axvline(bf['bf_threshold'],color=INK,lw=2)
    ax.text(pot['pot_threshold'],.78,'POT',color=RED,ha='center',fontsize=20)
    ax.text(bf['bf_threshold'],.78,'First BF200 step',color=INK,ha='center',fontsize=20)
    ax.set(xlim=(1e-8,.03),ylim=(0,1),yticks=[],xlabel='Threshold (log scale)')
    ax.tick_params(labelsize=16);ax.set_xlabel('Threshold (log scale)',fontsize=18)
    ax.text(1e-6,.24,'99% of scores',color=GRAY,ha='center',fontsize=18)
    rows=[['POT',f"{pot['pot_threshold']:.6f}",f"{100*pot['pot_f1']:.2f}"],['BF200',f"{bf['bf_threshold']:.6f}",f"{100*bf['bf_f1']:.2f}"]]
    fmt_table(fig,['Method','Threshold','F1 with PA (%)'],rows,[.24,.30,.36],top=.30,row_height=.060,fontsize=21)
    insight(fig,'The useful threshold region can fall between the 200 grid points.')
    DISPLAY['bf_spacing']={'single_example':True,'statistics':stats,'table':rows}
    return fig


def notes():
    return {
      '01_architectures':'Architecture derived from the current original model and core/models implementations. In two-pass variants one encoder and one reconstruction head are reused. TranAD+ has two decoder modules; one-decoder shares one decoder across both passes. Encoder-only retains the reconstruction head on the last encoder token.',
      '02_grid':'The four configurations are predefined controls, not selected best results. Reference W/FF: OPS-SAT 10/8, SMAP 10/50, MSL 50/110. Seeds 2026–2030, 50 training epochs, one layer per retained stack. Alpha is selected using test labels for BF200 and POT when two outputs exist. AUC-PA is binary balanced accuracy after PA, not continuous-score ROC-AUC.',
      '03_pa':'Synthetic example, not experiment data. Anomaly labels occupy positions 3–8. A detects 3 anomaly points and one normal point; B detects one anomaly point. PA fills all 6 anomaly points for each. False positives outside remain. The model does not apply PA to its reconstruction: it is an evaluation procedure using labels.',
      '04_cost':'V17 FP32 resources, reference configuration. Model parameter counts and analytic matmul MACs are deterministic; ratios are within each dataset against TranAD+. No onboard speed measurement is implied.',
      '05_bf':'V17 FP32, reference configuration only. Bars mean and sample SD across the same five model seeds, never maxima. BF200 with PA, original 200 thresholds; MSL variability partly reflects the coarse threshold grid. No dense Best-F1 appears in this deck.',
      '06_pot':'Same models as BF200 architecture slide, POT with PA. Means and sample SD of five model seeds.',
      '07_operating_points':'Same reference configuration, POT-selected operating points for TranAD+ and encoder-only. Precision, recall, F1 and AUC-PA are taken together from the same selected endpoint. Mean of individual run metrics, not metrics recomputed from mean precision/recall.',
      '08_window_ff':'All 240 V17 FP32 parents. Each cell is a predefined architecture/configuration mean over five seeds, POT+PA. No best-configuration or best-seed selection.',
      '09_continuation':'V14 comparison.json reference-configuration subset: 30 paired continuations, three datasets, two architectures, five seeds. Epochs 50 to 100 continue checkpoint/optimizer/scheduler. BF200/POT+PA; no continuous AUC or dense BF is shown.',
      '10_history_schema':'Illustrative windows. Past shuffle preserves the current sample, target and past multiset. Repeating current removes within-window variation. Frozen inference changes only test context; matched retraining is a separate experiment.',
      '11_history_frozen':'V15 frozen reference-configuration subset, all four architectures. POT alpha and threshold frozen to the parent. For each model seed the three permutation outcomes are averaged first; displayed error bars are sample SD across five model seeds, not across 15 independent replicas.',
      '12_retraining':'V15 OPS-SAT matched-retraining controls, reference W10 and FF8, five trained seeds. Current-only is separately trained with W1 and FF8. All main columns POT+PA, final column current-only raw F1 at the PA-selected setting. These are not independently optimized no-PA results. Two cells have four valid POT evaluations (marked dagger): encoder-only repeat_current lacks seed2030, and single-pass W1 lacks seed2029, because SPOT fitting was unavailable. Missing values are not replaced by zero or silently counted as five.',
      '13_quantization':'V16 weight-only uses per-tensor weight scales and dequantized FP32 matrix products. V17 quantizes weights per output channel and activations dynamically per vector, with integer QKV, QK, AV, FFN and head products. Hence V16-to-V17 changes quantization scheme as well as execution, not only hardware kernels. Nonlinearities and residual operations remain FP32. W4A4 uses INT8 instructions with four-bit values.',
      '14_weight_only':'All 240 V16 PTQ parents, finite FP32/INT8/INT4 cases. FP16 excluded. Equal-weight mean over 80 parents per dataset. BF200/POT+PA, with alpha/threshold reselected for each representation. No training or quantization-aware training claim.',
      '15_integer':'All 240 V17 parents, 720 measured precision cases. Same predefined grid as V16, original PA metric protocol. Per-precision means of all 80 parents per dataset. The 210 pre-recovery outputs remain unchanged; the final 30 FP32 cases use their parent batch. Actual batch and parity checks are retained in source reports.',
      '16_int4_architecture':'V17 INT4 minus paired FP32. Within each dataset and architecture, mean paired metric change across four W/FF configurations and five seeds. These are descriptive pooled changes, not a sample SD over seeds or a statement of monotonic physical improvement.',
      '17_pa_influence':'V17 metrics before and after PA use the same PA-selected alpha/threshold for each precision. Raw metrics are not independently optimized no-PA results. All pending V18 data, and temporary exploratory raw reoptimizations, are explicitly excluded.',
      '18_frozen_calibration':'V17 parent_locked endpoints freeze both FP32 alpha and threshold. Deltas compare integer and freshly evaluated FP32 under that same frozen selection. They are not deltas against independently optimized integer endpoints.',
      '19_storage':'V17 reference configuration. Tensor payload includes unique parameters/buffers, bias, scales and positional buffers. This is neither compressed NPZ file size nor total process/GPU memory. Integer runtime uses buffers for weights: parameter count zero does not mean model size zero.',
      '20_latency':'V17 A100-PCIE-40GB, PyTorch 2.8+CUDA12.8, batch1. For each model use median of100 warmed synchronized forward wall-time measurements; plot mean and sample SD of those medians across five model seeds. Includes quantization/unpacking/padding overhead, excludes data loading and profiler. Architecture comparison uses reference W/FF. All480 integer cases slower than matched FP32; not a bound on an optimized target-hardware backend.',
      '21_takeaways':'Conclusions restricted to measured code, protocols and A100. PA gains do not establish superior pointwise localization. No physical cause, integer speedup, native INT4 kernel or onboard deployment performance is claimed. V18 is excluded.',
      '22_bf_spacing':'Single illustrative measured V17 FP32 encoder-only compact MSL seed2026 case. The first non-min BF200 step is roughly25 times the POT threshold. 99th percentile from saved V17 score statistics. These are original BF200 results, not a dense search or a new V18 result.'}


def build(project,out):
    out.mkdir(parents=True,exist_ok=True);figures=out/'figures';figures.mkdir(exist_ok=True)
    data=out/'data';data.mkdir(exist_ok=True);render=out/'.render';render.mkdir(exist_ok=True)
    metric_rows=load(project)
    write_csv(data/'v17_per_seed.csv',metric_rows);write_csv(data/'continuation.csv',CONT)
    write_csv(data/'history_frozen.csv',FROZEN);write_csv(data/'history_retrained.csv',TRAIN)
    pages=[
      ('01_architectures','Four architectures',architecture_schema),
      ('02_grid','Experimental grid & evaluation',study_grid),
      ('03_pa','What point adjustment changes',pa_example),
      ('04_cost','Architecture cost',architecture_cost),
      ('05_bf','Architectures · BF200',lambda:architecture_scores('bf')),
      ('06_pot','Architectures · POT',lambda:architecture_scores('pot')),
      ('07_operating_points','Precision, recall & AUC-PA',operating_points),
      ('08_window_ff','Window & FF controls · POT',window_heatmaps),
      ('09_continuation','Does more training help?',continuation),
      ('10_history_schema','Changing the past, keeping the target',history_schema),
      ('11_history_frozen','Frozen models · context sensitivity',frozen_effects),
      ('12_retraining','OPS-SAT · matched retraining controls',retraining_controls),
      ('13_quantization','Two quantization experiments',quantization_schema),
      ('14_weight_only','Weight-only PTQ · F1 with PA',lambda:quantization_scores(True)),
      ('15_integer','Integer inference · F1 with PA',lambda:quantization_scores(False)),
      ('16_int4_architecture','INT4 · architecture-dependent effects',int4_architecture),
      ('17_pa_influence','INT4 · PA and pointwise results',pa_influence),
      ('18_frozen_calibration','Keeping FP32 calibration fixed',frozen_calibration),
      ('19_storage','Model representation size',storage),
      ('20_latency','Measured inference latency',latency),
    ]
    prs=Presentation();prs.slide_width=Inches(W);prs.slide_height=Inches(H)
    prs.core_properties.author='Dario Liotta';prs.core_properties.title='Architecture and quantization results'
    evidence_notes={stem:notes()[stem] for stem,_,_ in pages}
    evidence_notes['12_retraining']=evidence_notes['12_retraining'].replace('(marked dagger)', '(availability details retained in notes)')
    with PdfPages(out/'Results_II.pdf',metadata={'Title':'Architecture and quantization results','Author':'Dario Liotta'}) as pdf:
        for i,(stem,heading,factory) in enumerate(pages,1):
            print(f'[{i:02d}/{len(pages)}] {heading}',flush=True)
            fig=factory();svg=figures/(stem+'.svg');png=render/(stem+'.png')
            fig.savefig(svg);fig.savefig(png,dpi=300)
            slide=prs.slides.add_slide(prs.slide_layouts[6]);base.add_svg(slide,svg,png,prs.part.package)
            box=slide.shapes.add_textbox(Inches(.055*W),Inches(.05*H),Inches(12.0),Inches(.60))
            frame=box.text_frame;frame.margin_left=frame.margin_right=frame.margin_top=frame.margin_bottom=0
            p=frame.paragraphs[0];p.text=heading;p.font.name='Arial';p.font.size=Pt(29);p.font.bold=True;p.font.color.rgb=RGBColor.from_string(RED[1:])
            slide.notes_slide.notes_text_frame.text=evidence_notes[stem]
            base.title(fig,heading);pdf.savefig(fig);fig.savefig(figures/(stem+'.pdf'));plt.close(fig)
    prs.save(out/'Results_II.pptx');shutil.rmtree(render)
    (data/'slide_data.json').write_text(json.dumps(DISPLAY,indent=2)+'\n')
    (data/'speaker_notes.json').write_text(json.dumps(evidence_notes,indent=2)+'\n')
    manifest={'date':'2026-11-09','language':'English','color':RED,'slides':len(pages),'excluded_campaigns':['v18'],
              'scope':'Architecture and controls through completed V17. No benchmark replication or initial v10/v12 ablation repeats.',
              'seed_policy':'Five trained seeds; POT availability exceptions retained in speaker notes. Reference configuration for architectural comparisons; pooled scopes labelled explicitly.',
              'metric_policy':'BF200/POT+PA; AUC-PA binary. Raw V17 metrics retain PA-selected settings, never relabelled as independently optimized.',
              'sources':SOURCES,'builder_sha256':sha(__file__),
              'style_builder_sha256':sha(PROJECT/'presentations/2026-11-03_progress/build.py'),
              'pages':[{'file':stem,'title':title} for stem,title,_ in pages]}
    (data/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'README.md').write_text('''# Results II — architecture and quantization\n\nEnglish · 20 slides · Padova red #9B0014 · no title page.\n\n- Results_II.pdf: vector slides.\n- Results_II.pptx: embedded SVG with 300-dpi fallback and speaker notes.\n- figures/: standalone Python-generated PDF/SVG figures.\n- data/: displayed values, selected per-seed data, source hashes and speaker notes.\n\nRebuild from the project root: `python -B presentations/2026-11-09_architectures_quantization/build.py`.\nDependencies: numpy, matplotlib, python-pptx, Pillow and lxml. The original\nResults.pdf and all research result artifacts are preserved.\n\nArchitectural figures use the reference W/FF configuration and all five seeds.\nContinuation uses the reference configuration, two architectures and five seeds.\nFrozen context uses all four architectures at reference W/FF; three shuffles\nare averaged within each model before five-seed summaries. Matched retraining\nuses OPS-SAT W10/FF8 plus the W1 control. Two POT control cells have four\navailable evaluations with availability recorded in speaker notes; all five trained\nseeds remain in the provenance. Quantization aggregate tables use all four\narchitectures and W/FF configurations, 80 parents per dataset. All scopes are\nstated on the relevant slide and detailed in speaker notes.\n\nOnly completed results through V17 are included. V18 and temporary exploratory\nno-PA reoptimizations are excluded. To add V18 later, add a verified reader and\na new page factory; do not change the meaning of the existing raw-at-PA-selected\ncomparisons. The figures are regenerated from numerical sources, not screenshots.\n''')
    print('Complete:',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=PROJECT)
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
    build(a.project_root.resolve(),a.output_dir.resolve())
