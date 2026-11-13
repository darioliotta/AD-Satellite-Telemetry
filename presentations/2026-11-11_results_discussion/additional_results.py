"""Verified architecture/precision results. Figures only, no causal discussion."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

DS=('OPS-SAT','SMAP','MSL')
VAR=('reference','shared_decoder','single_pass','encoder_only')
NAMES=('TranAD+','One-decoder','One-pass','Encoder-only')
SHORT=('TranAD+','One-dec.','One-pass','Enc-only')
CONFIG=('reference','window_only','ff_only','compact')
FORMATS=('fp32','int8','int4')
RED='#9B0014';GRAY='#737373';INK='#252525'
PALETTE=['#FFF9FA','#F7DDE2','#DA8F9D',RED]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def create_pages(root,out,canvas,footer):
    sources=[];integer={};weights={};raw={};rows=[]
    def read(path):
        statuspath=path.parent/'status.json'
        status=json.loads(statuspath.read_text())
        assert status['status']=='complete',str(statuspath)
        digest=sha(path)
        assert digest==status['artifacts'][path.name],str(path)
        sources.append({'file':str(path.relative_to(root)),'sha256':digest,
                        'status':str(statuspath.relative_to(root)),'status_sha256':sha(statuspath)})
        return json.loads(path.read_text())
    expected={(d,v,c,s) for d in DS for v in VAR for c in CONFIG for s in range(2026,2031)}
    for directory,collection in [('v17/integer',integer),('v16/ptq',weights)]:
        for path in sorted((root/'results'/directory).glob('*/*/*/seed_*/metrics.json')):
            r=read(path);p=r['profile'];key=tuple(p[k] for k in ('dataset','variant','configuration','seed'))
            assert key not in collection and p['epochs']==50
            collection[key]=r
            for fmt in FORMATS:
                f=r['formats'][fmt];assert f['status']=='measured'
                if directory=='v17/integer' and fmt!='fp32':
                    assert 'INT8' in f['storage']['execution']
                if directory=='v16/ptq' and fmt!='fp32':
                    assert f['storage']['execution']=='dequantized_fp32_simulation'
                for method in ('bf','pot'):
                    m=f['oracle'][method]
                    assert m['point_adjustment'] in ('fragment_aware','upstream_global')
                    auc=.5*(m[method+'_TP']/(m[method+'_TP']+m[method+'_FN'])+
                             m[method+'_TN']/(m[method+'_TN']+m[method+'_FP']))
                    np.testing.assert_allclose(auc,m[method+'_ROC/AUC'],atol=1e-10,rtol=0)
                    rows.append({'family':directory,'dataset':p['dataset'],'variant':p['variant'],
                        'configuration':p['configuration'],'seed':p['seed'],'window':p['window_sz'],
                        'feedforward':p['dim_feedforward'],'format':fmt,'method':method,
                        **{k:100*m[method+'_'+k] for k in ('f1','precision','recall','ROC/AUC')},
                        'tensor_bytes':f['storage']['tensor_bytes_including_bias_scales_buffers'],
                        'source':str(path.relative_to(root))})
        assert set(collection)==expected,(directory,len(collection))
    for path in sorted((root/'results/v18/no_pa').glob('*/*/*/seed_*/*/metrics.json')):
        r=read(path);p=r['profile'];fmt=r['format']
        if fmt not in FORMATS:continue
        key=tuple(p[k] for k in ('dataset','variant','configuration','seed'))+(fmt,)
        assert key not in raw
        raw[key]=r
    assert set(raw)=={k+(fmt,) for k in expected for fmt in FORMATS}
    write(out/'data/architecture_precision_runs.csv',rows)
    rawrows=[]
    for (d,v,c,s,fmt),r in raw.items():
        for method in ('bf','pot'):
            m=r['evaluation']['protocols']['no_pa_optimized'][method]
            rawrows.append({'dataset':d,'variant':v,'configuration':c,'seed':s,'format':fmt,'method':method,
                            **{k:100*m[k] for k in ('f1','precision','recall','auc_binary')}})
    write(out/'data/architecture_precision_no_pa_runs.csv',rawrows)

    def vals(d,v=None,c='reference',fmt='fp32',method='bf',metric='f1',collection=None,nopa=False):
        a=[]
        source=integer if collection is None else collection
        for key,r in source.items():
            ds,variant,configuration,seed=key
            if ds!=d or (v is not None and variant!=v) or (c is not None and configuration!=c):continue
            m=(raw[key+(fmt,)]['evaluation']['protocols']['no_pa_optimized'][method]
               if nopa else r['formats'][fmt]['oracle'][method])
            a.append(100*m[metric if nopa else method+'_'+metric])
        assert a
        return np.array(a)

    def table(title,headers,body,widths,*,fontsize=18,top=.80,height=.068,groups=1,note=''):
        fig=canvas(title);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        left=.055;total=sum(widths)
        ax.add_patch(Rectangle((left,top-height),total,height,facecolor=RED,edgecolor='none'))
        for row,values in enumerate([headers]+body):
            y=top-height*(row+1)
            if row and ((row-1)//groups)%2==0:
                ax.add_patch(Rectangle((left,y),total,height,facecolor='#FAF5F6',edgecolor='none'))
            x=left
            for j,(text,width) in enumerate(zip(values,widths)):
                ax.text(x+.012 if j==0 else x+width/2,y+height/2,str(text),va='center',
                        ha='left' if j==0 else 'center',fontsize=fontsize-1 if row==0 else fontsize,
                        color='white' if row==0 else INK)
                x+=width
        if note:footer(fig,note)
        return fig

    def schema():
        fig=canvas('Architectural variants');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        def box(x,y,w,text):
            ax.add_patch(FancyBboxPatch((x,y),w,.105,boxstyle='round,pad=0,rounding_size=.008',
                         facecolor='#FFF9FA',edgecolor=RED,lw=1))
            ax.text(x+w/2,y+.0525,text,ha='center',va='center',fontsize=16 if '\n' in text else 18,color=RED)
        def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.3,color=GRAY))
        for i,(name,sub) in enumerate([('One-decoder','2 passes · same encoder and decoder'),('One-pass','1 encoder–decoder pass'),('Encoder-only','1 encoder pass + reconstruction head')]):
            y=.70-i*.23
            ax.text(.055,y+.065,name,fontsize=22,color=RED)
            ax.text(.055,y+.012,sub,fontsize=12,color=GRAY)
            box(.40,y,.15,'Encoder');box(.77,y,.18,'Linear + sigmoid')
            if i<2:
                box(.59,y,.14,'Shared\ndecoder' if i==0 else 'Decoder')
                arrow((.55,y+.0525),(.59,y+.0525));arrow((.73,y+.0525),(.77,y+.0525))
                if i==0:
                    ax.plot([.86,.86,.475,.475],[y,y-.065,y-.065,y],color=GRAY,lw=1)
                    arrow((.475,y-.035),(.475,y))
                    ax.text(.68,y-.095,'focus feedback to second pass',ha='center',fontsize=12,color=GRAY)
            else:arrow((.55,y+.0525),(.77,y+.0525))
        footer(fig,'The reference TranAD+ has two passes and two decoders with separate parameters.')
        return fig

    def grid_table():
        body=[]
        for d in DS:
            p=[integer[d,'reference',c,2026]['profile'] for c in CONFIG]
            body.append([d,*[f"{q['window_sz']} / {q['dim_feedforward']}" for q in p]])
        return table('Architecture study · configurations',
                     ['Dataset','Reference','Smaller W','Smaller FF','Both smaller'],body,[.17,.18,.18,.18,.18],
                     fontsize=20,height=.115,top=.77,
                     note='W / FF · 4 architectures × 4 settings × 5 seeds × 3 datasets = 240 trained models · 50 epochs')

    def arch_f1(nopa=False):
        fig=canvas('Architectures · F1 without PA' if nopa else 'Architectures · F1 with PA')
        axes=fig.subplots(1,3);fig.subplots_adjust(left=.065,right=.98,bottom=.22,top=.75,wspace=.25)
        for ax,d in zip(axes,DS):
            xs=np.arange(4)
            for j,method in enumerate(('bf','pot')):
                a=[vals(d,v,method=method,nopa=nopa) for v in VAR]
                means=[x.mean() for x in a];sd=[x.std(ddof=1) for x in a];pos=xs+(-.18 if j==0 else .18)
                ax.bar(pos,means,.31,color=RED if j==0 else '#A9A9A9',label='BF200' if j==0 else 'POT',
                       yerr=sd,capsize=2,error_kw={'lw':1,'ecolor':GRAY})
                for x,m,s in zip(pos,means,sd):ax.text(x,m+s+2,f'{m:.1f}',ha='center',fontsize=10)
            ax.set_title(d,loc='left',fontsize=21);ax.set_xticks(xs,SHORT,rotation=15);ax.tick_params(axis='x',labelsize=12)
            ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('F1 (%)' if d==DS[0] else '')
            ax.grid(axis='y',color='#EEEEEE');ax.set_axisbelow(True)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.52,.87),fontsize=17)
        footer(fig,'Reference W/FF · FP32 · mean ± sample SD over 5 seeds · '+('selection targets raw F1' if nopa else 'selection targets PA F1'))
        return fig

    def arch_metrics(d):
        body=[]
        for v,name in zip(VAR,NAMES):
            for method in ('bf','pot'):
                body.append([name,'BF200' if method=='bf' else 'POT',
                    *[f'{vals(d,v,method=method,metric=m).mean():.2f}' for m in ('precision','recall','f1','ROC/AUC')]])
        return table(d+' · architecture results (%)',['Model','Method','Precision','Recall','F1','AUC-PA'],body,
                     [.23,.12,.14,.13,.12,.15],fontsize=18,height=.068,groups=2,
                     note='Reference W/FF · FP32 · PA · mean over 5 seeds · AUC-PA uses binary predictions after PA')

    def arch_resources():
        fig=canvas('Architectures · parameters and matrix operations')
        axes=fig.subplots(1,3);fig.subplots_adjust(left=.065,right=.98,bottom=.23,top=.75,wspace=.25)
        for ax,d in zip(axes,DS):
            for j,field in enumerate(('unique_parameters','matmul_MACs_per_window')):
                numbers=[]
                for v in VAR:
                    q=integer[d,v,'reference',2026]['formats']['fp32']['resources']
                    numbers.append(q[field] if field=='unique_parameters' else q['analytic_operations'][field])
                percentages=100*np.array(numbers)/numbers[0];pos=np.arange(4)+(-.18 if j==0 else .18)
                ax.bar(pos,percentages,.31,color=RED if j==0 else '#A9A9A9',label='Parameters' if j==0 else 'MACs / window')
                for x,n in zip(pos,percentages):ax.text(x,n+2,f'{n:.0f}',ha='center',fontsize=12)
            ax.set_title(d,loc='left',fontsize=21);ax.set_xticks(range(4),SHORT,rotation=15);ax.tick_params(axis='x',labelsize=12)
            ax.set_ylim(0,115);ax.set_ylabel('% of TranAD+' if d==DS[0] else '')
            ax.grid(axis='y',color='#EEEEEE');ax.set_axisbelow(True)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.52,.87),fontsize=17)
        footer(fig,'Reference W/FF · analytical matrix-operation counts; biases, activations and softmax excluded')
        return fig

    def matrix_slide(title,labels,matrices,note):
        fig=canvas(title);axes=fig.subplots(1,3)
        fig.subplots_adjust(left=.12,right=.975,bottom=.30,top=.78,wspace=.33)
        cmap=LinearSegmentedColormap.from_list('Padova',PALETTE)
        for ax,d in zip(axes,DS):
            a=matrices[d];lo=np.floor(a.min());hi=max(np.ceil(a.max()),lo+1);norm=Normalize(lo,hi)
            im=ax.pcolormesh(np.arange(a.shape[1]+1)-.5,np.arange(5)-.5,a,
                             cmap=cmap,norm=norm,edgecolors='white',linewidth=.7,shading='flat',rasterized=False)
            ax.invert_yaxis();ax.set_title(d,loc='left',fontsize=21,pad=15)
            ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(4),SHORT if d==DS[0] else ['']*4)
            ax.tick_params(length=0,labelsize=13)
            for sp in ax.spines.values():sp.set_visible(False)
            for i in range(4):
                for j in range(len(labels)):
                    rgb=cmap(norm(a[i,j]));lum=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                    ax.text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',fontsize=17,color=INK if lum>.55 else 'white')
            b=ax.get_position();cax=fig.add_axes([b.x0,.15,b.width,.022]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
            cb.outline.set_visible(False);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face')
            cb.set_label('Mean F1 (%)',fontsize=13);cb.ax.tick_params(labelsize=11)
        footer(fig,note+' · dataset-specific colour scales')
        return fig

    def arch_grid(method):
        arrays={d:np.array([[vals(d,v,c,method=method).mean() for c in CONFIG] for v in VAR]) for d in DS}
        return matrix_slide('Architectures × W/FF · '+('BF200' if method=='bf' else 'POT'),
                            ['Ref.','W','FF','Both'],arrays,'FP32 · PA · 5 seeds per cell')

    def quant_table(kind):
        nopa=kind=='raw';collection=weights if kind=='weights' else integer
        title={'weights':'Quantized weights · FP32 calculations','integer':'Quantized weights + integer matrix products',
               'raw':'Integer quantization · F1 without PA'}[kind]
        body=[]
        for d in DS:
            for method in ('bf','pot'):
                body.append([d,'BF200' if method=='bf' else 'POT',
                    *[f'{vals(d,c=None,fmt=f,method=method,collection=collection,nopa=nopa).mean():.2f}' for f in FORMATS]])
        note='Mean F1 (%) · 4 architectures × 4 W/FF settings × 5 seeds per dataset · '+('raw-F1 selection' if nopa else 'PA')
        fig=table(title,['Dataset','Method','FP32','INT8','INT4'],body,[.20,.15,.18,.18,.18],
                  fontsize=22,height=.085,groups=2,note=note)
        if kind=='integer':
            fig.text(.055,.11,'INT8 = W8A8 · INT4 = W4A4 · W4A4 matrix products use INT8 instructions.',fontsize=13,color=GRAY)
        return fig

    def quant_arch(method):
        arrays={d:np.array([[vals(d,v,c=None,fmt=f,method=method).mean() for f in FORMATS] for v in VAR]) for d in DS}
        return matrix_slide('Integer precision × architecture · '+('BF200' if method=='bf' else 'POT'),
                            ['FP32','INT8','INT4'],arrays,'PA · means over 4 W/FF settings × 5 seeds per cell')

    def storage():
        body=[]
        for d in DS:
            for v,name in zip(VAR,NAMES):
                sizes=[]
                for fmt in FORMATS:
                    sizes.append(np.mean([integer[d,v,'reference',s]['formats'][fmt]['storage']['tensor_bytes_including_bias_scales_buffers']/1024
                                          for s in range(2026,2031)]))
                body.append([d if v=='reference' else '',name,*[f'{n:.1f}' for n in sizes]])
        return table('Integer quantization · stored model tensors (KiB)',
                     ['Dataset','Model','FP32','INT8','INT4'],body,[.16,.25,.16,.16,.16],
                     fontsize=17,height=.050,top=.81,groups=4,
                     note='Reference W/FF · packed tensors, biases, scales and buffers included · training checkpoints excluded')

    def quant_metrics(method):
        body=[]
        for d in DS:
            for fmt in FORMATS:
                body.append([d if fmt=='fp32' else '',fmt.upper(),
                    *[f'{vals(d,c=None,fmt=fmt,method=method,metric=m).mean():.2f}' for m in ('precision','recall','f1','ROC/AUC')]])
        return table('Integer quantization · '+('BF200' if method=='bf' else 'POT')+' metrics (%)',
                     ['Dataset','Format','Precision','Recall','F1','AUC-PA'],body,[.17,.14,.14,.14,.14,.16],
                     fontsize=18,height=.062,groups=3,
                     note='PA · mean over 4 architectures × 4 W/FF settings × 5 seeds · AUC-PA from binary predictions')

    pages=[('18_architecture_schema',schema),('19_architecture_grid',grid_table),
           ('20_architecture_f1_pa',lambda:arch_f1(False)),('21_architecture_f1_no_pa',lambda:arch_f1(True)),
           *[(f'{22+i}_{d.lower()}_architecture_metrics',lambda d=d:arch_metrics(d)) for i,d in enumerate(DS)],
           ('25_architecture_resources',arch_resources),
           ('26_architecture_grid_bf',lambda:arch_grid('bf')),('27_architecture_grid_pot',lambda:arch_grid('pot')),
           ('28_quantized_weights',lambda:quant_table('weights')),('29_integer_quantization',lambda:quant_table('integer')),
           ('30_integer_architecture_bf',lambda:quant_arch('bf')),('31_integer_architecture_pot',lambda:quant_arch('pot')),
           ('32_integer_no_pa',lambda:quant_table('raw')),('33_integer_storage',storage),
           ('34_integer_bf_metrics',lambda:quant_metrics('bf')),('35_integer_pot_metrics',lambda:quant_metrics('pot'))]
    # Retain exact means and seed SDs, including underlying conditions for pooled results.
    summary=[]
    for d in DS:
        for v in VAR:
            for c in CONFIG:
                for fmt in FORMATS:
                    for method in ('bf','pot'):
                        a=vals(d,v,c,fmt,method)
                        summary.append({'dataset':d,'variant':v,'configuration':c,'format':fmt,'method':method,
                                        'mean_F1_PA_pct':float(a.mean()),'sample_SD_pp':float(a.std(ddof=1)),'seeds':5})
    write(out/'data/architecture_precision_summary.csv',summary)
    manifest={'training_models':240,'seeds':list(range(2026,2031)),'epochs':50,
              'architecture_reference_setting':'Fixed largest W/FF per dataset; all architecture PA/no-PA bars use 5 seeds.',
              'all_settings':'Four W/FF configurations per architecture; pooled quantization entries explicitly average 80 parents per dataset, not 80 independent training seeds.',
              'raw_selection':'V18 no_pa_optimized: BF200 selects alpha and threshold for raw F1; POT chooses alpha with its SPOT threshold.',
              'integer_scope':'W8A8 and W4A4 integer matrix products; W4A4 uses INT8 instructions, not native INT4. Some nonlinear and scaling operations remain floating point.',
              'weight_only_scope':'V16 packed integer weights, dequantized FP32 matrix operations.',
              'resources':'Analytical MACs and unique parameters; tensor payload bytes include scales/biases/buffers. No speedup or energy claim.',
              'source_reports':sources,'builder_sha256':sha(__file__)}
    (out/'data/additional_results_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return pages
