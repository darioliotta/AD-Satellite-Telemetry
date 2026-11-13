"""Isolate original threshold and feature-contribution effects in real cases."""
from pathlib import Path
import csv,json,hashlib,os,sys,statistics as st
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/toast-matplotlib')
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tranad_plus'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from TranADPlus.src.omnianomaly import adjust_predicts
RED='#9B0014';INK='#202020';BLUE='#286A8C';GRAY='#777777'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})

def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    rows=list(csv.DictReader((OUT/'example_counts.csv').open()))
    globalrows=list(csv.DictReader((OUT/'original_endpoint_metrics.csv').open()))
    summaries=[]
    for ds in ['SMAP','MSL']:
        for method in ['bf','pot']:
            for variant,config in [('reference','reference'),('reference','window_only'),('reference','ff_only'),('reference','compact'),('encoder_only','reference')]:
                group=[r for r in globalrows if r['dataset']==ds and r['method']==method and r['variant']==variant and r['configuration']==config]
                assert len(group)==3
                summary={'dataset':ds,'method':method,'variant':variant,'configuration':config,'seeds':3}
                for k in ['PA_f1','PA_precision','PA_recall','raw_f1','raw_precision','raw_recall','raw_FPR','AUC_score_full']:
                    a=[100*float(r[k]) for r in group];summary[k+'_mean_pct']=st.mean(a);summary[k+'_SD_pp']=st.stdev(a)
                summaries.append(summary)
    write('matched_global_summary.csv',summaries)
    # Isolate alpha using two archived mixed reconstructions of the SAME outputs.
    matched=[]
    for case,ds,unit in [(1,'SMAP','E-13'),(3,'SMAP','P-4'),(5,'MSL','M-3'),(7,'MSL','M-7')]:
        with np.load(OUT/f'{case:02}_{ds.lower()}_{unit}_bf.npz') as bf,np.load(OUT/f'{case:02}_{ds.lower()}_{unit}_pot.npz') as pot:
            x=bf['target_all_features'];y=bf['labels'].astype(bool);xx=bf['sample_index']
            for j,name in enumerate(['TranAD+','Smaller W']):
                b=next(r for r in rows if int(r['example'])==case and r['method']=='bf' and r['model']==name)
                p=next(r for r in rows if int(r['example'])==case and r['method']=='pot' and r['model']==name)
                ab,ap=float(b['alpha']),float(p['alpha']);assert ab!=ap
                reconstruction=((.5-ap)/(ab-ap))*bf[f'model{j}_reconstruction']+((ab-.5)/(ab-ap))*pot[f'model{j}_reconstruction']
                score=np.mean((reconstruction-x)**2,axis=1);config='reference' if j==0 else 'window_only'
                source=ROOT/f'results/v13/architectures/{ds}/reference/{config}/seed_2026';r=json.loads((source/'metrics.json').read_text())
                starts={};offset=0
                for path in sorted((ROOT/f'data/processed/tranad_plus/{ds}/Test/Tracks').glob('*.npy')):
                    starts[path.stem]=offset;offset+=len(np.load(path,mmap_mode='r'))+r['profile']['window_sz']-1
                idx=starts[unit]+xx
                with np.load(source/'scores.npz') as z:
                    original=z['test_scores'][10][idx]
                np.testing.assert_allclose(score,original,rtol=1e-3,atol=1e-7)
                oracle=json.loads((ROOT/f'results/v17/integer/{ds}/reference/{config}/seed_2026/metrics.json').read_text())['formats']['fp32']['oracle']
                endpoint=next(a for a in oracle['alpha_grid'] if a['alpha']==.5)
                item={'case':case,'dataset':ds,'track':unit,'model':name,'fixed_alpha':.5,
                     'mean_crop_MSE':float(score.mean()),'mean_positive_MSE':float(score[y].mean()) if y.any() else None,
                     'mean_nominal_MSE':float(score[~y].mean()) if (~y).any() else None,
                     'same_alpha_BF_threshold':endpoint['bf_threshold'],
                     'same_alpha_BF_direct_TP':int(np.sum(y&(score>endpoint['bf_threshold']))),
                     'same_alpha_BF_direct_FP':int(np.sum(~y&(score>endpoint['bf_threshold']))),
                     'max_score_reconstruction_drift':float(np.max(np.abs(score-original))),
                     'spike_reconstruction_4813':None,'spike_reconstruction_4815':None}
                if case==3:
                    for position in [4813,4815]:item[f'spike_reconstruction_{position}']=float(reconstruction[np.flatnonzero(xx==position)[0],0])
                matched.append(item)
    write('matched_alpha_reconstruction_controls.csv',matched)
    with PdfPages(OUT/'Isolated_controls.pdf') as pdf:
        # Exactly the same BF-alpha score; change only threshold to its original POT counterpart.
        controls=list(csv.DictReader((OUT/'MSL_matched_alpha_threshold_control.csv').open()))
        bf=next(r for r in controls if r['variant']=='reference' and r['configuration']=='reference' and r['seed']=='2026' and r['endpoint']=='original_BF200')
        pot=next(r for r in controls if r['variant']=='reference' and r['configuration']=='reference' and r['seed']=='2026' and r['endpoint']=='POT_threshold_at_same_BF_alpha')
        with np.load(OUT/'07_msl_M-7_bf.npz') as z:
            xx=z['sample_index'];y=z['labels'];x=z['target_all_features'];rec=z['model0_reconstruction'];scores=z['model0_score']
        fig,axes=plt.subplots(3,1,figsize=(14,9.4),sharex=True,gridspec_kw={'height_ratios':[2.4,2,1.6]})
        fig.subplots_adjust(left=.16,right=.97,top=.86,bottom=.15,hspace=.3)
        fig.text(.035,.97,'MSL / M-7: same model, same alpha, same score',fontsize=19,color=RED,va='top')
        fig.text(.035,.925,'Only the threshold changes. This control does not retrain or optimize another alpha.',fontsize=11)
        axes[0].plot(xx,x[:,0],color=INK,label='Observed telemetry');axes[0].plot(xx,rec[:,0],color=RED,label='TranAD+ reconstruction; alpha 0.65')
        axes[0].set_ylabel('Normalized value');axes[0].legend(frameon=False,fontsize=9)
        axes[1].plot(xx,scores,color=RED,label='Identical anomaly score')
        axes[1].axhline(float(bf['threshold']),color=INK,ls='--',label=f"BF200: {float(bf['threshold']):.6g}")
        axes[1].axhline(float(pot['threshold']),color=BLUE,ls=':',label=f"POT at same alpha: {float(pot['threshold']):.6g}")
        axes[1].set_yscale('log');axes[1].set_ylabel('MSE, all 55 inputs');axes[1].legend(frameon=False,fontsize=9)
        pred=[]
        for e in [bf,pot]:
            threshold=float(e['threshold']);direct=scores>threshold
            # This crop contains the complete annotated event with nominal margins.
            assert not y[0] and not y[-1]
            pa=adjust_predicts(scores,y.copy(),threshold).astype(bool);pred.extend([direct,pa])
        for i,pr in enumerate(pred):axes[2].scatter(xx[pr],np.full(int(pr.sum()),i),marker='|',s=55,color=RED if i<2 else BLUE)
        axes[2].set_yticks(range(4),['BF200 / no PA','BF200 / PA','POT, same alpha / no PA','POT, same alpha / PA'],fontsize=9)
        axes[2].set_ylim(3.5,-.5);axes[2].set_xlabel('Sample index within track')
        for ax in axes:
            ax.axvspan(939.5,1040.5,color='#F5E8EA',zorder=0);ax.grid(axis='y',color='#EEEEEE')
        fig.text(.035,.075,f"Entire MSL test stream, alpha 0.65: PA F1 {100*float(bf['PA_f1']):.2f}% -> {100*float(pot['PA_f1']):.2f}% | raw F1 {100*float(bf['raw_f1']):.2f}% -> {100*float(pot['raw_f1']):.2f}%",fontsize=11)
        fig.text(.035,.045,f"Continuous-score ROC-AUC is unchanged: {100*float(bf['AUC_score_same_scores']):.2f}%. BF200 first grid step: {float(bf['BF_grid_step']):.6g}.",fontsize=10,color=GRAY)
        fig.savefig(OUT/'MSL_threshold_only_control.png',dpi=180);fig.savefig(OUT/'MSL_threshold_only_control.pdf');pdf.savefig(fig);plt.close(fig)
        # Decompose the actual W25 BF score into telemetry and encoded-coordinate errors.
        with np.load(OUT/'07_msl_M-7_bf.npz') as z:
            xx=z['sample_index'];y=z['labels'];x=z['target_all_features'];rec=z['model1_reconstruction'];scores=z['model1_score']
        errors=(rec-x)**2;part=errors/x.shape[1];dominant=int(np.argmax(part.sum(axis=0)));peak=int(np.argmax(scores))
        np.testing.assert_allclose(part.sum(axis=1),scores,rtol=5e-4,atol=1e-7)
        feature_rows=[{'coordinate':i,'share_of_crop_MSE':float(part[:,i].sum()/part.sum()),
                      'share_of_peak_MSE':float(part[peak,i]/part[peak].sum()),
                      'target_at_peak':float(x[peak,i]),'reconstruction_at_peak':float(rec[peak,i]),
                      'peak_sample':int(xx[peak])} for i in range(x.shape[1])]
        write('MSL_M7_feature_contributions.csv',feature_rows)
        bfrow=next(r for r in rows if r['example']=='7' and r['method']=='bf' and r['model']=='Smaller W')
        threshold=float(bfrow['threshold']);telemetry=part[:,0];encoded=part[:,1:].sum(axis=1)
        fig,axes=plt.subplots(3,1,figsize=(14,9.4),sharex=True,gridspec_kw={'height_ratios':[2,2,2.1]})
        fig.subplots_adjust(left=.09,right=.97,top=.86,bottom=.15,hspace=.3)
        fig.text(.035,.97,'MSL / M-7: which input creates the BF200 alarm?',fontsize=19,color=RED,va='top')
        fig.text(.035,.925,'W25 / FF110; original BF200 alpha=0.0. Coordinate 0 is the telemetry; others encode command-related inputs.',fontsize=10)
        axes[0].plot(xx,x[:,0],color=INK,label='Observed telemetry');axes[0].plot(xx,rec[:,0],color=RED,label='Reconstruction')
        axes[0].set_ylabel('Telemetry coord. 0');axes[0].legend(frameon=False,fontsize=9)
        axes[1].plot(xx,x[:,dominant],color=INK,label=f'Observed encoded coordinate {dominant}')
        axes[1].plot(xx,rec[:,dominant],color=RED,label='Reconstruction');axes[1].set_ylabel(f'Input coordinate {dominant}');axes[1].legend(frameon=False,fontsize=9)
        axes[2].plot(xx,scores,color=INK,label='Total anomaly score')
        axes[2].plot(xx,telemetry,color=BLUE,ls='--',label='Telemetry-coordinate contribution')
        axes[2].plot(xx,encoded,color=RED,ls=':',label='Other-coordinate contribution')
        axes[2].axhline(threshold,color=GRAY,ls='--',label='Original BF200 threshold')
        axes[2].set_yscale('log');axes[2].set_ylabel('Contribution to mean MSE');axes[2].set_xlabel('Sample index within track');axes[2].legend(frameon=False,fontsize=9)
        for ax in axes:ax.axvspan(939.5,1040.5,color='#F5E8EA',zorder=0);ax.grid(axis='y',color='#EEEEEE')
        fig.text(.035,.075,f"At sample {int(xx[peak])}, telemetry contributes {100*telemetry[peak]/scores[peak]:.2f}% of the score; other coordinates contribute {100*encoded[peak]/scores[peak]:.2f}%.",fontsize=11)
        fig.text(.035,.045,'This alarm lies inside the annotated event. Its feature contribution does not identify a physical command fault or prove that it is spurious.',fontsize=9,color=GRAY)
        fig.savefig(OUT/'MSL_feature_contribution_control.png',dpi=180);fig.savefig(OUT/'MSL_feature_contribution_control.pdf');pdf.savefig(fig);plt.close(fig)
    result={'matched_reconstruction_checks':matched,'same_alpha_threshold_control':{'bf':bf,'pot':pot},
        'feature_peak':feature_rows[dominant],'telemetry_share_at_peak':float(telemetry[peak]/scores[peak]),
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'isolated_controls.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE isolated controls',flush=True)

if __name__=='__main__':main()
