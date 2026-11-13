# Moved from the session scratchpad (2026-11-13). Run from the repository root with PYTHONPATH=.
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from tranad_plus.campaigns.knn.calibration import load_tracks
from tranad_plus.campaigns.knn.metrics import runs
S=np.load('results/v27/msl_diagnosis/msl_scores.npy')
tracks,_=load_tracks('MSL'); names=[t['name'] for t in tracks]
y=np.concatenate([t['labels'] for t in tracks]); off=np.cumsum([0]+[len(t['test']) for t in tracks])
thK=np.quantile(S[0][~y],.95); thT=np.quantile(S[1][~y],.95)
base=Path('data/processed/tranad_plus/MSL')
pick=['D-15.npy','F-7.npy','M-2.npy','D-14.npy']
fig,ax=plt.subplots(4*len(pick),1,figsize=(13,3.2*len(pick)*1.6),sharex=False,gridspec_kw={'height_ratios':[2.2,1,1,1.2]*len(pick)})
for i,n in enumerate(pick):
    k=names.index(n); a,b=off[k],off[k+1]; t=tracks[k]
    te=np.load(base/'Test/Tracks'/n); tr=np.load(base/'Train/Tracks'/n)
    x=np.arange(b-a); lab=t['labels']
    A=ax[4*i]; A.plot(x,te[:,0],lw=.7,c='k'); A.axhline(tr[:,0].min(),c='g',ls='--',lw=.8); A.axhline(tr[:,0].max(),c='g',ls='--',lw=.8,label='training range')
    for s,e in runs(lab): A.axvspan(s,e,color='r',alpha=.15)
    A.set_title(f'MSL {n[:-4]}: telemetry (coordinate 0); red = annotated anomaly; green = training min/max'); A.legend(loc='upper right',fontsize=7)
    C=ax[4*i+1]; act=(te[:,1:]!=0); C.imshow(act.T,aspect='auto',interpolation='nearest',cmap='Greys'); C.set_ylabel('commands'); C.set_yticks([])
    for s,e in runs(lab): C.axvspan(s,e,color='r',alpha=.15)
    T=ax[4*i+2]; T.semilogy(x,S[1][a:b]+1e-9,lw=.6,c='C0'); T.axhline(thT,c='C0',ls=':'); T.set_ylabel('TranAD+\nerror')
    for s,e in runs(lab): T.axvspan(s,e,color='r',alpha=.15)
    K=ax[4*i+3]; K.plot(x,S[0][a:b],lw=.6,c='C1'); K.axhline(thK,c='C1',ls=':'); K.set_ylabel('KNN\ndistance')
    for s,e in runs(lab): K.axvspan(s,e,color='r',alpha=.15)
plt.tight_layout(); plt.savefig('results/v27/msl_diagnosis/msl_examples.png',dpi=110)
