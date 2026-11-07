"""Regression for the v11 OPS-SAT scorer and data-mode timing interfaces."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from opssat_pipeline import FragmentWindows, make_model, score_decoders

def test_opssat_score_decoders_cpu_no_undefined_transfer_path():
    groups=[{'id':1,'channel':'A','label':0,'values':np.array([.1,.3,.5,.7],dtype=np.float32)},
            {'id':2,'channel':'A','label':1,'values':np.array([.2,.8,.6],dtype=np.float32)}]
    scaler=MinMaxScaler().fit(groups[0]['values'].reshape(-1,1))
    dataset=FragmentWindows(groups,{'A':scaler},window=4)
    profile={'padding':True,'window_sz':4,'batch_sz':2,'dim_feedforward':4,'num_layers':1}
    model=make_model(profile).eval()
    a,b,target,performance=score_decoders(model,dataset,'cpu',batch_size=2)
    assert len(a)==len(b)==len(target)==7
    assert performance['points_scored']==7
    assert np.isfinite(a).all() and np.isfinite(b).all()

def test_cpu_five_epoch_data_modes_equal_batches():
    from tools.profile_five_epochs import create_source
    x=torch.arange(72,dtype=torch.float32).reshape(6,4,3)
    for mode in ('ordinary','pinned','ram','vram'):
        batches,storage,preparation,n=create_source(x,mode,bs=2,device='cpu',limit=6)
        output=torch.cat(list(batches),dim=0)
        torch.testing.assert_close(output,x)
        assert n==6 and preparation>=0

def test_native_cached_decoder_matches_original_generator():
    from torch.utils.data import DataLoader, TensorDataset
    from TranADPlus.src.helpers import gen_TranAD_predictions
    from paper_metrics import decoder_arrays, anomaly_scores
    class LabeledWindows(TensorDataset):
        padding=True
    class FixedDecoder:
        def eval(self):return self
        def __call__(self,w,t):return t*.25,t*.75
    x=torch.linspace(.05,.95,8*4*3,dtype=torch.float32).reshape(8,4,3)
    labels=torch.zeros_like(x)
    labels[2:4,-1,0]=1
    ds=LabeledWindows(x,labels)
    loader=DataLoader(ds,batch_size=3,shuffle=False)
    arr=decoder_arrays(FixedDecoder(),loader,'cpu')
    for alpha in (.0,.2,.5,1.):
        original=gen_TranAD_predictions(FixedDecoder(),torch.nn.MSELoss(reduction='none'),loader,
                                        alpha=alpha,beta=1-alpha,device='cpu')
        expected=original[0].mean(axis=1)
        np.testing.assert_allclose(anomaly_scores(arr,alpha),expected,rtol=1e-6,atol=1e-8)
        np.testing.assert_array_equal(arr[3],(original[4].sum(axis=1)>0).astype('i1'))

def test_pot_threshold_matches_original_on_valid_synthetic_data():
    from opssat_pipeline import _pot_threshold
    from TranADPlus.src.omnianomaly import pot_eval
    rng=np.random.default_rng(43)
    train=rng.lognormal(mean=-2,sigma=.65,size=1200)
    test=np.r_[rng.lognormal(mean=-2,sigma=.65,size=550),rng.lognormal(mean=.5,sigma=.5,size=50)]
    label=np.r_[np.zeros(550),np.ones(50)].astype(int)
    for q in (0.01,0.07):
        measured,error=_pot_threshold(train,test,q)
        assert error is None
        native,_,_=pot_eval(train,test,label.copy(),q)
        np.testing.assert_allclose(measured,native['pot_threshold'],rtol=0,atol=1e-13)

def test_dense_best_f1_matches_brute_score_sweep():
    from TranADPlus.src.omnianomaly import calc_seq
    from paper_metrics import dense_best_f1
    rng=np.random.default_rng(7)
    for n in (10,25,45):
        for trial in range(12):
            scores=rng.integers(0,10,size=n).astype(float)
            labels=rng.integers(0,2,size=n)
            if labels.min()==labels.max():continue
            candidates=np.r_[np.nextafter(scores.min(),-np.inf),np.unique(scores)]
            ref=max(calc_seq(scores,labels.copy(),t)[0][0] for t in candidates)
            got=dense_best_f1(scores,labels)['bf_dense_f1']
            np.testing.assert_allclose(got,ref,rtol=0,atol=1e-12)
