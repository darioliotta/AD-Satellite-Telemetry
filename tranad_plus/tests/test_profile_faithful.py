"""CPU regression of original hyperparameters and *full* timed oracle metric path."""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tools.profile_five_epochs import baseline_profile,batch_choices,create_source
from tools.profile_full_metrics import pot_bounded,metric_sweep,extract,eval_batches
from TranADPlus.src.omnianomaly import pot_eval
from TranADPlus.src.global_config import Config


def test_profiles_exact_upstream_and_distinct_adapted_opssat():
    upstream=Config().TranAD
    for name in ('SMAP','MSL'):
        p=baseline_profile(name)
        for key in ('window_sz','dim_feedforward','batch_sz','lr','pot_q','padding','num_layers'):
            assert p[key]==upstream[name][key]
        assert upstream[name]['epochs']==5
        assert batch_choices(name,p)==[4096,8192,16384]
    ops=baseline_profile('OPS-SAT')
    assert ops['batch_sz']==256
    assert batch_choices('OPS-SAT',ops)==[256,1024,4096,8192]
    assert 'NO upstream' in ops['profiling_baseline_source']


def test_all_four_cpu_modes_same_windows():
    windows=torch.arange(15*10*3,dtype=torch.float32).reshape(15,10,3)
    for mode in ('ordinary','pinned','ram','vram'):
        batches,store,seconds,n=create_source(windows,mode,8,'cpu',15)
        torch.testing.assert_close(torch.cat(list(batches)),windows)
        assert n==15 and seconds>=0
        assert len(batches)==2


def test_eval_all_four_cpu_modes_and_label_alignment():
    class Fake(torch.utils.data.Dataset):
        padding=True
        def __init__(self):
            self.windows=torch.arange(16*3*1,dtype=torch.float32).reshape(16,3,1)/100
            self.point_labels=np.array([0]*6+[1]*10,dtype='int8')
        def __len__(self):return len(self.windows)
        def __getitem__(self,i):return self.windows[i]
    class Fixed:
        def eval(self):return self
        def __call__(self,x,t):return t*.3,t*.7
    data=Fake()
    for mode in ('ordinary','pinned','ram','vram'):
        a,perf=extract(Fixed(),data,mode,5,'cpu')
        assert len(a)==4 and len(a[0])==16
        np.testing.assert_array_equal(a[3],data.point_labels)
        np.testing.assert_allclose(a[0],data.windows[:,-1,:].numpy()*.3,atol=1e-7)
        assert perf['windows']==16


def test_pot_bounded_parity_to_original_spot():
    rng=np.random.default_rng(2026)
    train=rng.lognormal(-2,.6,1800)
    test=np.r_[rng.lognormal(-2,.6,620),rng.lognormal(.6,.4,70)]
    labels=np.r_[np.zeros(620),np.ones(70)].astype('int8')
    for q in (.01,.07):
        original,_,_=pot_eval(train,test,labels.copy(),q=q)
        bounded=pot_bounded(train,test,labels,q)
        for key in ('pot_f1','pot_precision','pot_recall','pot_threshold','pot_ROC/AUC'):
            np.testing.assert_allclose(bounded[key],original[key],rtol=0,atol=1e-12)


def test_21_alphas_all_three_metric_families_with_known_scores():
    rng=np.random.default_rng(77)
    ntr,nte=900,500
    tr=rng.lognormal(-2,.35,size=(ntr,1))
    te=np.r_[rng.lognormal(-2,.35,size=(460,1)),rng.lognormal(.4,.2,size=(40,1))]
    train=(tr*.95,tr*1.05,np.zeros_like(tr),np.zeros(ntr,dtype='int8'))
    test=(te*.95,te*1.05,np.zeros_like(te),np.r_[np.zeros(460),np.ones(40)].astype('int8'))
    result=metric_sweep('SMAP',train,test,None,.07,alpha_count=21)
    assert result['alpha_count']==21 and len(result['full_alpha_results'])==21
    assert result['pot'] is not None and result['best_f1_original'] is not None
    assert result['best_f1_dense_oracle']['value']>=result['best_f1_original']['value']-1e-12
    assert result['pot_failed_alpha_count']==0
    assert result['components']['pot_seconds']>0


def test_profiling_sweep_total_and_no_original_overrides():
    """One 5-epoch run per distinct (batch, mode, schedule, repeat) combination."""
    mode_count, schedule_count, repeat_count = 4, 2, 2
    n_ops = len(batch_choices('OPS-SAT', baseline_profile('OPS-SAT')))
    n_smap = len(batch_choices('SMAP', baseline_profile('SMAP')))
    n_msl = len(batch_choices('MSL', baseline_profile('MSL')))
    assert (n_ops, n_smap, n_msl) == (4, 3, 3)
    assert n_ops * mode_count * schedule_count * repeat_count == 64
    assert n_smap * mode_count * schedule_count * repeat_count == 48
    assert n_msl * mode_count * schedule_count * repeat_count == 48
    assert (n_ops + n_smap + n_msl) * mode_count * schedule_count * repeat_count == 160
