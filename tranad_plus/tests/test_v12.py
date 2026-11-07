"""V12 frozen design, upstream metric parity, bounded point adjustment.

All tests run on CPU. A100 throughput and reproducibility require an actual A100.
"""
import json
import sys
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from campaign_v12 import BASES,GRID,HISTORICAL,ADDITIONS,SEEDS,new_pairs,identifier,make_plan
from paper_metrics import _stats,dense_best_f1,paper_best_f1,decoder_arrays,anomaly_scores
from TranADPlus.src import omnianomaly
from tools.report_stage1_complexity import report


class V12Contracts(unittest.TestCase):
    def test_plan_frozen_and_coverage(self):
        planned=make_plan()
        self.assertEqual(planned,json.loads((ROOT/'experiment_plan_v12.json').read_text()))
        entries=planned['experiments']
        self.assertEqual(len(entries),126)
        self.assertEqual(len({x['identifier'] for x in entries}),126)
        self.assertEqual({x['seed'] for x in entries},set(SEEDS))
        self.assertEqual({x['epochs'] for x in entries},{50})
        self.assertFalse(planned['measurement']['enabled'])
        self.assertTrue(planned['measurement']['paper_alpha_search'])
        for dataset,base in BASES.items():
            group=[x for x in entries if x['base_profile']==base['base']]
            self.assertEqual(len(group),len(new_pairs(dataset))*len(SEEDS))
            self.assertEqual({x['batch_sz'] for x in group},{base['batch']})
            self.assertEqual({(x['window_sz'],x['dim_feedforward']) for x in group},
                 set(new_pairs(dataset)))
            if dataset=='OPS-SAT':
                self.assertEqual({x['split_seed'] for x in group},{2026})
                self.assertEqual({x['validation_fraction'] for x in group},{0.2})

    def test_new_integer_grids_and_no_duplicate_historical_pairs(self):
        expected={'OPS-SAT':(4,6),'SMAP':(4,6),'MSL':(32,44)}
        for ds, windows in expected.items():
            self.assertEqual(ADDITIONS[ds]['windows'],windows)
            self.assertEqual(len(new_pairs(ds)),14)
            self.assertFalse(set(new_pairs(ds)) & {(w,f) for w in HISTORICAL[ds]['windows'] for f in HISTORICAL[ds]['feedforwards']})

    def test_report_parameters_match_theory(self):
        rows=report(ROOT/'experiment_plan_v12.json')
        self.assertEqual(len(rows),126)
        for item in rows:
            f=item['features'];h=item['feedforward']
            self.assertEqual(item['trainable_parameters'],82*f*f+12*f*h+47*f+3*h)
            self.assertEqual(item['matmul_flops_per_window_batch1'],2*item['macs_per_window_batch1'])

    def test_upstream_bf_parity(self):
        rng=np.random.default_rng(123)
        for n in (16,58,125):
            for trial in range(5):
                scores=rng.lognormal(-5,2,size=n).astype(np.float32)
                labels=rng.choice([0,1],p=[.8,.2],size=n)
                labels[0]=trial%2
                labels[1]=1-labels[0]
                old=omnianomaly.bf_search(scores,labels.copy(),
                    start=np.min(scores),end=np.mean(scores)+3*np.std(scores),
                    step_num=200,verbose=False)
                fast=paper_best_f1(scores,labels)
                for field in ('bf_f1','bf_precision','bf_recall','bf_ROC/AUC',
                              'bf_TP','bf_TN','bf_FP','bf_FN','bf_latency'):
                    self.assertAlmostEqual(old[field],fast[field],places=9,msg=f'{n} {trial} {field}')
                self.assertAlmostEqual(old['bf_threshold'],fast['bf_threshold'],delta=1e-6)

    def test_dense_is_not_lower_than_200_step_grid(self):
        scores=np.r_[np.linspace(.00001,.0001,25),np.array([.1,.15,.2])].astype(np.float32)
        labels=np.r_[np.ones(25),np.zeros(3)]
        bf=paper_best_f1(scores,labels)
        dense=dense_best_f1(scores,labels)
        self.assertGreaterEqual(dense['bf_dense_f1']+1e-8,bf['bf_f1'])
        self.assertTrue(dense['bf_dense_oracle_uses_test_labels'])

    def test_opssat_fragment_boundaries(self):
        scores=np.array([.1,.8,.1,.2,.1,.6,.1,.1,.7,.1],dtype=float)
        labels=np.array([1,1,1,1,0,1,1,0,0,0],dtype=int)
        boundaries=[4,5,7]
        from opssat_pipeline import adjust_per_fragment
        groups=[{'values':np.ones(n),'label':int(labels[start])}
                for start,n in [(0,4),(4,1),(5,2),(7,3)]]
        for thr in [.0,.12,.5,.75,1.0]:
            fast=_stats(scores,labels,[thr],boundaries=boundaries,preserve_upstream_index_zero=False)
            output=adjust_per_fragment(scores,labels,groups,thr)
            tp=int((output*labels).sum());fp=int((output*(1-labels)).sum())
            self.assertEqual(fast['tp'][0],tp)
            self.assertEqual(fast['fp'][0],fp)

    def test_decoder_alpha_equivalence(self):
        import torch
        from torch.utils.data import DataLoader,TensorDataset
        class Tiny:
            def eval(self):return self
            def __call__(self,windows,target):
                return target*.25, target*.75
        windows=torch.linspace(0,1,50,dtype=torch.float32).reshape(10,5,1)
        labels=torch.zeros_like(windows);labels[3:5,-1,:]=1
        class Dset(TensorDataset):
            padding=True
        loader=DataLoader(Dset(windows,labels),batch_size=3)
        result=decoder_arrays(Tiny(),loader,'cpu')
        for alpha in (0,.05,.5,.75,1):
            output=anomaly_scores(result,alpha)
            with torch.no_grad():
                target=windows[:,-1,:].numpy()
                expect=np.mean((alpha*.25*target+(1-alpha)*.75*target-target)**2,axis=1)
            np.testing.assert_allclose(output,expect,atol=1e-7,rtol=1e-6)

if __name__=='__main__':unittest.main()
