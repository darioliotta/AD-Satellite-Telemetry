"""v11 additions: uniform logging, batching, split audit, optimizer contracts."""
import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from campaign_v12 import make_plan
from console_telemetry import epoch_line
from gpu_batches import training_batches
from gpu_settings import make_adam, select_optimizer, sync_upstream_optimizer
from tools.audit_splits import audit_native, audit_opssat


class V12NewContracts(unittest.TestCase):
    def test_v10_aligned_split_contract_and_isolation(self):
        obj=make_plan()
        self.assertEqual(len(obj['experiments']),126)
        self.assertTrue(all(x['identifier'].startswith('toast12_') for x in obj['experiments']))
        self.assertEqual({x['validation_fraction'] for x in obj['experiments'] if 'opssat' in x['identifier']},{0.2})
        self.assertTrue(obj['measurement']['paper_alpha_search'])
        self.assertEqual({x['base_profile'] for x in obj['experiments']},
                         {'opssat_shared_ws10_1f','smap_plus_ws10','msl_plus_ws50_dff2x'})

    def test_console_uniform_and_speed_units(self):
        for ds in ('OPS-SAT','MSL','SMAP'):
            out=epoch_line(dataset=ds,identifier='ex',epoch=1,epochs=50,
                          loss=0.123456,lr=0.001,seconds=2,batches=4,windows=64)
            self.assertIn('[TRAIN]',out)
            self.assertIn('001/050',out)
            self.assertIn('32 win/s',out)
            self.assertIn('GPU peak=n/a',out)

    def test_cpu_batch_order_and_last_partial_batch(self):
        x=torch.arange(21,dtype=torch.float32).reshape(7,3)
        ds=TensorDataset(x)
        # standard Dataset returns tuples, but only the loader ordering is under test
        loaders, mode=training_batches(ds,batch_size=3,shuffle=False,device='cpu')
        self.assertEqual(mode,'cpu')
        self.assertEqual([b[0].size(0) for b in loaders],[3,3,1])
        np.testing.assert_allclose(torch.cat([b[0] for b in loaders]).numpy(),x.numpy())

    def test_fused_requires_cuda_and_standard_unchanged(self):
        old=sys.modules.get('gpu_settings')
        from unittest.mock import patch
        model=torch.nn.Linear(2,2)
        with patch.dict('os.environ',{'TRANADPLUS_ADAM_IMPL':'standard'}):
            opt=torch.optim.Adam(model.parameters(),lr=0.001)
            sch=torch.optim.lr_scheduler.StepLR(opt,5)
            a,b=select_optimizer(model,opt,sch)
            self.assertIs(a,opt);self.assertIs(b,sch)
        with patch.dict('os.environ',{'TRANADPLUS_ADAM_IMPL':'fused'}):
            with self.assertRaises(ValueError):
                make_adam(model.parameters(),lr=0.001,device='cpu')

    def test_replacement_optimizer_state_is_reflected_in_upstream_checkpoint(self):
        model=torch.nn.Linear(2,1)
        upstream=torch.optim.Adam(model.parameters(),lr=0.001)
        active=torch.optim.Adam(model.parameters(),lr=0.001)
        output=model(torch.ones(2,2)).sum();output.backward();active.step()
        self.assertFalse(bool(upstream.state_dict()['state']))
        sync_upstream_optimizer(upstream,active)
        self.assertEqual(len(upstream.state_dict()['state']),len(active.state_dict()['state']))
        self.assertGreater(len(upstream.state_dict()['state']),0)

    def test_native_split_audit_rejects_anomalous_training(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)/'processed'/'tranad_plus'/'MSL'
            for split in ('Train','Test'):
                for kind in ('Tracks','Labels'):(base/split/kind).mkdir(parents=True)
                np.save(base/split/'Tracks'/'one.npy',np.ones((12,2),dtype=np.float32))
                np.save(base/split/'Labels'/'one.npy',np.zeros((12,),dtype=np.int8))
            data=audit_native('MSL',td)
            self.assertFalse(data['validation']['exists'])
            self.assertEqual(data['splits']['train']['labelled_anomalous_points'],0)
            np.save(base/'Train'/'Labels'/'one.npy',np.ones((12,),dtype=np.int8))
            with self.assertRaises(ValueError):audit_native('MSL',td)

    def test_ops_split_audit_counts_held_out_anomalies(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)/'raw'/'OPSSAT';d.mkdir(parents=True)
            pd.DataFrame({'segment':[1,1,2,2,3,3], 'channel':['A']*6,
                         'train':[1,1,1,1,0,0], 'anomaly':[0,0,1,1,1,1]}).to_csv(d/'segments.csv',index=False)
            report=audit_opssat(td)
            self.assertEqual(report['splits']['train']['fragments'],1)
            self.assertEqual(report['splits']['excluded_from_training']['fragments'],1)
            self.assertEqual(report['splits']['test']['fragments'],1)

if __name__=='__main__':unittest.main()
