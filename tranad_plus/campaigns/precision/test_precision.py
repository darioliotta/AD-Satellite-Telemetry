"""Numerical coverage, packed round trips, exact QAT resume and fair plans."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tranad_plus.core.bootstrap import activate_baseline
activate_baseline()
from tranad_plus.core.common import save_torch,write_json
from tranad_plus.core.models import build_model
from tranad_plus.core.data import DataBundle
from tranad_plus.core.training.engine import TrainingInterrupted
from tranad_plus.campaigns.precision.quantization import recipes,apply_qat,master_copy,pack4,unpack4,export_model,load_export
from tranad_plus.campaigns.precision.finetune import fit
from tranad_plus.campaigns.precision.evaluation import evaluate,probe_inputs,verify_parent_endpoint
from tranad_plus.campaigns.precision.run import plan,PROJECT
from tranad_plus.campaigns.precision.resources import analytic_operations
from tranad_plus.campaigns.precision.summarize import summarize
from tranad_plus.core.common import hash_file

torch.set_num_threads(1)
os.environ['V16_NO_PROGRESS']='1'


def profile(variant='encoder_only'):
    return dict(dataset='OPS-SAT',variant=variant,features=1,num_layers=1,dim_feedforward=4,
                window_sz=5,batch_sz=8,epochs=50,seed=2026,lr=.001,pot_q=.01,configuration='compact',identifier='fixture')


class Dataset(torch.utils.data.Dataset):
    def __init__(self,values):self.windows=values
    def __len__(self):return len(self.windows)
    def __getitem__(self,i):return self.windows[i]


def bundle():
    rng=np.random.default_rng(72)
    datasets={split:Dataset(torch.tensor(rng.uniform(.05,.85,(64,5,1)),dtype=torch.float32)) for split in ('train','test')}
    labels={'train':np.zeros(64,dtype=np.int8),'test':np.r_[np.zeros(32,dtype=np.int8),np.ones(32,dtype=np.int8)]}
    groups={split:[{'id':i,'channel':'CADC0872','label':int(labels[split][i*16]),'values':np.zeros(16)} for i in range(4)] for split in datasets}
    return DataBundle(datasets,labels,{'synthetic':True},groups)


class PrecisionTests(unittest.TestCase):
    def test_plan_and_smd_guard(self):
        spec=json.loads((PROJECT/'tranad_plus/campaigns/precision/plan.json').read_text())
        args=argparse.Namespace(seed=None,dataset=None,variant=None,only=None,stage='all')
        jobs=plan(spec,args)
        self.assertEqual(len(jobs),242)
        self.assertEqual(sum(j['kind']=='ptq' for j in jobs),240)
        self.assertEqual(sum(j['kind']=='control' for j in jobs),0)
        self.assertEqual(sum(j['kind']=='qat' for j in jobs),0)
        self.assertEqual(spec['campaigns'],['ptq','smd'])
        self.assertEqual(set(spec['configurations']),{'reference','window_only','ff_only','compact'})
        self.assertTrue(all(j['kind']=='smd' for j in jobs[-2:]))
        for key,value in {'epochs':100,'dim_feedforward':76,'num_layers':5,'batch_sz':8192,'window_sz':20}.items():
            self.assertEqual(jobs[-1]['profile'][key],value)
        spec['smd']['epochs']=500
        with self.assertRaises(ValueError):plan(spec,args)

    def test_compact_storage_estimate_and_budget(self):
        from tranad_plus.campaigns.precision.storage import model_archive_estimate,storage_estimate,check_output_budget,StorageBudgetError
        from tranad_plus.campaigns.precision.run import arguments
        part=model_archive_estimate(profile(),['fp16','int8','int4'])
        estimate=storage_estimate([part],3)
        self.assertEqual(estimate['format_evaluations'],4)
        self.assertFalse(estimate['pointwise_arrays_saved'])
        self.assertLess(estimate['conservative_total_GiB'],.05)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'fixture').write_bytes(b'x'*1024)
            self.assertEqual(check_output_budget(root,3),1024)
            with self.assertRaises(StorageBudgetError):check_output_budget(root,1e-8)
            self.assertTrue((root/'fixture').exists())
        with self.assertRaises(SystemExit):arguments(['--stage','qat'])

    def test_preflight_streaming_rejects_nonfinite_inputs(self):
        from tranad_plus.campaigns.precision.run import check_input_arrays
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);values=np.zeros((20000,1),dtype=np.float32)
            np.save(root/'track.npy',values)
            files={'files':[{'file':'track.npy'}]}
            check_input_arrays(files,root)
            values[-1]=np.nan;np.save(root/'track.npy',values)
            with self.assertRaises(ValueError):check_input_arrays(files,root)

    def test_int4_pack_odd_shapes(self):
        codes=np.array([-7,-6,-1,0,1,6,7],dtype=np.int8)
        packed=pack4(codes)
        self.assertEqual(packed.nbytes,4)
        np.testing.assert_array_equal(unpack4(packed,len(codes)),codes)

    def test_all_architectures_all_attention_matrices_roundtrip(self):
        for variant in ('reference','shared_decoder','single_pass','encoder_only'):
            model=build_model(profile(variant)).eval()
            pristine={k:v.clone() for k,v in model.state_dict().items()}
            for bits in (8,4):
                scales=recipes(model,bits)
                self.assertIn('transformer_encoder.layers.0.self_attn.in_proj_weight',scales)
                self.assertIn('transformer_encoder.layers.0.self_attn.out_proj.weight',scales)
                self.assertIn('fcn.0.weight',scales)
                self.assertFalse(any(k.endswith('bias') for k in scales))
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'packed.npz'
                    stats=export_model(model,path,profile(variant),'int'+str(bits),scales)
                    measured=export_model(model,None,profile(variant),'int'+str(bits),scales,measure_only=True)
                    self.assertEqual(measured['artifact_file_bytes'],stats['artifact_file_bytes'])
                    restored=load_export(path,'cpu')
                    self.assertEqual(stats['matrix_coverage_fraction'],1)
                    for name,param in model.named_parameters():
                        actual=dict(restored.named_parameters())[name]
                        expected=torch.fake_quantize_per_tensor_affine(param,scales[name],0,-(2**(bits-1)-1),2**(bits-1)-1) if name in scales else param
                        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
                    if variant=='shared_decoder':self.assertIs(restored.transformer_decoder1,restored.transformer_decoder2)
                    x=torch.rand(5,8,1);output=restored(x,x[-1:])
                    for v in output if isinstance(output,tuple) else (output,):self.assertTrue(torch.isfinite(v).all())
            for key,value in model.state_dict().items():torch.testing.assert_close(value,pristine[key],rtol=0,atol=0)

    def test_fp32_and_fp16_storage(self):
        p=profile();model=build_model(p).eval();x=torch.rand(5,4,1)
        with tempfile.TemporaryDirectory() as tmp:
            a=export_model(model,Path(tmp)/'a.npz',p,'fp32');b=export_model(model,Path(tmp)/'b.npz',p,'fp16')
            self.assertEqual(a['tensor_bytes_including_bias_scales_buffers'],2*b['tensor_bytes_including_bias_scales_buffers'])
            restored=load_export(Path(tmp)/'a.npz','cpu').eval()
            torch.testing.assert_close(restored(x,x[-1:]),model(x,x[-1:]),rtol=0,atol=0)
            half=load_export(Path(tmp)/'b.npz','cpu')
            self.assertTrue(all(p.dtype==torch.float16 for p in half.parameters()))

    def test_qat_gradient_and_shared_alias(self):
        for variant in ('reference','shared_decoder','single_pass','encoder_only'):
            model=build_model(profile(variant));scales=recipes(model,4);apply_qat(model,scales,4)
            x=torch.rand(5,8,1);outputs=model(x,x[-1:]);outputs=outputs if isinstance(outputs,tuple) else (outputs,)
            sum(v.square().mean() for v in outputs).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if variant=='shared_decoder':self.assertIs(model.transformer_decoder1,model.transformer_decoder2)
            master=master_copy(model)
            self.assertFalse(any('parametrizations' in k for k in master.state_dict()))
            # A second checkpoint/format export must leave the live QAT model usable.
            with tempfile.TemporaryDirectory() as tmp:
                for i in range(3):
                    export_model(model,Path(tmp)/f'round_{i}.npz',profile(variant),'int4',scales)
                    values=model(x,x[-1:]);values=values if isinstance(values,tuple) else (values,)
                    self.assertTrue(all(torch.isfinite(v).all() for v in values))

    def test_exact_qat_and_control_resume(self):
        p=profile();base=build_model(p).state_dict();data=bundle();settings={'epochs':4,'lr_multiplier':.1,'step_size':2,'gamma':.9}
        for bits in (None,8,4):
            with tempfile.TemporaryDirectory() as tmp:
                full,resumed=Path(tmp)/'full',Path(tmp)/'resumed'
                a=fit(p,base,data,full,'same',torch.device('cpu'),settings,bits=bits)
                with self.assertRaises(TrainingInterrupted):fit(p,base,data,resumed,'same',torch.device('cpu'),settings,bits=bits,stop_after_epoch=2)
                b=fit(p,base,data,resumed,'same',torch.device('cpu'),settings,bits=bits)
                for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
                records=[json.loads(line) for line in (resumed/'epochs.jsonl').read_text().splitlines()]
                self.assertEqual([r['absolute_epoch'] for r in records],[51,52,53,54])
                self.assertEqual(len(records),4)
                # Crash after final recovery, before the final file is written.
                (resumed/'checkpoints/final.ckpt').unlink()
                fit(p,base,data,resumed,'same',torch.device('cpu'),settings,bits=bits)
                self.assertTrue((resumed/'checkpoints/final.ckpt').exists())

    def test_budget_extension_preserves_optimizer_rng_and_milestones(self):
        p=profile();base=build_model(p).state_dict();data=bundle()
        settings={'epochs':6,'milestones':[2,4,6],'lr_multiplier':.1,'step_size':2,'gamma':.9}
        for bits in (None,4):
            with tempfile.TemporaryDirectory() as tmp:
                full,partial=Path(tmp)/'full',Path(tmp)/'partial'
                a=fit(p,base,data,full,'same',torch.device('cpu'),settings,bits=bits)
                fit(p,base,data,partial,'same',torch.device('cpu'),settings,bits=bits,target_epoch=2)
                first=(partial/'milestones/epoch_002/checkpoint.ckpt').read_bytes()
                # Evaluation/format construction may consume RNG before extension.
                torch.rand(100);np.random.rand(100)
                b=fit(p,base,data,partial,'same',torch.device('cpu'),settings,bits=bits,target_epoch=6)
                self.assertEqual(first,(partial/'milestones/epoch_002/checkpoint.ckpt').read_bytes())
                for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
                self.assertTrue(all((partial/f'milestones/epoch_{e:03d}/checkpoint.ckpt').exists() for e in (2,4,6)))

    def test_analytic_operations_match_archived_counts(self):
        root=PROJECT/'results/v13/architectures/OPS-SAT'
        for variant in ('reference','shared_decoder','single_pass','encoder_only'):
            path=root/variant/'reference/seed_2026/metrics.json'
            if not path.exists():continue
            report=json.loads(path.read_text());counts=analytic_operations(report['profile'])
            self.assertEqual(counts['matmul_MACs_per_window'],report['resources']['analytical_matmul_macs_per_window'])
            self.assertEqual(counts['matmul_FLOPs_per_window'],2*counts['matmul_MACs_per_window'])

    def test_summary_pairs_equal_finetuning_budgets(self):
        import csv
        def result(value):
            return {'oracle':{'bf':{'bf_f1':value}},'parent_locked':{},
                    'storage':{'tensor_bytes_including_bias_scales_buffers':10,'artifact_file_bytes':20,
                               'matrix_coverage_fraction':1,'execution':'dequantized_fp32_simulation'},
                    'resources':{'parameter_bytes_resident':40,'median_ms':1,'p95_ms':2}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            reports=[{'stage':'ptq','kind':'ptq','profile':profile(),'formats':{'int4':result(.8)}},
                     {'stage':'qat','kind':'control','profile':profile(),'milestones':{
                         'epoch_002':{'extra_epochs':2,'formats':{'int4':result(.70)}},
                         'epoch_004':{'extra_epochs':4,'formats':{'int4':result(.75)}}}},
                     {'stage':'qat','kind':'qat','profile':profile(),'milestones':{
                         'epoch_002':{'extra_epochs':2,'formats':{'int4':result(.72)}},
                         'epoch_004':{'extra_epochs':4,'formats':{'int4':result(.79)}}}}]
            for i,r in enumerate(reports):
                folder=root/str(i);write_json(folder/'metrics.json',r)
                write_json(folder/'status.json',{'status':'complete','artifacts':{'metrics.json':hash_file(folder/'metrics.json')}})
            summarize(root)
            with (root/'tables/paired_deltas.csv').open() as stream:rows=list(csv.DictReader(stream))
            paired={int(r['extra_epochs']):float(r['delta_pp']) for r in rows if r['comparison']=='matched_FP32_training_then_PTQ'}
            self.assertAlmostEqual(paired[2],2.);self.assertAlmostEqual(paired[4],4.)

    def test_probe_invariants(self):
        values,rows=probe_inputs(10)
        self.assertEqual(len(values),45)
        for x,row in zip(values,rows):
            self.assertAlmostEqual(float(x[-1,0]),row['current'],places=6)
            self.assertAlmostEqual(float(x[:-1,0].mean()),row['current'],places=6)
            self.assertAlmostEqual(float(np.std(x[:-1,0])),row['rms'],places=6)
            self.assertTrue(((x>=0)&(x<=1)).all())

    def test_ptq_paired_deltas_match_same_parent(self):
        import csv
        def result(value,size):
            return {'oracle':{'bf':{'bf_f1':value}},'parent_locked':{},
                    'storage':{'tensor_bytes_including_bias_scales_buffers':size,'artifact_file_bytes':size+10,
                               'matrix_coverage_fraction':1,'execution':'dequantized_fp32_simulation'},
                    'resources':{'parameter_bytes_resident':40,'median_ms':1,'p95_ms':2}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'ptq/fixture'
            write_json(folder/'metrics.json',{'stage':'ptq','kind':'ptq','profile':profile(),
                                             'formats':{'fp32':result(.8,40),'int4':result(.75,10)}})
            write_json(folder/'status.json',{'status':'complete','artifacts':{'metrics.json':hash_file(folder/'metrics.json')}})
            summarize(root)
            with (root/'tables/paired_deltas.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['comparison'],'PTQ_vs_FP32')
            self.assertAlmostEqual(float(rows[0]['delta_pp']),-5.)
            self.assertAlmostEqual(float(rows[0]['tensor_compression_ratio']),4.)

    def test_parent_verification_records_roundoff_but_rejects_protocol_changes(self):
        from tranad_plus.core.evaluation.metrics import fixed_metrics
        labels=np.array([0,0,1,1]);old=np.array([.1,.5,.7,.8],dtype=np.float32)
        scores=old.copy();scores[1]=np.nextafter(scores[1],np.float32(1))
        original=fixed_metrics(old,labels,.5,'pot');current=fixed_metrics(scores,labels,.5,'pot')
        checked=verify_parent_endpoint(scores,old,labels,.5,'pot',None,original,current)
        self.assertEqual(checked['direct_alarm_boundary_flips'],1)
        self.assertFalse(checked['exact_confusion_counts_match'])
        with self.assertRaises(RuntimeError):
            verify_parent_endpoint(scores,old,labels,.5,'pot',None,{**original,'pot_FP':99},current)
        with self.assertRaises(AssertionError):
            verify_parent_endpoint(scores+.01,old,labels,.5,'pot',None,original,current)

    def test_native_smd_resume_matches_uninterrupted_training(self):
        from tranad_plus.core.training import engine
        from tranad_plus.campaigns.precision.progress import native_training
        from tranad_plus.campaigns.temporal.run import reconcile_epochs
        p={**profile('reference'),'dataset':'SMD','epochs':3};data=bundle()
        with tempfile.TemporaryDirectory() as tmp:
            full,partial=Path(tmp)/'full',Path(tmp)/'partial'
            with native_training(full,p,torch.device('cpu')):
                engine.train(p,data,full,'native-resume',torch.device('cpu'))
            with self.assertRaises(TrainingInterrupted):
                with native_training(partial,p,torch.device('cpu')):
                    engine.train(p,data,partial,'native-resume',torch.device('cpu'),stop_after_epoch=1)
            reconcile_epochs(partial)
            with native_training(partial,p,torch.device('cpu')):
                engine.train(p,data,partial,'native-resume',torch.device('cpu'))
            a=torch.load(full/'checkpoints/final.ckpt',map_location='cpu',weights_only=True)
            b=torch.load(partial/'checkpoints/final.ckpt',map_location='cpu',weights_only=True)
            for key,value in a['model_state'].items():torch.testing.assert_close(value,b['model_state'][key],rtol=0,atol=0)
            self.assertEqual(a['scheduler_state'],b['scheduler_state'])

    def test_native_training_saves_raw_losses_and_restores_hooks(self):
        from tranad_plus.core.training import engine
        from tranad_plus.campaigns.precision.progress import native_training
        p={**profile('reference'),'dataset':'SMD','epochs':2}
        original=engine.reconstruction_loss
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with native_training(folder,p,torch.device('cpu')):
                engine.train(p,bundle(),folder,'native-fixture',torch.device('cpu'))
            self.assertIs(engine.reconstruction_loss,original)
            history=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
            self.assertEqual(len((folder/'epoch_resources.jsonl').read_text().splitlines()),2)
            for row in history:
                with np.load(folder/f"batch_losses/epoch_{row['epoch_one_based']:03d}.npz") as saved:
                    self.assertEqual(len(saved['loss']),row['batches'])
                    self.assertAlmostEqual(float(saved['loss'].astype(np.float64).mean()),row['loss_mean_of_batch_means'],places=7)

    def test_end_to_end_quantized_evaluation(self):
        p=profile();data=bundle();model=build_model(p)
        settings={'score_batch_size_by_dataset':{'OPS-SAT':16},'latency_warmup':1,'latency_iterations':2}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'packed.npz';storage=export_model(model,path,p,'int4')
            restored=load_export(path,'cpu')
            folder=Path(tmp)/'job/evaluation/int4'
            result=evaluate(restored,data,p,folder,torch.device('cpu'),settings,None,None,PROJECT/'data',storage)
            self.assertIn('bf',result['oracle']);self.assertIn('pot',result['oracle']);self.assertNotIn('bf_dense',result['oracle'])
            self.assertTrue(result['resources']['not_integer_kernel_latency'])
            self.assertIn('analytic_operations',result['resources'])
            self.assertEqual(result['resources']['profiler']['status'],'measured')
            self.assertTrue((folder/'latency_samples.npz').exists())
            self.assertTrue((folder/'operators.json').exists())
            self.assertFalse((folder/'predictions.npz').exists())
            self.assertFalse((folder/'reconstructions.npz').exists())
            self.assertFalse((folder/'scores.npz').exists())
            self.assertEqual(result['archive_policy'],'compact_summary_only')
            cached=evaluate(restored,data,p,folder,torch.device('cpu'),settings,None,None,PROJECT/'data',storage)
            self.assertEqual(result['oracle'],cached['oracle'])


if __name__=='__main__':unittest.main()
