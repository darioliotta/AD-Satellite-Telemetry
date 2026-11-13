import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tranad_plus.core.bootstrap import activate_baseline,PROJECT_ROOT
from tranad_plus.core.common import digest,hash_file,write_json
activate_baseline()
from .metrics import point_metrics,bf200,select,comparison
from .provenance import assert_score_statistics
from .run import arguments,plan,validate_output,complete,status
from .summarize import summarize


def reference_example():
    labels=np.array([0,1,1,1,1,0,0,0,0,0],dtype=np.int8)
    grid=np.array([[0,1,0,0,0,0,0,0,0,0],
                   [0,.8,.8,.8,.8,.8,0,0,0,0]],dtype=np.float32)
    records=[{'alpha':0.,'pot_threshold':.5},{'alpha':1.,'pot_threshold':.5}]
    from tranad_plus.core.evaluation.metrics import fixed_metrics
    oracle={}
    for method,threshold in [('bf',0.),('pot',.5)]:
        metric=fixed_metrics(grid[0],labels,threshold,method)
        raw=point_metrics(labels,grid[0]>threshold)
        metric.update(alpha=0.,raw_without_PA={method+'_'+k:raw[v] for k,v in [('f1','f1'),('TP','TP'),('TN','TN'),('FP','FP'),('FN','FN')]})
        oracle[method]=metric
    return grid,labels,records,{'oracle':oracle}


class ThresholdTests(unittest.TestCase):
    def test_counts_and_binary_auc(self):
        r=point_metrics([0,0,1,1],[0,1,1,1])
        self.assertEqual((r['TP'],r['TN'],r['FP'],r['FN']),(2,1,1,0))
        self.assertAlmostEqual(r['f1'],.8);self.assertAlmostEqual(r['auc_binary'],.75)

    def test_bf200_matches_direct_sweep_with_ties(self):
        rng=np.random.default_rng(30)
        for n in (10,137,512):
            scores=np.round(rng.random(n),1).astype(np.float32);labels=np.resize([0,1],n)
            thresholds=np.linspace(float(scores.min()),float(scores.mean()+3*scores.std()),200)
            direct=[point_metrics(labels,scores>t)['f1'] for t in thresholds]
            result=bf200(scores,labels)
            self.assertEqual(result['threshold_index'],int(np.argmax(direct)))
            self.assertAlmostEqual(result['f1'],max(direct))

    def test_selection_optimizes_raw_not_adjusted_and_keeps_pot_thresholds(self):
        grid,y,records,reference=reference_example()
        with patch('tranad_plus.core.evaluation.metrics.fixed_metrics',side_effect=AssertionError('PA used during selection')):
            chosen,candidates=select(grid,y,records)
        for method in ('bf','pot'):
            self.assertEqual(chosen[method]['alpha'],1.)
            self.assertAlmostEqual(chosen[method]['f1'],8/9)
        self.assertEqual(chosen['pot']['threshold'],records[1]['pot_threshold'])
        result=comparison(grid,y,records,reference)
        views=result['protocols']
        for method in ('bf','pot'):
            self.assertAlmostEqual(views['no_pa_at_pa_selection'][method]['f1'],.4)
            self.assertAlmostEqual(views['no_pa_optimized'][method]['f1'],8/9)
            self.assertGreater(views['pa_at_pa_selection'][method]['f1'],views['pa_at_no_pa_selection'][method]['f1'])
            self.assertTrue(result['v17_endpoint_parity'][method]['raw_confusion_counts_match_v17'])
            self.assertTrue(result['v17_endpoint_parity'][method]['PA_confusion_counts_match_v17'])

    def test_single_output_and_missing_or_invalid_pot(self):
        grid,y,_,_=reference_example()
        result,_=select(grid[:1],y,[{'alpha':None,'pot_threshold':.5}])
        self.assertIsNone(result['bf']['alpha']);self.assertIsNone(result['pot']['alpha'])
        with self.assertRaises(RuntimeError):select(grid[:1],y,[{'alpha':None,'pot_threshold':None}])
        with self.assertRaises(ValueError):select(grid[:1],y,[{'alpha':None,'pot_threshold':np.inf}])
        invalid=grid.copy();invalid[0,0]=np.nan
        with self.assertRaises(FloatingPointError):select(invalid,y,[{'alpha':0.,'pot_threshold':.5},{'alpha':1.,'pot_threshold':.5}])

    def test_plan_balance_filters_and_protected_outputs(self):
        args=arguments(['--dry-run']);spec=json.loads(args.plan.read_text());jobs=plan(spec,args)
        self.assertEqual(len(jobs),720);self.assertEqual(len(set(jobs)),720)
        for i,key,count in [(0,'datasets',240),(1,'variants',180),(2,'configurations',180),(4,'formats',240)]:
            for value in spec[key]:self.assertEqual(sum(Path(j).parts[i]==value for j in jobs),count)
        chosen=plan(spec,arguments(['--dataset','MSL','--seed','2026','--format','int8']))
        self.assertEqual(len(chosen),16)
        for folder in ('results/v17','results/v13','data','tranad_plus','results'):
            with self.assertRaises(ValueError):validate_output(arguments(['--output-root',str(PROJECT_ROOT/folder)]))
        wrong=json.loads(args.plan.read_text());wrong['evaluation']['bf_threshold_steps']=201
        with self.assertRaises(ValueError):plan(wrong,args)

    def test_statistics_parity_and_rejection(self):
        grid,y,records,reference=reference_example()
        saved=[dict(alpha=r['alpha'],mean=float(s.mean()),std=float(s.std()),min=float(s.min()),max=float(s.max()),
                    quantiles_50_90_99_999=np.quantile(s,[.5,.9,.99,.999]).tolist()) for s,r in zip(grid,records)]
        ref={'score_statistics_by_split':{'test':saved}}
        result=assert_score_statistics(grid,records,ref,5e-4,1e-7)
        self.assertTrue(result['score_statistics_within_v17_tolerance'])
        with self.assertRaises(AssertionError):assert_score_statistics(grid+1,records,ref,5e-4,1e-7)

    def test_compact_output_verification_resume_and_summary(self):
        grid,y,records,reference=reference_example();evaluation=comparison(grid,y,records,reference)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folders=[]
            for seed in range(2026,2031):
                for fmt in ('fp32','int8'):
                    folder=root/'OPS-SAT/encoder_only/compact'/f'seed_{seed}'/fmt;folders.append(folder)
                    profile=dict(dataset='OPS-SAT',variant='encoder_only',configuration='compact',seed=seed,window_sz=5,dim_feedforward=4)
                    binding={'seed':seed,'format':fmt};signature=digest(binding)
                    write_json(folder/'manifest.json',{'binding':binding,'signature':signature})
                    write_json(folder/'metrics.json',{'profile':profile,'format':fmt,'evaluation':evaluation,
                                                     'inference':{'seconds_including_transfer':.1,'batch_size':16}})
                    artifacts={name:hash_file(folder/name) for name in ('manifest.json','metrics.json')}
                    write_json(folder/'status.json',{'status':'complete','signature':signature,'artifacts':artifacts})
                    self.assertTrue(complete(folder,signature))
            summary=summarize(root)
            self.assertEqual(summary,{'per_seed_rows':80,'mean_std_rows':16,'selection_comparisons':20,'precision_comparisons':40})
            self.assertTrue((root/'tables/selection_comparison.csv').exists())
            self.assertFalse(list(root.rglob('*.npz')))
            with self.assertRaises(RuntimeError):complete(folders[0],'wrong_signature')
            write_json(folders[0]/'metrics.json',{'modified':True})
            with self.assertRaises(RuntimeError):complete(folders[0])

    def test_existing_v17_numerical_sources_remain_compatible(self):
        source=PROJECT_ROOT/'results/v17/integer/OPS-SAT/reference/reference/seed_2026/manifest.json'
        if not source.exists():self.skipTest('Local V17 source results not available')
        from .provenance import numerical_sources
        previous=json.loads(source.read_text())['binding']['sources']
        checked=numerical_sources(previous)
        self.assertIn('campaigns/integer/runtime.py',checked)
        changed=dict(previous);changed['campaigns/integer/runtime.py']='unknown'
        with self.assertRaises(RuntimeError):numerical_sources(changed)

    def test_actual_model_outputs_through_no_pa_selection_all_variants_and_precisions(self):
        import torch
        from copy import deepcopy
        from tranad_plus.core.models import build_model
        from tranad_plus.core.evaluation.scoring import collect_outputs,reconstruction_scores
        from tranad_plus.core.evaluation.metrics import fixed_metrics
        from tranad_plus.campaigns.integer.runtime import convert,export,load
        torch.set_num_threads(1)
        class Dataset(torch.utils.data.Dataset):
            def __init__(self):self.values=torch.rand(24,5,3)
            def __len__(self):return len(self.values)
            def __getitem__(self,i):return self.values[i]
        labels=np.r_[np.zeros(12,dtype=np.int8),np.ones(12,dtype=np.int8)]
        with tempfile.TemporaryDirectory() as tmp:
            for variant in ('reference','shared_decoder','single_pass','encoder_only'):
                profile=dict(variant=variant,features=3,window_sz=5,batch_sz=8,num_layers=1,dim_feedforward=7)
                torch.manual_seed(2026);original=build_model(profile);data=Dataset()
                for fmt in ('fp32','int8','int4'):
                    model=deepcopy(original)
                    if fmt!='fp32':
                        bits=int(fmt[3:]);model=convert(model,bits,'reference_cpu')
                        path=Path(tmp)/'model.npz';export(model,profile,bits,path)
                        model=load(path,'cpu','reference_cpu')
                    outputs=collect_outputs(model,data,torch.device('cpu'),8,'V18 CPU integration',Path(tmp)/'invalid.json')
                    alphas=np.linspace(0,1,21).tolist() if model.output_count==2 else [None]
                    grid=np.stack([reconstruction_scores(outputs,a) for a in alphas])
                    # Synthetic pre-fitted thresholds test reuse and selection;
                    # this test does not claim to fit or validate SPOT itself.
                    records=[{'alpha':a,'pot_threshold':float(np.quantile(s,.9))} for a,s in zip(alphas,grid)]
                    oracle={}
                    for method in ('bf','pot'):
                        threshold=records[0]['pot_threshold'];endpoint=fixed_metrics(grid[0],labels,threshold,method)
                        direct=point_metrics(labels,grid[0]>threshold)
                        endpoint.update(alpha=alphas[0],raw_without_PA={method+'_'+k:direct[k] for k in ('f1','TP','TN','FP','FN')})
                        oracle[method]=endpoint
                    result=comparison(grid,labels,records,{'oracle':oracle})
                    self.assertEqual(len(result['alpha_search']),len(alphas))
                    for method in ('bf','pot'):
                        winner=result['protocols']['no_pa_optimized'][method]
                        idx=alphas.index(winner['alpha']);expected=point_metrics(labels,grid[idx]>winner['threshold'])
                        self.assertEqual(winner['TP'],expected['TP']);self.assertAlmostEqual(winner['f1'],expected['f1'])
                        self.assertTrue(result['v17_endpoint_parity'][method]['raw_confusion_counts_match_v17'])


if __name__=='__main__':unittest.main()
