"""Recovery from the original V17 bindings without altering completed results."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tranad_plus.core.common import digest, hash_file, write_json
from . import recovery


class RecoveryTests(unittest.TestCase):
    def bindings(self):
        patch_record=json.loads(recovery.PATCH.read_text())
        spec=json.loads(Path(__file__).with_name('plan.json').read_text())
        previous={'sources':{k:v[0] for k,v in patch_record['allowed_before_sources'].items()},
                  'profile':{'dataset':'MSL','seed':2026},'files':{'sha256':'data'},'relocation_sha256':'relocation',
                  'plan':spec,'execution':{'gpu':'A100','torch':'2.8','cpu_threads':8},
                  'parent_signature':'parent','parent_artifacts':{'checkpoint':'original'}}
        current={**deepcopy(previous),'sources':patch_record['after_sources'],
                 'evaluation_policy':recovery.batch_policy(spec,previous['profile'],
                     {'evaluation_settings':{'score_batch_size_by_dataset':{'MSL':128}}})}
        return previous,current

    def prepare(self,folder,complete=False):
        previous,current=self.bindings();signature=digest(previous)
        write_json(folder/'manifest.json',{'signature':signature,'binding':previous})
        write_json(folder/'metrics.json',{'format':'original','precision':.8})
        write_json(folder/'status.json',{'signature':signature,'status':'complete' if complete else 'running',
                                        'artifacts':{name:hash_file(folder/name) for name in ('manifest.json','metrics.json')}})
        return previous,current

    def snapshot(self,folder):
        return {str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob('*') if p.is_file()}

    def test_batch_settings_are_per_format_and_do_not_modify_plan(self):
        original=json.loads(Path(__file__).with_name('plan.json').read_text());before=deepcopy(original)
        for dataset,size in [('OPS-SAT',1024),('SMAP',1024),('MSL',128)]:
            parent={'evaluation_settings':{'score_batch_size_by_dataset':{dataset:size}}}
            profile={'dataset':dataset}
            for name in ('fp32','int8','int4'):
                result=recovery.evaluation_settings(original,profile,parent,name)
                expected=size if name=='fp32' else original['evaluation']['score_batch_size_by_dataset'][dataset]
                self.assertEqual(result['score_batch_size_by_dataset'][dataset],expected)
                self.assertEqual(result['paper_threshold_steps'],200)
                self.assertEqual(result['alpha_steps'],21)
        self.assertEqual(original,before)
        with self.assertRaises(KeyError):recovery.batch_policy(original,{'dataset':'MSL'},{})

    def test_completed_jobs_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);old,new=self.prepare(folder,complete=True);before=self.snapshot(folder)
            result=recovery.inspect_existing(folder,new)
            self.assertEqual(result['action'],'preserve_complete')
            self.assertEqual(result['previous_fp32_batch_size'],64)
            self.assertEqual(before,self.snapshot(folder))
            with self.assertRaises(RuntimeError):recovery.rebind_incomplete(folder,new)
            self.assertEqual(before,self.snapshot(folder))

    def test_known_metadata_predecessor_is_accepted(self):
        old,new=self.bindings();record=json.loads(recovery.PATCH.read_text())
        for name,values in record['allowed_before_sources'].items():
            if len(values)>1:old['sources'][name]=values[-1]
        self.assertEqual(recovery.approved_transition(old,new),'v17_fp32_parent_batch_v1')

    def test_reject_unreviewed_code_data_plan_or_hardware_change(self):
        old,new=self.bindings()
        for kind in ('old_code','new_code','data','plan','hardware','profile'):
            left,right=deepcopy(old),deepcopy(new)
            if kind=='old_code':left['sources']['campaigns/integer/run.py']='unknown'
            elif kind=='new_code':right['sources']['campaigns/integer/run.py']='unknown'
            elif kind=='data':right['files']={'sha256':'changed'}
            elif kind=='plan':right['plan']['evaluation']['paper_threshold_steps']=100
            elif kind=='hardware':right['execution']['gpu']='other'
            else:right['profile']['seed']=2027
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):recovery.approved_transition(left,right)

    def test_reject_changed_artifact_or_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);old,new=self.prepare(folder,True)
            write_json(folder/'metrics.json',{'tampered':True})
            with self.assertRaises(RuntimeError):recovery.inspect_existing(folder,new)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);old,new=self.prepare(folder)
            manifest=json.loads((folder/'manifest.json').read_text());manifest['binding']['files']={'changed':True}
            write_json(folder/'manifest.json',manifest)
            with self.assertRaises(RuntimeError):recovery.inspect_existing(folder,new)

    def test_incomplete_attempt_archived_and_new_binding_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);old,new=self.prepare(folder)
            write_json(folder/'evaluation/fp32/result.json',{'batch':64,'old':True})
            (folder/'models').mkdir();(folder/'models/int8.npz').write_bytes(b'old packed model')
            write_json(folder/'error.json',{'reason':'parity failure'})
            before=self.snapshot(folder)
            result=recovery.rebind_incomplete(folder,new)
            archive=folder/'recovery'/digest(old)
            for name in ('manifest.json','status.json','error.json','evaluation/fp32/result.json','models/int8.npz'):
                self.assertEqual((archive/name).read_bytes(),before[name])
            self.assertFalse((folder/'evaluation').exists())
            self.assertFalse((folder/'models').exists())
            saved=json.loads((folder/'manifest.json').read_text())
            self.assertEqual(saved['signature'],digest(new));self.assertEqual(saved['binding'],new)
            transition=json.loads((archive/'transition.json').read_text())
            for name,expected in transition['archived_artifacts'].items():self.assertEqual(hash_file(archive/name),expected)
            self.assertEqual(recovery.inspect_existing(folder,new)['action'],'resume')
            # Complete it, then an identical retry verifies and skips it.
            write_json(folder/'status.json',{'status':'complete','signature':digest(new),
                                            'artifacts':{'manifest.json':hash_file(folder/'manifest.json')}})
            self.assertEqual(recovery.inspect_existing(folder,new)['action'],'skip_complete')

    def test_interruption_before_manifest_write_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);old,new=self.prepare(folder)
            write_json(folder/'evaluation/fp32/result.json',{'old':True})
            real=recovery.write_json
            def interrupted(path,value):
                if Path(path)==folder/'manifest.json':raise KeyboardInterrupt()
                return real(path,value)
            with patch.object(recovery,'write_json',side_effect=interrupted),self.assertRaises(KeyboardInterrupt):
                recovery.rebind_incomplete(folder,new)
            self.assertEqual(recovery.inspect_existing(folder,new)['action'],'rebind_incomplete')
            recovery.rebind_incomplete(folder,new)
            self.assertEqual(recovery.inspect_existing(folder,new)['action'],'resume')

    def test_interruption_after_manifest_write_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);old,new=self.prepare(folder)
            real=recovery.write_json
            def interrupted(path,value):
                if Path(path)==folder/'status.json':raise KeyboardInterrupt()
                return real(path,value)
            with patch.object(recovery,'write_json',side_effect=interrupted),self.assertRaises(KeyboardInterrupt):
                recovery.rebind_incomplete(folder,new)
            decision=recovery.inspect_existing(folder,new)
            self.assertEqual(decision['action'],'finish_rebind_status')
            write_json(folder/'status.json',{'status':'running','signature':digest(new)})
            self.assertEqual(recovery.inspect_existing(folder,new)['action'],'resume')

    def test_historical_patch_still_pins_unchanged_numerical_sources(self):
        # V17's full source snapshot stays historical as new campaigns are
        # added. Test its numerical dependencies, not a permanently frozen
        # repository file set or the campaign dispatcher.
        from tranad_plus.campaigns.thresholds.provenance import numerical_sources
        record=json.loads(recovery.PATCH.read_text())
        checked=numerical_sources(record['after_sources'])
        self.assertIn('campaigns/integer/runtime.py',checked)
        self.assertIn('core/evaluation/scoring.py',checked)

    def test_reported_210_complete_one_failed_and_29_new(self):
        from .run import arguments, plan
        args=arguments(['--dry-run']);spec=json.loads(args.plan.read_text());jobs=plan(spec,args)
        decisions={};baseline,updated=self.bindings()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for index,job in enumerate(jobs):
                ds,variant,configuration,seed=Path(job).parts
                old,new=deepcopy(baseline),deepcopy(updated)
                old['profile']={'dataset':ds,'variant':variant,'configuration':configuration,'seed':int(seed.split('_')[1])}
                new['profile']=deepcopy(old['profile'])
                new['evaluation_policy']=recovery.batch_policy(spec,new['profile'],
                    {'evaluation_settings':{'score_batch_size_by_dataset':{ds:128 if ds=='MSL' else 1024}}})
                folder=root/job
                if index<=210:
                    signature=digest(old)
                    write_json(folder/'manifest.json',{'signature':signature,'binding':old})
                    write_json(folder/'metrics.json',{'original':True})
                    write_json(folder/'status.json',{'signature':signature,'status':'complete' if index<210 else 'running',
                        'artifacts':{n:hash_file(folder/n) for n in ('manifest.json','metrics.json')}})
                before=self.snapshot(folder)
                action=recovery.inspect_existing(folder,new)['action']
                decisions[action]=decisions.get(action,0)+1
                self.assertEqual(before,self.snapshot(folder))
            self.assertEqual(decisions,{'preserve_complete':210,'rebind_incomplete':1,'new':29})


if __name__=='__main__':unittest.main()
