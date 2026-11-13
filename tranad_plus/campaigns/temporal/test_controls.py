"""Target invariance, protocol parity and exact epoch-resume regression checks."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tranad_plus.campaigns.temporal.controls import transform, Windows, changed_bundle
from tranad_plus.campaigns.temporal.run import build_plan, complete, reconcile_epochs, main, PROJECT
from tranad_plus.core.common import hash_file, write_json
from tranad_plus.core.data import DataBundle
from tranad_plus.core.training.engine import train, TrainingInterrupted
from tranad_plus.campaigns.temporal.evaluation import bf200, oracle, trained_evaluation

torch.set_num_threads(1)
os.environ['TOAST_NO_TQDM'] = '1'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.windows = windows
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, i):
        return self.windows[i]


def tiny_profile():
    return dict(dataset='OPS-SAT', features=1, variant='encoder_only', configuration='test',
                seed=2026, identifier='test_v15', window_sz=5, dim_feedforward=8, num_layers=1,
                batch_sz=8, epochs=4, lr=.001, pot_q=.01, input_condition='shuffle_1101')


def tiny_bundle():
    rng = np.random.default_rng(42)
    ds = {key: Dataset(torch.tensor(rng.uniform(.05, .8, (32, 5, 1)), dtype=torch.float32))
          for key in ('train', 'test', 'validation')}
    labels = {key: np.r_[np.zeros(16, dtype=np.int8), np.ones(16, dtype=np.int8)] for key in ds}
    groups = {key: [{'id': 0, 'channel': 'test', 'label': 0, 'values': np.zeros(16)},
                    {'id': 1, 'channel': 'test', 'label': 1, 'values': np.zeros(16)}] for key in ds}
    return DataBundle(ds, labels, {'fixture': 'synthetic'}, groups)


class HistoryTests(unittest.TestCase):
    def test_target_multiset_and_feature_alignment(self):
        x = torch.arange(20*10*3).reshape(20, 10, 3).float()
        original = x.clone()
        for condition in ('shuffle_1101', 'shuffle_2202', 'reverse_past', 'repeat_current'):
            y = transform(x, condition)
            torch.testing.assert_close(y[:, -1], x[:, -1], rtol=0, atol=0)
            if condition != 'repeat_current':
                torch.testing.assert_close(y[:, :-1].sort(dim=1).values, x[:, :-1].sort(dim=1).values, rtol=0, atol=0)
                # All features of a past telemetry row travel together.
                torch.testing.assert_close(y[:, :, 1]-y[:, :, 0], torch.ones_like(y[:, :, 0]))
        torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_private_rng_and_reproducibility(self):
        x = torch.arange(200).reshape(20, 10, 1).float()
        torch.manual_seed(17)
        np.random.seed(18)
        t, n = torch.get_rng_state().clone(), np.random.get_state()
        a = transform(x, 'shuffle_1101')
        torch.testing.assert_close(a, transform(x, 'shuffle_1101'), rtol=0, atol=0)
        self.assertFalse(torch.equal(a, transform(x, 'shuffle_2202')))
        self.assertFalse(torch.equal(a[0, :-1, 0]-a[0, 0, 0], a[1, :-1, 0]-a[1, 0, 0]))
        self.assertTrue(torch.equal(torch.get_rng_state(), t))
        np.testing.assert_array_equal(np.random.get_state()[1], n[1])

    def test_no_history_and_reverse(self):
        x = torch.randn(4, 1, 2)
        for condition in ('original', 'shuffle_1101', 'repeat_current', 'reverse_past'):
            torch.testing.assert_close(transform(x, condition), x, rtol=0, atol=0)
        y = torch.arange(5.).reshape(1, 5, 1)
        self.assertEqual(transform(y, 'reverse_past').ravel().tolist(), [3, 2, 1, 0, 4])

    def test_native_window_adapter(self):
        class Native:
            window_size = 4
            big_track = torch.arange(24.).reshape(12, 2)
        ds = Windows(Native(), 'shuffle_1101')
        self.assertEqual(ds.windows.shape, (9, 4, 2))
        torch.testing.assert_close(ds.windows[:, -1], Native.big_track[3:], rtol=0, atol=0)

    def test_plan_counts_order_and_unique_ids(self):
        spec = json.loads((PROJECT/'tranad_plus/campaigns/temporal/plan.json').read_text())
        args = argparse.Namespace(seed=None, stage='all', only=None)
        jobs = build_plan(spec, args)
        self.assertEqual(len(jobs), 222)
        self.assertEqual(len({j['id'] for j in jobs}), 222)
        self.assertEqual([j['stage'] for j in jobs[-2:]], ['smd', 'smd'])
        self.assertEqual(sum(j['stage']=='frozen' for j in jobs), 120)
        self.assertEqual(sum(j['stage']=='train' for j in jobs), 100)
        self.assertEqual(sum(j.get('profile', {}).get('window_sz')==1 for j in jobs), 20)
        self.assertTrue(all(j['profile']['epochs']==500 for j in jobs[-2:]))
        for variant in ('reference', 'shared_decoder', 'single_pass', 'encoder_only'):
            for dataset in ('OPS-SAT', 'SMAP', 'MSL'):
                self.assertEqual(sum(j['stage']=='frozen' and Path(j['parent']).parts[:2]==(dataset, variant)
                                     for j in jobs), 10)
            self.assertEqual(sum(j['stage']=='train' and j['profile']['variant']==variant for j in jobs), 25)

    def test_smd_profile_matches_original_benchmark(self):
        spec = json.loads((PROJECT/'tranad_plus/campaigns/temporal/plan.json').read_text())
        benchmark = json.loads((PROJECT/'tranad_plus/campaigns/temporal/smd_benchmark.json').read_text())
        jobs = build_plan(spec, argparse.Namespace(seed=None, stage='smd', only=None))
        for job in jobs:
            for key, value in benchmark['profile'].items():
                if key != 'identifier':
                    self.assertEqual(job['profile'][key], value)
        # Refuse an accidental return to the different author-weight protocol.
        spec['smd']['dim_feedforward'] = 76
        with self.assertRaises(ValueError):
            build_plan(spec, argparse.Namespace(seed=None, stage='smd', only=None))

    def test_all_four_architectures_with_history_controls(self):
        from tranad_plus.core.models import build_model
        from tranad_plus.core.training.objectives import reconstruction_loss
        windows = torch.rand(8, 5, 1)
        for variant, output_count in (('reference', 2), ('shared_decoder', 2), ('single_pass', 1), ('encoder_only', 1)):
            profile = {**tiny_profile(), 'variant': variant}
            model = build_model(profile)
            self.assertEqual(model.output_count, output_count)
            for condition in ('original', 'shuffle_1101', 'repeat_current'):
                x = transform(windows, condition).permute(1, 0, 2)
                y = x[-1:]
                output = model(x, y)
                values = output if isinstance(output, tuple) else (output,)
                self.assertEqual(len(values), output_count)
                self.assertTrue(all(v.shape == y.shape and torch.isfinite(v).all() for v in values))
                loss = reconstruction_loss(output, y, 2)
                model.zero_grad(set_to_none=True)
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_bf200_matches_archived_opssat_endpoint(self):
        parent = PROJECT/'results/v13/architectures/OPS-SAT/encoder_only/window_only/seed_2026'
        if not parent.exists():
            self.skipTest('Archived checkpoint results unavailable')
        report = json.loads((parent/'metrics.json').read_text())
        with np.load(parent/'scores.npz') as saved:
            labels, scores, ids = saved['test_labels'], saved['test_scores'][0], saved['test_segment_ids']
        cuts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1, len(ids)]
        groups = [{'values': np.empty(b-a)} for a, b in zip(cuts[:-1], cuts[1:])]
        result = bf200(scores, labels, groups)
        for metric in ('f1', 'precision', 'recall', 'ROC/AUC', 'threshold', 'TP', 'TN', 'FP', 'FN'):
            self.assertAlmostEqual(result['bf_'+metric], report['paper_oracle']['bf']['bf_'+metric], places=13)

    def test_oracle_has_only_bf_and_pot(self):
        rng = np.random.default_rng(12)
        train_scores = rng.exponential(.1, (1, 256))
        test_scores = rng.exponential(.1, (1, 80))
        labels = np.r_[np.zeros(40, dtype=np.int8), np.ones(40, dtype=np.int8)]
        with tempfile.TemporaryDirectory() as tmp:
            result = oracle(train_scores, test_scores, labels, [None], .01, None, Path(tmp), 'test')
        self.assertIn('bf', result)
        self.assertIn('pot', result)
        self.assertNotIn('bf_dense', result)
        self.assertEqual(result['auc_name'], 'AUC-PA')

    def test_exact_training_resume_with_shuffled_history(self):
        from tranad_plus.core.models import build_model
        profile = tiny_profile()
        bundle = changed_bundle(tiny_bundle(), profile['input_condition'])
        device = torch.device('cpu')
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'full', Path(tmp)/'resumed'
            train(profile, bundle, a, 'fixture', device)
            with self.assertRaises(TrainingInterrupted):
                train(profile, bundle, b, 'fixture', device, stop_after_epoch=2)
            # Simulate a torn epoch line after the durable second checkpoint.
            with (b/'epochs.jsonl').open('a') as stream:
                stream.write('{"epoch_one_based": 3}\n{"epoch_one_based":')
            reconcile_epochs(b)
            train(profile, bundle, b, 'fixture', device)
            full = torch.load(a/'checkpoints/final.ckpt', map_location='cpu', weights_only=True)
            resumed = torch.load(b/'checkpoints/final.ckpt', map_location='cpu', weights_only=True)
            for key in full['model_state']:
                torch.testing.assert_close(full['model_state'][key], resumed['model_state'][key], rtol=0, atol=0)
            self.assertEqual(full['scheduler_state'], resumed['scheduler_state'])
            self.assertEqual(full['final_loss'], resumed['final_loss'])
            self.assertEqual(len((b/'epochs.jsonl').read_text().splitlines()), 4)
            model = build_model(profile)
            model.load_state_dict(resumed['model_state'])
            settings = {'score_batch_size_by_dataset': {'OPS-SAT': 8}}
            result = trained_evaluation(model, tiny_bundle(), b, profile, device, settings, PROJECT/'data')
            self.assertEqual(set(result)-{'note'}, {'matched', 'ordered_test'})
            with np.load(b/'outputs_matched.npz') as arrays:
                np.testing.assert_array_equal(arrays['test_target'], tiny_bundle().datasets['test'].windows[:, -1].numpy())

    def test_completion_checks_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'artifact.npz').write_bytes(b'original')
            write_json(folder/'status.json', {'signature': 'a', 'status': 'complete',
                       'artifacts': {'artifact.npz': hash_file(folder/'artifact.npz')}})
            self.assertTrue(complete(folder, 'a'))
            with self.assertRaises(RuntimeError):
                complete(folder, 'different')
            (folder/'artifact.npz').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                complete(folder, 'a')

    def test_cli_training_evaluation_resume_and_verified_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data/raw/OPSSAT'
            data.mkdir(parents=True)
            rng = np.random.default_rng(81)
            with (data/'segments.csv').open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['channel', 'timestamp', 'value', 'label', 'anomaly', 'segment', 'train'])
                for seg in range(24):
                    is_train = int(seg < 16)
                    anomaly = int(12 <= seg < 16 or seg >= 20)
                    for t, value in enumerate(rng.normal(0, 1, 24)):
                        writer.writerow(['CADC0872', str(t), value, 'anomaly' if anomaly else 'nominal',
                                         anomaly, seg, is_train])
            spec = json.loads((PROJECT/'tranad_plus/campaigns/temporal/plan.json').read_text())
            spec['training_epochs'] = 4
            plan = root/'plan.json'
            write_json(plan, spec)
            argv = ['--only', 'train/OPS-SAT/encoder_only/window_only/shuffle_1101/seed_2026',
                    '--plan', str(plan), '--data-root', str(root/'data'), '--output-root', str(root/'out'),
                    '--device', 'cpu', '--cpu-threads', '1', '--no-progress']
            self.assertEqual(main(argv+['--stop-after-epoch', '2']), 75)
            self.assertEqual(main(argv), 0)
            report = json.loads((root/'out/campaign_latest.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['tables']['metric_rows'], 4)
            self.assertEqual(main(argv), 0)
            report = json.loads((root/'out/campaign_latest.json').read_text())
            self.assertEqual(report['runs'][0]['status'], 'skipped_verified_complete')


if __name__ == '__main__':
    unittest.main()
