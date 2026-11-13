"""Meaningful checks for held-out selection, trajectory parity and exact resume."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ['TOAST_NO_TQDM'] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'toast-matplotlib'))

import numpy as np
import torch

from tranad_plus.core.bootstrap import activate_baseline
activate_baseline()
from tranad_plus.core.data import DataBundle
from tranad_plus.core.training.engine import TrainingInterrupted
from .policy import EarlyStopping
from .training import train, evaluate_mse, nominal_loader


class Windows(torch.utils.data.Dataset):
    def __init__(self, values):
        self.windows = values
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, index):
        return self.windows[index]


def fixture():
    generator = torch.Generator().manual_seed(17)
    datasets = {s: Windows(torch.rand(n, 3, 1, generator=generator))
                for s, n in (('train', 24), ('validation', 16), ('test', 16))}
    labels = {'train': np.zeros(24, dtype=np.int8),
              'validation': np.repeat([0, 1], 8).astype(np.int8),
              'test': np.repeat([0, 1], 8).astype(np.int8)}
    groups = {split: [{'id': i, 'channel': 'a', 'label': int(label), 'values': np.array([0.])}
                      for i, label in enumerate(y)] for split, y in labels.items()}
    bundle = DataBundle(datasets, labels, {}, groups)
    profile = dict(dataset='OPS-SAT', features=1, variant='reference', configuration='reference',
                   window_sz=3, dim_feedforward=4, num_layers=1, batch_sz=8,
                   seed=2026, epochs=6, lr=.001, pot_q=.01, identifier='test')
    spec = {'evaluation': {'batch_size': 8}, 'fixed_epoch_checkpoint': 3,
            'early_stopping': {'patience': 2, 'min_epochs': 3, 'relative_min_delta': .5}}
    return bundle, profile, spec


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.arm = {'weight_decay': 0., 'dropout': .1}

    def test_policy_freezes_at_first_stop(self):
        policy = EarlyStopping(patience=2, min_epochs=3, relative_min_delta=.001)
        self.assertTrue(policy.step(1, 1.))
        self.assertFalse(policy.step(2, .9999))
        self.assertFalse(policy.step(3, 1.1))
        self.assertEqual(policy.stop_epoch, 3)
        self.assertFalse(policy.step(4, .01))
        self.assertEqual(policy.best_epoch, 1)
        with self.assertRaises(ValueError):
            policy.step(5, float('nan'))

    def test_campaign_plan_and_end_to_end_recovery(self):
        from .run import arguments, build_plan, main
        spec = json.loads(Path(__file__).with_name('plan.json').read_text())
        self.assertEqual(len(build_plan(spec, arguments([]))), 60)
        self.assertEqual(len(build_plan(spec, arguments(['--all-configs']))), 108)
        self.assertEqual(len(build_plan(spec, arguments(['--stage', 'compression']))), 144)
        self.assertEqual(len(build_plan(spec, arguments(['--stage', 'all']))), 156)
        self.assertEqual(len(build_plan(spec, arguments(['--stage', 'all', '--all-configs']))), 300)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data/raw/OPSSAT'
            data.mkdir(parents=True)
            import csv
            with (data/'segments.csv').open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['channel', 'segment', 'timestamp', 'value', 'train', 'anomaly', 'label'])
                for segment in range(16):
                    official_train = int(segment < 12)
                    anomaly = int(segment in (10, 11, 14, 15))
                    for t in range(8):
                        writer.writerow(['channel_a', segment, f'2026-01-01T00:00:{t:02}',
                                         .1+.02*t+.001*segment+.3*anomaly,
                                         official_train, anomaly, 'anomaly' if anomaly else 'normal'])
            spec.update(max_epochs=6, fixed_epoch_checkpoint=3)
            spec['early_stopping'].update(patience=2, min_epochs=3)
            plan = root/'plan.json'
            plan.write_text(json.dumps(spec))
            argv = ['--plan', str(plan), '--output-root', str(root/'output'), '--data-root', str(root/'data'),
                    '--device', 'cpu', '--cpu-threads', '1', '--no-progress', '--fail-fast',
                    '--only', 'OPS-SAT/encoder_only/compact/seed_2026/baseline']
            self.assertEqual(main(argv+['--stop-after-epoch', '2']), 75)
            self.assertEqual(main(argv), 0)
            self.assertEqual(main(argv), 0)  # Verified skip; no duplicate training.
            table = root/'output/tables/paired_checkpoint_differences.csv'
            self.assertTrue(table.is_file())
            self.assertEqual(len(list(csv.DictReader(table.open()))), 8)
            # Exercise integer reporting against a genuinely completed FP32 job,
            # including the combined size/accuracy table and immutable hashes.
            from .run import complete, summarize
            from .data import load_data
            from .integer import evaluate_integer
            job = build_plan(spec, arguments(argv))[0]
            folder = root/'output'/job['id']
            signature = json.loads((folder/'status.json').read_text())['signature']
            bundle = load_data(job['profile'], root/'data')
            with patch('tranad_plus.campaigns.temporal.evaluation._pot_threshold', return_value=(.02, None)):
                evaluate_integer(job, bundle, folder, signature, spec, 'cpu', backend='reference_cpu')
            self.assertTrue(complete(folder, signature))
            summarize(root/'output')
            with (root/'output/tables/compression_tradeoffs.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r['precision'] for r in rows}, {'fp32', 'int8', 'int4'})

    def test_native_split_scaler_and_official_test_parity(self):
        from .data import load_data
        from TranADPlus.src.datasets import TSDataset_tracks
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root/'processed/tranad_plus/SMAP'
            for split in ('Train', 'Test'):
                for kind in ('Tracks', 'Labels'):
                    (parent/split/kind).mkdir(parents=True)
            for i in range(2):
                train = np.repeat(np.arange(20, dtype=np.float32)[:, None], 25, axis=1)+i
                train[16:] = 1000+i  # Must never enter scaler fitting.
                test = np.repeat(np.arange(12, dtype=np.float32)[:, None], 25, axis=1)
                labels = np.zeros_like(test)
                labels[6:] = 1
                for split, a, y in (('Train', train, np.zeros_like(train)), ('Test', test, labels)):
                    np.save(parent/split/'Tracks'/f'{i}.npy', a)
                    np.save(parent/split/'Labels'/f'{i}.npy', y)
            profile = {'dataset': 'SMAP', 'features': 25, 'window_sz': 3}
            bundle = load_data(profile, root)
            self.assertEqual(bundle.inventory['split']['scaler_data_max'], [16.]*25)
            self.assertGreater(float(bundle.datasets['validation'].big_track.max()), 1.)
            for row in bundle.inventory['split']['ranges']:
                self.assertEqual(row['optimization_range'], [0, 16])
                self.assertEqual(row['validation_range'], [16, 20])
            self.assertFalse(bundle.labels['validation'].any())
            old_test = TSDataset_tracks(window_size=3, padding=True, scaling=True,
                                        track_paths=[str(p) for p in sorted((parent/'Test/Tracks').glob('*.npy'))],
                                        anom_paths=[str(p) for p in sorted((parent/'Test/Labels').glob('*.npy'))],
                                        scaler=bundle.datasets['train'].scaler)
            torch.testing.assert_close(old_test.big_track, bundle.datasets['test'].big_track, rtol=0, atol=0)
            np.testing.assert_array_equal(bundle.labels['test'], (old_test.big_anom[2:].sum(dim=1)>0).numpy())
            # Full native CLI path: nominal-only validation, short original
            # horizon, FP32 training/evaluation, missing supervised BF reported.
            from .run import main
            spec = json.loads(Path(__file__).with_name('plan.json').read_text())
            spec['author_anchors']['SMAP'].update(epochs=3, fixed_epoch_checkpoint=2)
            spec['evaluation']['batch_size_by_dataset']['SMAP'] = 16
            spec['early_stopping'].update(min_epochs=2, patience=1)
            plan = root/'plan.json'
            plan.write_text(json.dumps(spec))
            argv = ['--plan', str(plan), '--dataset', 'SMAP', '--stage', 'diagnostic',
                    '--only', 'SMAP/reference/benchmark_smap_w10/seed_2026/baseline',
                    '--output-root', str(root/'output'), '--data-root', str(root),
                    '--device', 'cpu', '--cpu-threads', '1', '--no-progress', '--fail-fast']
            # data root cannot contain the output tree (protected by design),
            # so use a sibling destination for this synthetic fixture.
            with tempfile.TemporaryDirectory() as destination:
                argv[argv.index('--output-root')+1] = str(Path(destination)/'output')
                self.assertEqual(main(argv), 0)
                self.assertEqual(main(argv), 0)

    def test_native_nominal_validation_cannot_select_bf(self):
        from .evaluation import evaluate
        from .training import configured_model
        bundle, profile, spec = fixture()
        bundle.groups = {}
        bundle.labels['validation'][:] = 0
        profile['dataset'] = 'MSL'
        with tempfile.TemporaryDirectory() as tmp, patch('opssat_pipeline._pot_threshold', return_value=(.02, None)), patch(
                'tranad_plus.campaigns.temporal.evaluation.bounded_pot',
                return_value={'status': 'not_available', 'pot_f1': None, 'reason': 'test-only unavailable fit'}):
            result = evaluate(configured_model(profile, self.arm), bundle, profile, tmp, 'cpu', spec)
        self.assertEqual(result['test_validation_calibrated']['bf']['status'], 'not_available')
        self.assertEqual(result['test_validation_calibrated']['pot']['alpha'], .5)
        self.assertFalse(result['test_validation_calibrated']['pot']['selection_uses_test_labels'])

    def test_integer_reports_resume_without_duplicate_weights(self):
        from .integer import evaluate_integer
        bundle, profile, spec = fixture()
        spec['integer'] = {'bits': [8, 4], 'checkpoints': ['early_stopped', 'final']}
        job = {'id': 'test', 'profile': profile, 'arm': self.arm}
        with tempfile.TemporaryDirectory() as tmp, patch(
                'tranad_plus.campaigns.temporal.evaluation._pot_threshold', return_value=(.02, None)):
            folder = Path(tmp)
            train(profile, self.arm, bundle, folder, 'sig', spec, 'cpu')
            original_weights = sorted(str(p.relative_to(folder)) for p in folder.rglob('*.ckpt'))
            evaluate_integer(job, bundle, folder, 'sig', spec, 'cpu', backend='reference_cpu')
            self.assertEqual(len(list((folder/'integer').glob('*/*/status.json'))), 4)
            self.assertEqual(original_weights, sorted(str(p.relative_to(folder)) for p in folder.rglob('*.ckpt')))
            self.assertFalse(list((folder/'integer').rglob('*.npz')))
            with patch('tranad_plus.campaigns.regularization.integer.configured_model', side_effect=AssertionError('must skip')):
                evaluate_integer(job, bundle, folder, 'sig', spec, 'cpu', backend='reference_cpu')

    def test_validation_loader_excludes_anomalies_and_rng_is_preserved(self):
        bundle, profile, spec = fixture()
        from .training import configured_model
        model = configured_model(profile, self.arm)
        loader, transfer = nominal_loader(bundle, 'validation', 8, 'cpu')
        before = torch.get_rng_state().clone()
        result = evaluate_mse(model, loader, transfer, 'cpu')
        self.assertEqual(result['target_values'], 8)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_exact_resume_and_no_test_dependency(self):
        bundle, profile, spec = fixture()
        spec['opssat_detection'] = {'fixed_alpha': .5, 'every_epochs': 1}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = train(profile, self.arm, bundle, root/'full', 'sig', spec, 'cpu')
            altered = copy.deepcopy(bundle)
            altered.datasets['test'].windows[:] = 1000
            altered.labels['test'][:] = 0
            with self.assertRaises(TrainingInterrupted):
                train(profile, self.arm, altered, root/'resume', 'sig', spec, 'cpu', stop_after_epoch=2)
            resumed = train(profile, self.arm, altered, root/'resume', 'sig', spec, 'cpu')
            self.assertEqual(full['policy'], resumed['policy'])
            self.assertEqual(full['snapshots'], resumed['snapshots'])
            for name in full['snapshots']:
                a = torch.load(root/'full/checkpoints'/f'{name}.ckpt', weights_only=True)
                b = torch.load(root/'resume/checkpoints'/f'{name}.ckpt', weights_only=True)
                for key in a['model_state']:
                    torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
            rows = [json.loads(s) for s in (root/'resume/epochs.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in rows], list(range(1, 7)))

    def test_monitoring_does_not_change_original_training_trajectory(self):
        bundle, profile, spec = fixture()
        spec['opssat_detection'] = {'fixed_alpha': .5, 'every_epochs': 1}
        # The existing engine uses final loss and optimizer exactly as baseline.
        from tranad_plus.core.training.engine import train as original
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original(profile, bundle, root/'original', 'original', 'cpu')
            train(profile, self.arm, bundle, root/'monitored', 'monitored', spec, 'cpu')
            a = torch.load(root/'original/checkpoints/final.ckpt', weights_only=True)
            b = torch.load(root/'monitored/checkpoints/final.ckpt', weights_only=True)
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)

    def test_actual_stop_matches_audit_selected_checkpoint(self):
        bundle, profile, spec = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = train(profile, self.arm, bundle, root/'audit', 'sig', spec, 'cpu')
            stopped = train(profile, self.arm, bundle, root/'stop', 'sig', spec, 'cpu', stop_on_patience=True)
            self.assertEqual(audit['policy'], stopped['policy'])
            self.assertEqual(stopped['snapshots']['final'], stopped['policy']['stop_epoch'])
            self.assertEqual(audit['snapshots']['final'], 6)
            a = torch.load(root/'audit/checkpoints/early_stopped.ckpt', weights_only=True)
            b = torch.load(root/'stop/checkpoints/early_stopped.ckpt', weights_only=True)
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)

    def test_validation_calibration_is_independent_of_test_labels(self):
        from .evaluation import evaluate
        from .training import configured_model
        bundle, profile, spec = fixture()
        other = copy.deepcopy(bundle)
        other.labels['test'] = 1-bundle.labels['test']
        other.groups['test'] = [{**g, 'label': 1-g['label']} for g in bundle.groups['test']]
        model = configured_model(profile, self.arm)
        # POT fitting is separately audited in the shared pipeline. A fixed
        # successful fit isolates which labels this campaign uses for selection.
        with tempfile.TemporaryDirectory() as tmp, patch(
                'tranad_plus.campaigns.temporal.evaluation._pot_threshold', return_value=(.02, None)):
            a = evaluate(model, bundle, profile, Path(tmp)/'a', 'cpu', spec)
            b = evaluate(model, other, profile, Path(tmp)/'b', 'cpu', spec)
        for method in ('bf', 'pot'):
            self.assertFalse(a['test_validation_calibrated'][method]['selection_uses_test_labels'])
            self.assertEqual(a['test_validation_calibrated'][method]['alpha'], b['test_validation_calibrated'][method]['alpha'])
            self.assertEqual(a['test_validation_calibrated'][method]['threshold'], b['test_validation_calibrated'][method]['threshold'])


if __name__ == '__main__':
    unittest.main()
