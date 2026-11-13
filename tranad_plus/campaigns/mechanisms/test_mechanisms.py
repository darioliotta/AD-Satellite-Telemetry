"""Target-access isolation, split/label integrity, calibration and durable recovery."""
import copy
import csv
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
from sklearn.preprocessing import MinMaxScaler
from tranad_plus.core.bootstrap import activate_baseline
activate_baseline()
from tranad_plus.core.data import DataBundle, inventory, load_data
from tranad_plus.core.training.engine import TrainingInterrupted
from .control import build_control
from .data import IndexedWindows, Layout, control_data, historical_layout, batches
from .engine import train, stream_scores
from .metrics import events, endpoint, lifted_scores, bf200_pa, evaluate_scores
from .run import arguments, plan, main


def profile(variant='reference', condition='observed_current'):
    return dict(dataset='OPS-SAT', features=1, variant=variant, condition=condition,
                configuration='reference', window_sz=3, dim_feedforward=4, num_layers=1,
                batch_sz=8, epochs=4, lr=.001, pot_q=.01, seed=2026, identifier='synthetic')


def settings():
    return {'fixed_alpha': .5, 'nominal_calibration_quantile': .99, 'segment_quantile': .95,
            'bf_threshold_steps': 200, 'histogram_bins': 10,
            'batch_size_by_dataset': {'OPS-SAT': 8, 'SMAP': 8, 'MSL': 8}}


def fixture():
    tr = [{'id': i, 'channel': 'a', 'values': np.linspace(.1+i*.02, .4+i*.02, 12), 'label': 0} for i in range(2)]
    va = [{'id': 2+i, 'channel': 'a', 'values': np.linspace(.15, .4, 12)+i*.3, 'label': i} for i in range(2)]
    te = [{'id': 4+i, 'channel': 'a', 'values': np.linspace(.13, .38, 12)+i*.4, 'label': i} for i in range(2)]
    scaler = {'a': MinMaxScaler().fit(np.concatenate([g['values'] for g in tr])[:, None])}
    groups = {'train': tr, 'validation': va, 'test': te}
    datasets = {s: IndexedWindows(g, scaler, 3) for s, g in groups.items()}
    layouts = {s: Layout(d.point_labels, d.unit_ids, np.ones(len(d), dtype=bool)) for s, d in datasets.items()}
    return DataBundle(datasets, {s: x.labels for s, x in layouts.items()}, {}, groups), layouts


def write_opssat(root):
    p = Path(root)/'raw/OPSSAT/segments.csv'; p.parent.mkdir(parents=True)
    with p.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['channel', 'segment', 'timestamp', 'value', 'train', 'anomaly', 'label'])
        for i in range(16):
            label = int(i in (10, 11, 14, 15))
            for t in range(14):
                writer.writerow(['a', i, f'2026-01-01T00:00:{t:02}', .1+.01*t+.002*i+.3*label,
                                 int(i < 12), label, 'anomaly' if label else 'normal'])


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_counts_and_filters(self):
        spec = json.loads(Path(__file__).with_name('plan.json').read_text())
        jobs = plan(spec, arguments([]))
        self.assertEqual(len(jobs), 510)
        self.assertEqual(sum(j['stage'] == 'train' for j in jobs), 72)
        self.assertEqual(sum(j['stage'] == 'analysis' for j in jobs), 432)
        self.assertEqual(sum(j['stage'] == 'baselines' for j in jobs), 6)
        self.assertEqual(len(set(j['id'] for j in jobs)), len(jobs))
        with self.assertRaises(ValueError):
            plan(spec, arguments(['--seed', '2030']))

    def test_hidden_current_cannot_enter_any_architecture_or_query(self):
        torch.manual_seed(123)
        x = torch.rand(3, 4, 1)
        changed = x.clone(); changed[-1] = 100.
        for variant in ('reference', 'shared_decoder', 'single_pass', 'encoder_only'):
            model = build_control(profile(variant, 'hidden_current')).eval()
            a, b = model(x, x[-1:]), model(changed, changed[-1:])
            a = a if isinstance(a, tuple) else (a,)
            b = b if isinstance(b, tuple) else (b,)
            for first, second in zip(a, b):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            v = x.clone().requires_grad_()
            result = model(v, v[-1:])
            result = result if isinstance(result, tuple) else (result,)
            sum(o.sum() for o in result).backward()
            self.assertEqual(float(v.grad[-1].abs().sum()), 0.)

    def test_complete_windows_never_cross_units_or_pad_targets(self):
        groups = [{'id': 0, 'channel': 'a', 'values': np.array([1., 2., 3., 4.]), 'label': 0},
                  {'id': 1, 'channel': 'a', 'values': np.array([100., 101., 102., 103.]), 'label': 1},
                  {'id': 2, 'channel': 'a', 'values': np.array([5.]), 'label': 0}]
        scaler = {'a': MinMaxScaler().fit(np.array([[0.], [1.]]))}
        data = IndexedWindows(groups, scaler, 3)
        self.assertEqual(len(data), 4)
        self.assertEqual(data.unit_ids.tolist(), [0, 0, 1, 1])
        np.testing.assert_array_equal(data[2][:, 0], [100., 101., 102.])
        self.assertEqual(data.exclusions[0]['id'], 2)
        source, _ = batches(data, 3, 'cpu')
        torch.testing.assert_close(torch.cat(list(source)), torch.stack([data[i] for i in range(len(data))]))

    def test_native_holdout_and_historical_real_target_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = root/'processed/tranad_plus/SMAP'
            for split in ('Train', 'Test'):
                for kind in ('Tracks', 'Labels'):
                    (parent/split/kind).mkdir(parents=True)
            for i in range(2):
                values = np.repeat(np.arange(20, dtype=np.float32)[:, None], 25, axis=1)+i
                values[16:] = 1000
                test = np.repeat(np.arange(8, dtype=np.float32)[:, None], 25, axis=1)
                labels = np.zeros_like(test); labels[4:] = 1
                for split, a, y in (('Train', values, np.zeros_like(values)), ('Test', test, labels)):
                    np.save(parent/split/'Tracks'/f'{i}.npy', a)
                    np.save(parent/split/'Labels'/f'{i}.npy', y)
            p = dict(dataset='SMAP', features=25, window_sz=3)
            files = inventory(p, root)
            bundle, layouts = control_data(p, root, files)
            self.assertEqual(bundle.inventory['split']['scaler_max'], [16.]*25)
            self.assertGreater(float(bundle.datasets['validation'].series.max()), 1.)
            self.assertEqual(len(layouts['test'].labels), 12)
            old = load_data(p, root, files)
            original = historical_layout(old, p, root)['test']
            self.assertEqual(int(original.real.sum()), 16)
            np.testing.assert_array_equal(original.labels[original.real], np.tile(np.repeat([0, 1], 4), 2))

    def test_PA_is_bounded_by_units_and_bf200_matches_brute_force(self):
        y = np.array([0, 1, 1, 1, 1, 0]); ids = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([.1, .9, .2, .2, .3, .4])
        self.assertEqual(events(y, ids), [(1, 3), (3, 5)])
        e = endpoint(scores, y, ids, .5)
        self.assertEqual(e['raw']['TP'], 1); self.assertEqual(e['PA']['TP'], 2)
        self.assertEqual(e['event_coverage'], .5)
        best = bf200_pa(scores, y, ids)
        candidates = np.linspace(scores.min(), scores.mean()+3*scores.std(), 200)
        exhaustive = [endpoint(scores, y, ids, t)['PA']['f1'] for t in candidates]
        self.assertEqual(best['threshold_index'], int(np.argmax(exhaustive)))

    def test_calibration_does_not_use_test_labels(self):
        bundle, layouts = fixture()
        scores = {s: np.linspace(.01, .9, len(d)) for s, d in bundle.datasets.items()}
        changed = copy.deepcopy(layouts); changed['test'].labels = 1-changed['test'].labels
        with patch('opssat_pipeline._pot_threshold', return_value=(.5, None)):
            a, _ = evaluate_scores(scores, layouts, 'OPS-SAT', settings(), 'heldout_nominal_validation')
            b, _ = evaluate_scores(scores, changed, 'OPS-SAT', settings(), 'heldout_nominal_validation')
        self.assertEqual(a['calibration'], b['calibration'])
        self.assertEqual(a['fragment_classification']['threshold'], b['fragment_classification']['threshold'])
        self.assertFalse(a['calibration']['selection_uses_test_labels'])

    def test_simple_baselines_use_only_past_for_prediction(self):
        bundle, _ = fixture(); ds = bundle.datasets['test']
        windows = torch.stack([ds[i] for i in range(len(ds))]).numpy()
        for name in ('persistence', 'past_median'):
            scores, _ = stream_scores(None, ds, 'cpu', 8, None, 'test', baseline=name)
            predicted = windows[:, -2] if name == 'persistence' else np.median(windows[:, :-1], axis=1)
            expected = np.mean((predicted-windows[:, -1])**2, axis=1)
            np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-8)

    def test_integer_streaming_reference_all_architectures(self):
        from tranad_plus.core.models import build_model
        from tranad_plus.campaigns.integer.runtime import convert
        bundle, layouts = fixture()
        for variant in ('reference', 'shared_decoder', 'single_pass', 'encoder_only'):
            for bits in (8, 4):
                model = convert(build_model(profile(variant)), bits, 'reference_cpu')
                alpha = .5 if model.output_count == 2 else None
                scores, measured = stream_scores(model, bundle.datasets['test'], 'cpu', 8, alpha,
                                                 'integer test', labels=layouts['test'].labels)
                self.assertTrue(np.isfinite(scores).all())
                self.assertEqual(len(scores), len(layouts['test'].labels))
                self.assertEqual(measured['class_errors']['1']['values'], 10)

    def test_native_MSL_cli_observed_and_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root/'data/processed/tranad_plus/MSL'
            for split, n in (('Train', 300), ('Test', 70)):
                for kind in ('Tracks', 'Labels'):
                    (data/split/kind).mkdir(parents=True)
                values = np.repeat(np.linspace(.1, .8, n, dtype=np.float32)[:, None], 55, axis=1)
                labels = np.zeros_like(values)
                if split == 'Test':
                    values[50:65] += .3; labels[50:65] = 1
                np.save(data/split/'Tracks/a.npy', values); np.save(data/split/'Labels/a.npy', labels)
            spec = json.loads(Path(__file__).with_name('plan.json').read_text()); spec['epochs'] = 2
            file = root/'plan.json'; file.write_text(json.dumps(spec))
            argv = ['--plan', str(file), '--stage', 'train', '--dataset', 'MSL', '--variant', 'encoder_only', '--seed', '2026',
                    '--device', 'cpu', '--cpu-threads', '1', '--data-root', str(root/'data'), '--output-root', str(root/'out'),
                    '--no-progress', '--fail-fast']
            with patch('opssat_pipeline._pot_threshold', return_value=(.02, None)):
                self.assertEqual(main(argv), 0)
                self.assertEqual(main(argv), 0)
            self.assertEqual(len(list((root/'out').glob('train/MSL/*/seed_*/*/checkpoints/final.ckpt'))), 2)

    def test_exact_resume_and_no_test_influence(self):
        bundle, _ = fixture(); altered = copy.deepcopy(bundle)
        altered.datasets['test'].series[:] = 1000.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = profile(condition='hidden_current')
            a = train(p, bundle, root/'a', 'sig', 'cpu', settings())
            with self.assertRaises(TrainingInterrupted):
                train(p, altered, root/'b', 'sig', 'cpu', settings(), stop_after_epoch=2)
            b = train(p, altered, root/'b', 'sig', 'cpu', settings())
            first = torch.load(a, weights_only=True); second = torch.load(b, weights_only=True)
            for key in first['model_state']:
                torch.testing.assert_close(first['model_state'][key], second['model_state'][key], rtol=0, atol=0)
            rows = [json.loads(s) for s in (root/'b/epochs.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in rows], [1, 2, 3, 4])

    def test_cli_recovery_evaluation_tables_and_verified_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); write_opssat(root/'data')
            spec = json.loads(Path(__file__).with_name('plan.json').read_text()); spec['epochs'] = 2
            spec['evaluation']['batch_size_by_dataset']['OPS-SAT'] = 16
            file = root/'plan.json'; file.write_text(json.dumps(spec))
            argv = ['--plan', str(file), '--stage', 'train', '--dataset', 'OPS-SAT', '--variant', 'encoder_only',
                    '--seed', '2026', '--device', 'cpu', '--cpu-threads', '1', '--no-progress', '--fail-fast',
                    '--data-root', str(root/'data'), '--output-root', str(root/'out')]
            with patch('opssat_pipeline._pot_threshold', return_value=(.02, None)):
                self.assertEqual(main(argv+['--stop-after-epoch', '1']), 75)
                self.assertEqual(main(argv), 0)
                self.assertEqual(main(argv), 0)
                baseline = [a for a in argv]
                baseline[baseline.index('--stage')+1] = 'baselines'
                self.assertEqual(main(baseline), 0)
            with (root/'out/tables/per_case.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r['condition'] for r in rows if r['stage'] == 'train'}, {'observed_current', 'hidden_current'})
            self.assertTrue((root/'out/tables/paired_comparisons.csv').exists())
            self.assertTrue((root/'out/tables/simple_detector_comparisons.csv').exists())
            self.assertFalse(list((root/'out').rglob('*scores.npy')))


if __name__ == '__main__':
    unittest.main()
