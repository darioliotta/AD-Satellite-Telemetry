"""Small, exact tests for the legacy-score window study and output integrity."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tranad_variants.common import hash_file
from tranad_v14.metrics import ranking
from tranad_v14.windows import run_window_study, interior_mask
from tranad_v14.run import main


class WindowTests(unittest.TestCase):
    def fixture(self, root):
        data, results = root / 'data', root / 'results'
        lengths = [12, 15]
        labels = [np.r_[np.zeros(3), np.ones(5), np.zeros(4)].astype(np.int8),
                  np.r_[np.zeros(4), np.ones(6), np.zeros(5)].astype(np.int8)]
        tracks = data / 'processed/tranad_plus/SMAP/Test/Tracks'
        tracks.mkdir(parents=True)
        for i, n in enumerate(lengths):
            np.save(tracks / f'{i}.npy', np.zeros((n, 25), dtype=np.float32))
        parent = results / 'v10/study/SMAP'
        parent.mkdir(parents=True)
        csv_rows = []
        for w in (3, 5):
            identifier = f'fixture_w{w}'
            y = np.concatenate([np.r_[np.repeat(a[0], w-1), a] for a in labels])[w-1:]
            s = np.linspace(.1, .2, len(y)).astype(np.float32) + .3*y
            tr = np.linspace(.01, .08, 40).astype(np.float32)
            score_path = parent / (identifier + '.npz')
            np.savez_compressed(score_path, scores=s, labels=y, train_scores=tr)
            checkpoint = parent / (identifier + '.ckpt')
            checkpoint.write_bytes(b'fixture checkpoint')
            profile = dict(dataset='SMAP', identifier=identifier, window_sz=w, dim_feedforward=50, seed=2026, epochs=50)
            report = dict(dataset='SMAP', identifier=identifier, profile=profile,
                          source={'models_sha256': 'fixture_model'},
                          checkpoint={'file': checkpoint.name, 'sha256': hash_file(checkpoint)},
                          raw_pointwise_evaluation={'alpha_fixed_pre_test': .5, 'scores_file': score_path.name,
                                                    'scores_sha256': hash_file(score_path), 'roc_auc_continuous_scores': ranking(y, s)['auc']},
                          original_paper_evaluation={'metrics': {'pot': {'pot_f1': .8},
                                                               'bf': {'bf_f1': .8, 'bf_dense_f1': .9}}})
            path = parent / (identifier + '.json')
            path.write_text(json.dumps(report))
            csv_rows.append({'dataset': 'SMAP', 'identifier': identifier, 'seed': 2026,
                             'checkpoint_sha256': hash_file(checkpoint), 'report_file': str(path.relative_to(parent.parent))})
        with (parent.parent / 'all_experiments.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        settings = dict(versions=['v10'], datasets=['SMAP'], seeds=[2026], windows_per_dataset=2,
                        ff_widths_per_dataset=1, diagnostic_test_normal_fpr=.005, baseline_ff={'SMAP': 50})
        return data, results, settings

    def test_real_and_interior_population_and_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'TOAST_NO_TQDM': '1'}):
            root = Path(directory)
            data, results, settings = self.fixture(root)
            source = results / 'v10/study/SMAP/fixture_w3.npz'
            before = hash_file(source)
            summary = run_window_study(results, data, root/'out', settings, 'fixture_model')
            self.assertEqual(summary, {'runs': 2, 'cells': 2, 'paired_comparisons': 1})
            with (root/'out/runs.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({int(r['points_real']) for r in rows}, {27})
            self.assertEqual({int(r['points_interior']) for r in rows}, {19})
            self.assertEqual(hash_file(source), before)
            self.assertTrue((root/'out/window_effects.pdf').exists())
            self.assertTrue((root/'out/paired_window_effects.csv').exists())
            for w in (3, 5):
                m = interior_mask({'dataset': 'SMAP', 'window_sz': w}, data, 5, 27)
                self.assertEqual(int(m.sum()), 19)

    def test_corrupt_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'TOAST_NO_TQDM': '1'}):
            root = Path(directory)
            data, results, settings = self.fixture(root)
            source = results / 'v10/study/SMAP/fixture_w3.npz'
            with source.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'missing/changed'):
                run_window_study(results, data, root/'out', settings, 'fixture_model')


if __name__ == '__main__':
    unittest.main()
