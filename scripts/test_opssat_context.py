"""Small scientific-invariant tests for the offline OPS-SAT analysis."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from opssat_context import fragment_counts, load_fragments, metrics, paired_windows, select_scores


class ContextAnalysisTests(unittest.TestCase):
    def test_adjacent_anomalous_fragments_are_not_merged_by_pa(self):
        rows = fragment_counts(np.array([0., 1., 0., 0., 1., 0.]),
                               np.array([1, 1, 1, 1, 0, 0]),
                               np.array([10, 10, 11, 11, 12, 12]), .5)
        result = metrics(rows)
        self.assertEqual((result['tp'], result['fn'], result['fp'], result['tn']), (2, 2, 1, 1))
        self.assertEqual(result['anomalous_fragment_recall'], .5)
        self.assertEqual(result['auc_pa'], .5)

    def test_binary_auc_and_precision_use_different_denominators(self):
        rows = [{'tp': 80, 'fn': 40, 'fp': 0, 'tn': 0, 'label': 1, 'points': 120, 'hit': 1},
                {'tp': 0, 'fn': 0, 'fp': 20, 'tn': 980, 'label': 0, 'points': 1000, 'hit': 1}]
        result = metrics(rows)
        self.assertAlmostEqual(result['precision_pa'], .8)
        self.assertAlmostEqual(result['recall_pa'], 2/3)
        self.assertAlmostEqual(result['auc_pa'], ((2/3)+.98)/2)

    def test_single_class_auc_is_unavailable(self):
        rows = fragment_counts(np.array([1., 0.]), np.array([0, 0]), np.array([1, 1]), .5)
        self.assertIsNone(metrics(rows)['auc_pa'])
        self.assertIsNone(metrics(rows)['f1_pa'])

    def test_saved_alpha_is_used_without_new_selection(self):
        arrays = {'alpha_values': np.array([0., .5, 1.]), 'test_scores': np.array([[1., 2.], [3., 4.], [5., 6.]])}
        np.testing.assert_array_equal(select_scores(arrays, {'alpha_applicable': True, 'alpha': .5}), [3., 4.])
        with self.assertRaises(ValueError):
            select_scores(arrays, {'alpha_applicable': True, 'alpha': .25})

    def test_repeated_timestamps_are_retained_and_history_uses_actual_time(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'segments.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['segment','channel','timestamp','value','anomaly','train','sampling'])
                writer.writeheader()
                for segment in [1, 2]:
                    for sec, value in zip([0, 0, 1, 2, 5], [0., .1, .2, .3, .4]):
                        writer.writerow(dict(segment=segment, channel='CADC0872', timestamp=f'2022-01-01T00:00:{sec:02d}Z',
                                             value=value, anomaly=int(segment==2), train=int(segment==1), sampling=1))
            fragments, cutoffs = load_fragments(path, {1})
            self.assertEqual(fragments[1]['n_samples'], 5)
            self.assertEqual(fragments[1]['history_span_w5_median_s'], 5.)
            self.assertEqual(fragments[1]['repeated_timestamp_fraction'], .25)
            self.assertEqual(cutoffs['CADC0872']['n_nominal_optimization_fragments'], 1)

    def test_disjoint_score_blocks_fail_instead_of_silently_realigning(self):
        with self.assertRaises(ValueError):
            fragment_counts(np.ones(3), np.zeros(3), np.array([1, 2, 1]), .5)


if __name__ == '__main__':
    unittest.main()
