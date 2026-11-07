"""Fast CPU-only unit tests, safe without datasets or GPU."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from campaign_plan import read_plan, register_plan_profiles
from pipeline_metrics import _raw_quality, measurement_key


class IntegratedCampaignTests(unittest.TestCase):
    def test_plan_subset_overrides_and_order(self):
        catalog = {'SMAP': {'smoke_original': {
            'epochs': 50, 'batch_sz': 32, 'window_sz': 10, 'lr': 0.001,
            'dim_feedforward': '2x', 'pot_q': 0.07,
        }}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'plan.json'
            path.write_text(json.dumps({'experiments': [
                {'base_profile': 'smoke_original', 'identifier': 'smoke_short', 'epochs': 1, 'seed': 123}
            ]}))
            order = register_plan_profiles(catalog, path)
            self.assertEqual(order, ['smoke_short'])
            self.assertEqual(catalog['SMAP']['smoke_short']['epochs'], 1)
            self.assertEqual(catalog['SMAP']['smoke_original']['epochs'], 50)
            self.assertEqual(catalog['SMAP']['smoke_short']['seed'], 123)

    def test_duplicate_plan_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'plan.json'
            path.write_text(json.dumps({'experiments':['same','same']}))
            with self.assertRaises(ValueError):
                register_plan_profiles({'SMAP': {'same': {'epochs': 2}}}, path)

    def test_raw_auc_not_binary_auc(self):
        result = _raw_quality(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0,0,1,1]), 0.5)
        self.assertEqual(result['f1'], 1)
        self.assertEqual(result['roc_auc_continuous_scores'], 1)
        self.assertEqual(result['point_adjustment'], False)

    def test_single_class_auc_undefined(self):
        result = _raw_quality(np.array([0.1,0.2]), np.array([0,0]), 0.15)
        self.assertIsNone(result['roc_auc_continuous_scores'])

    def test_measurement_key_is_stable(self):
        a, _ = measurement_key({'warmup': 2}, 'cpu')
        b, _ = measurement_key({'warmup': 2}, 'cpu')
        self.assertEqual(a, b)
        self.assertNotEqual(a, measurement_key({'warmup': 3}, 'cpu')[0])


if __name__ == '__main__':
    unittest.main()
