"""Invariants for matched examples and exact counterfactual bookkeeping."""
import unittest
import numpy as np

from opssat_window_cases import shapley, consensus, cube_decisions, confusion, peak_alignment, select_cases


class WindowCaseTests(unittest.TestCase):
    def test_additive_allocation_and_total_difference(self):
        terms = [np.array([1., -2.]), np.array([3., 0.]), np.array([-1., 4.])]
        cube = {mask: sum((terms[i] if mask & (1 << i) else np.zeros(2)) for i in range(3)) for mask in range(8)}
        result = shapley(cube)
        for key, expected in zip(('model_window', 'alpha', 'threshold'), terms):
            np.testing.assert_allclose(result[key], expected)

    def test_reversing_both_endpoints_reverses_all_contributions(self):
        cube = {mask: np.array([float(mask**2), float(mask % 3)]) for mask in range(8)}
        forward, reverse = shapley(cube), shapley({m:cube[7-m] for m in range(8)})
        for key in forward:
            np.testing.assert_allclose(forward[key], -reverse[key], atol=1e-12)

    def test_identical_scores_make_a_threshold_only_alarm_change(self):
        def endpoint(threshold, f1, precision, recall, auc):
            return {'report': {'paper_oracle': {'bf': {'alpha': None, 'bf_threshold': threshold,
                    'bf_f1': f1, 'bf_precision': precision, 'bf_recall': recall, 'bf_ROC/AUC': auc}}},
                    'scores': np.array([[.4, .1]]), 'alphas': np.array([]),
                    'starts': np.array([0, 1]), 'fragment_labels': np.array([1, 0]), 'lengths': np.array([1, 1])}
        hits, _, _ = cube_decisions(endpoint(.5,0.,0.,0.,.5), endpoint(.3,1.,1.,1.,1.), 'bf')
        effects=shapley({k:v.astype(float) for k,v in hits.items()})
        np.testing.assert_allclose(effects['model_window'], [0.,0.])
        np.testing.assert_allclose(effects['alpha'], [0.,0.])
        np.testing.assert_allclose(effects['threshold'], [1.,0.])

    def test_pa_weights_positive_fragments_by_length_but_keeps_nominal_point_alarms(self):
        result=confusion(np.array([True,True]),np.array([1,1]),np.array([1,0]),np.array([2,3]))
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['tn']), (2,1,0,2))
        self.assertAlmostEqual(result['auc_pa'],5/6)

    def test_stability_requires_four_agreeing_seeds(self):
        def row(cat):
            return {'decision_category':cat,'short_hit':int(cat in ('short_only','both')),
                    'long_hit':int(cat in ('long_only','both')),
                    **{k+'_contribution':0. for k in ('model_window','alpha','threshold')}}
        self.assertEqual(consensus([row('short_only')]*3+[row('long_only')]*2)['category'],'seed_dependent')
        self.assertEqual(consensus([row('short_only')]*4+[row('long_only')])['category'],'short_only')

    def test_peak_alignment_ignores_padding_region_and_reports_all_radii(self):
        x=np.zeros(20);x[10:]=10
        short=np.zeros(20);long=np.zeros(20)
        short[0]=100;short[12]=2;long[10]=3
        r=peak_alignment({'x':x},short,long)
        self.assertEqual(r['short_peak_index'],12)
        self.assertEqual(r['short_peak_near_step_r0'],0)
        self.assertEqual(r['short_peak_near_step_r2'],1)
        self.assertEqual(r['long_peak_near_step_r0'],1)
        self.assertEqual(peak_alignment({'x':x},np.ones(20),long)['peak_alignment_available'],0)

    def test_examples_use_median_duration_not_most_extreme_segment(self):
        fragments={i:{'official_train':0,'channel':'CADC0872','label':int(i<=5),'duration_s':{1:10,2:20,3:1000}.get(i,50)} for i in range(1,11)}
        rows=[dict(variant='reference',epochs=50,method='bf',label=1,category='short_only',channel='CADC0872',segment=i) for i in (1,2,3)]
        decisions=[dict(variant='reference',epochs=50,method='bf',decision_category='short_only',segment=i,seed=s) for i in (1,2,3) for s in range(2026,2031)]
        selected=select_cases(rows,decisions,fragments)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['segment'],2)
        self.assertEqual(selected[0]['representative_seed'],2028)


if __name__=='__main__':unittest.main()
