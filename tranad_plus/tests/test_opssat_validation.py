"""OPS-SAT validation and calibration tests (no model training)."""
import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from opssat_pipeline import prepare_data
from opssat_scoring import (fit_calibrator,transform_scores,mix_scores,selected_scoring,
                            evaluate_selected,segment_statistics,auc_ap)


def group(segment, channel, label, n=2):
    return {'id':segment, 'channel':channel, 'label':label,
            'values':np.zeros(n,dtype=np.float32)}

class ValidationScoringTests(unittest.TestCase):
    def test_fit_is_train_only_and_channel_conditional(self):
        scores=np.array([1.,2.,3.,100.,110.,120.]); channels=np.array(['a']*3+['b']*3)
        cal=fit_calibrator(scores,channels,'robust_channel')
        transformed=transform_scores(scores,channels,cal)
        self.assertAlmostEqual(float(np.median(transformed[:3])),0.)
        self.assertAlmostEqual(float(np.median(transformed[3:])),0.)
        self.assertEqual(cal['fit_source'],'normal_training_scores_only')
        with self.assertRaises(ValueError):transform_scores([9],['unseen'],cal)

    def test_decoder_mix(self):
        d1=np.array([1.,2.]);d2=np.array([3.,4.]);y=np.array([0.,1.])
        np.testing.assert_allclose(mix_scores(d1,d2,y,0.),[9.,9.])
        np.testing.assert_allclose(mix_scores(d1,d2,y,1.),[1.,1.])

    def test_fragment_metric_alignment(self):
        gs=[group(1,'a',0,3),group(2,'b',1,2)]
        values,labels,ids,channels=segment_statistics([1,2,3,4,5],gs,'max')
        self.assertEqual(values.tolist(),[3,5]);self.assertEqual(labels.tolist(),[0,1]);self.assertEqual(ids.tolist(),[1,2])
        with self.assertRaises(ValueError):segment_statistics([1,2],gs,'max')

    def test_validation_selection_never_uses_test_labels(self):
        tr=[group(1,'a',0,2),group(2,'b',0,2)]
        val=[group(3,'a',0,2),group(4,'a',1,2),group(5,'b',0,2),group(6,'b',1,2)]
        tst=[group(7,'a',0,2),group(8,'a',1,2)]
        train=(np.zeros(4),np.zeros(4),np.zeros(4))
        # alpha=0 makes validation perfectly ordered, alpha=1 reverses it.
        validation=(np.array([4,4,0,0,4,4,0,0],dtype=float),
                    np.array([0,0,4,4,0,0,4,4],dtype=float),
                    np.zeros(8))
        selected=selected_scoring(train,validation,tr,val,alphas=(0.,1.),calibrations=('identity',),aggregations=('max',))
        self.assertEqual(selected['chosen']['alpha'],0.)
        test_outputs=(np.array([2,2,1,1],float),np.array([1,1,3,3],float),np.zeros(4))
        evaluated=evaluate_selected(train,test_outputs,tr,tst,selected)
        self.assertEqual(evaluated['selection']['chosen']['alpha'],0.)
        self.assertAlmostEqual(evaluated['fragment']['roc_auc'],1.)
        self.assertFalse(evaluated['pointwise']['point_adjusted'])

    def test_weak_one_class_auc_undefined(self):
        self.assertIsNone(auc_ap([0,0],[1,2])['roc_auc'])

if __name__=='__main__':unittest.main()
