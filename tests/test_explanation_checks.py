"""Reject contradictory measurements and unrelated or invalid candidate evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import copy
import unittest
import numpy as np
from PIL import Image
from artfae.features.aesthetic import extract,NAMES
from artfae.explanations.report import explain


class ExplanationCheckTests(unittest.TestCase):
    def setUp(self):
        self.attributes=extract(Image.new('RGB',(32,32),'blue'));self.probabilities=np.array([.6,*([.4/8]*8)])
        self.region={'patch_index':0,'attention':.3,'class_index':0,'box_xyxy':[0,0,10,10]}
    def test_checks_and_uncertainty(self):
        report=explain(self.probabilities,self.attributes,[self.region],image_size=(100,100))
        self.assertTrue(report['consistency_checks']['region_bounds_checked']);self.assertTrue(report['consistency_checks']['region_class_checked'])
        self.assertIn('upper left',report['explanation']);self.assertIn('cool',report['explanation'])
        self.assertTrue(0<=report['uncertainty']['normalized_predictive_entropy']<=1)
        self.assertAlmostEqual(report['uncertainty']['top_two_probability_margin'],.55)
        uniform=explain(np.full(9,1/9,dtype=np.float32),self.attributes,[])
        self.assertTrue(0<=uniform['uncertainty']['normalized_predictive_entropy']<=1)
        with self.assertRaises(ValueError):explain(np.array([1.000001,*([0.]*8)]),self.attributes,[])
    def test_invalid_or_contradictory_measurements(self):
        for invalid in (self.attributes[:-1],np.full(len(NAMES),np.nan)):
            with self.assertRaises(ValueError):explain(self.probabilities,invalid,[])
        invalid=self.attributes.copy();invalid[NAMES.index('warm')]=.9;invalid[NAMES.index('cool')]=.9
        with self.assertRaisesRegex(ValueError,'Contradictory'):explain(self.probabilities,invalid,[])
        invalid=self.attributes.copy();invalid[NAMES.index('bright')]=1;invalid[NAMES.index('dark')]=0
        with self.assertRaisesRegex(ValueError,'contradict'):explain(self.probabilities,invalid,[])
    def test_region_class_bounds_and_duplicate_checks(self):
        for changes in ({'class_index':1},{'box_xyxy':[0,0,101,20]},{'attention':-.1}):
            with self.assertRaises(ValueError):explain(self.probabilities,self.attributes,[{**self.region,**changes}],image_size=(100,100))
        with self.assertRaises(ValueError):explain(self.probabilities,self.attributes,[self.region,copy.deepcopy(self.region)],image_size=(100,100))
    def test_missing_geometry_is_not_claimed_as_checked(self):
        report=explain(np.ones(9)/9,self.attributes,[{'patch_index':0,'attention':.1}])
        self.assertFalse(report['consistency_checks']['region_bounds_checked']);self.assertFalse(report['consistency_checks']['region_class_checked'])
        self.assertEqual(report['predicted_emotion'],'amusement')


if __name__=='__main__':unittest.main()
