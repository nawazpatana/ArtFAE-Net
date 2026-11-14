"""Intervention geometry and paired statistical checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
from PIL import Image
from artfae.evaluation.faithfulness import perturb,region_mask,evaluate_evidence
from artfae.evaluation.statistics import paired_bootstrap

class EvidenceTests(unittest.TestCase):
    def test_matched_area(self):
        scores=np.arange(49).reshape(7,7)
        top=region_mask(scores,(224,224),.2);random=region_mask(scores,(224,224),.2,np.random.default_rng(3));self.assertEqual(top.sum(),random.sum())
    def test_interventions(self):
        rgb=np.zeros((32,32,3),dtype=np.uint8);rgb[:16,:16]=255;image=Image.fromarray(rgb);scores=np.array([[1,0],[0,0]])
        def predict(im):
            value=np.asarray(im)[:16,:16].mean()/255;return np.array([value,*([float(1-value)/8]*8)])
        rows=evaluate_evidence(image,scores,0,predict,budgets=[.25],repeats=4)
        self.assertEqual(len(rows),2);self.assertGreater(rows[0]['deletion_advantage'],0);self.assertGreater(rows[0]['retention_advantage'],0)
        with self.assertRaises(ValueError):perturb(image,np.zeros((2,2),bool),'mean')
    def test_bootstrap(self):
        y=np.arange(20);r=paired_bootstrap(y,y,y,lambda a,b:float((a==b).mean()),repeats=20);self.assertEqual(r['ci95'],[0,0])
