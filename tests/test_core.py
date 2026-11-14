"""CPU tests with synthetic images and distributions; no held-out test access."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest,tempfile
import numpy as np
import torch
from PIL import Image
from artfae import EMOTIONS
from artfae.datasets.manifests import validate_targets,validate_disjoint,open_image,read_manifest
from artfae.features.aesthetic import extract,describe,NAMES,fit_normalizer,normalize
from artfae.models.network import AGF,ECSA,ArtFAENet,FusionBaseline
from artfae.training.losses import soft_cross_entropy
from artfae.training.checkpoint import save,load
from artfae.evaluation.metrics import evaluate
from artfae.explanations.report import explain

class CoreTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(2)
    def test_order(self):self.assertEqual(EMOTIONS,('amusement','awe','contentment','excitement','anger','disgust','fear','sadness','something_else'))
    def test_distributions(self):
        validate_targets(np.ones((3,9))/9)
        for bad in (np.zeros((2,9)),np.ones((2,8))/8,np.full((2,9),np.nan),np.array([[-1,2,0,0,0,0,0,0,0]])):
            with self.assertRaises(ValueError):validate_targets(bad)
    def test_overlap(self):
        validate_disjoint([{'painting':'a'}],[{'painting':'b'}])
        with self.assertRaises(ValueError):validate_disjoint([{'painting':'a'}],[{'painting':'a'}])
    def test_attributes(self):
        a=extract(Image.new('RGB',(50,90),'black'));b=extract(Image.new('RGB',(90,50),'white'))
        self.assertEqual(len(a),len(NAMES));self.assertTrue(np.isfinite(a).all());self.assertEqual(a[NAMES.index('dark')],1);self.assertEqual(b[NAMES.index('bright')],1)
        self.assertIn('Low',describe(a));self.assertIn('High',describe(b))
        self.assertIn('warm',describe(extract(Image.new('RGB',(32,32),'red'))));self.assertIn('cool',describe(extract(Image.new('RGB',(32,32),'blue'))))
        norm=fit_normalizer(np.stack([a,b]));self.assertTrue(np.allclose(normalize(np.stack([a,b]),norm).mean(0),0,atol=1e-5))
        for text in (describe(a),describe(b)):
            for emotion in EMOTIONS:self.assertNotIn(emotion,text.lower())
    def test_agf_backward(self):
        m=AGF(12,7,10,8);x=m(torch.randn(3,12),torch.randn(3,7),torch.randn(3,10));self.assertEqual(x.shape,(3,8));x.square().mean().backward();self.assertTrue(all(p.grad is not None for p in m.parameters()))
    def test_ecsa_backward(self):
        m=ECSA(11,8);logits,maps=m(torch.randn(3,8),torch.randn(3,15,11));self.assertEqual(maps.shape,(3,9,15));self.assertTrue(torch.allclose(maps.sum(-1),torch.ones(3,9)));logits.square().mean().backward();self.assertTrue(all(p.grad is not None for p in m.parameters()))
    def test_variants(self):
        g,a,t,p=torch.randn(2,12),torch.randn(2,7),torch.randn(2,10),torch.randn(2,16,11)
        for variant in ('B2','B3','B4','B5'):self.assertEqual(FusionBaseline(12,7,10,variant)(g,a,t,p).shape,(2,9))
        for spatial in (True,False):
            model=ArtFAENet(12,7,10,11,hidden=8,spatial=spatial);logits,maps=model(g,a,t,p,True);self.assertEqual(logits.shape,(2,9));soft_cross_entropy(logits,torch.ones(2,9)/9).backward()
    def test_attribute_group_mask(self):
        model=ArtFAENet(12,7,10,11,hidden=8,attribute_indices=[0,1]).eval()
        g,a,t,p=torch.randn(2,12),torch.randn(2,7),torch.randn(2,10),torch.randn(2,4,11)
        changed=a.clone();changed[:,2:]+=100
        self.assertTrue(torch.equal(model(g,a,t,p),model(g,changed,t,p)))
    def test_loss(self):
        logits=torch.zeros(3,9,requires_grad=True);loss=soft_cross_entropy(logits,torch.ones(3,9)/9);self.assertAlmostEqual(loss.item(),np.log(9),places=5);loss.backward()
        with self.assertRaises(ValueError):soft_cross_entropy(logits,torch.zeros(3,9))
    def test_metrics(self):
        y=torch.eye(9);m=evaluate(y*50,y);self.assertAlmostEqual(m['macro_f1'],1);self.assertAlmostEqual(m['accuracy'],1);self.assertLess(m['brier'],1e-10)
        y=torch.cat([y,torch.ones(1,9)/9]);m=evaluate(y,y);self.assertEqual(m['excluded_ties'],1);self.assertEqual(m['unique_top'],9)
    def test_resume(self):
        model=torch.nn.Linear(3,9);opt=torch.optim.AdamW(model.parameters(),lr=.01);x=torch.randn(2,3);y=torch.ones(2,9)/9
        soft_cross_entropy(model(x),y).backward();opt.step();opt.zero_grad()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'last.pt';save(path,model,opt,1,{},{});expected=model(x).detach().clone()
            fresh=torch.nn.Linear(3,9);newopt=torch.optim.AdamW(fresh.parameters(),lr=.01);state=load(path,fresh,newopt)
            self.assertEqual(state['epoch'],1);self.assertTrue(torch.equal(expected,fresh(x)));self.assertEqual(len(newopt.state),2)
            for m,o in ((model,opt),(fresh,newopt)):soft_cross_entropy(m(x),y).backward();o.step()
            self.assertTrue(torch.equal(model(x),fresh(x)))
    def test_image_and_explanation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'image.png';Image.new('RGBA',(41,32),(0,0,255,255)).save(p);im=open_image(p);self.assertEqual(im.mode,'RGB');a=extract(im)
            report=explain(np.ones(9)/9,a,[]);self.assertIn(describe(a),report['explanation']);self.assertEqual(report['measured_attributes']['blue'],1)
    def test_manifest(self):
        import csv
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'images').mkdir();Image.new('RGB',(8,8)).save(root/'images/a.png')
            file=root/'train.csv';keys=['painting','filename','n_annotations',*[f'p_{e}' for e in EMOTIONS]]
            with file.open('w') as f:
                writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerow(dict(painting='a',filename='a.png',n_annotations='9',**{f'p_{e}':1/9 for e in EMOTIONS}))
            self.assertEqual(read_manifest(file,root)[0]['painting'],'a')
if __name__=='__main__':unittest.main()
