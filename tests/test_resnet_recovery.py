"""Exercise ResNet trainer recovery with a tiny synthetic encoder, not research weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv2d(3,5,1);self.pool=torch.nn.AdaptiveAvgPool2d(1);self.fc=torch.nn.Linear(5,9)
    def forward(self,pixels):return self.fc(self.pool(self.conv(pixels)).flatten(1))


class TinyResNet(torch.nn.Module):
    def __init__(self):super().__init__();self.network=Network()
    def preprocess(self,image):return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
    def forward(self,pixels):return self.network(pixels)


class ResNetRecoveryTests(unittest.TestCase):
    def test_selected_model_survives_completed_run_resume(self):
        source=Path(__file__).resolve().parents[1]/'scripts/train_resnet.py';spec=importlib.util.spec_from_file_location('synthetic_resnet_trainer',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'image.png';Image.new('RGB',(8,8),(30,70,100)).save(path)
            rows={split:[{'painting':f'{split}_{i}','path':path,'target':np.eye(9,dtype=np.float32)[i]} for i in range(2)] for split in ('train','val')}
            base=['train_resnet','--data-dir',str(root),'--epochs','2','--batch-size','2','--freeze-backbone','--cache-features']
            def launch(output,resume=None):
                argv=[*base,'--output',str(output)]
                if resume is not None:argv.extend(['--resume',str(resume)])
                with patch.object(sys,'argv',argv),patch.object(module,'ResNetBaseline',TinyResNet),patch.object(module,'read_manifest',side_effect=lambda p,d:rows[p.name.split('_')[0]]):module.main()
            launch(root/'first');expected=torch.load(root/'first/best.pt',weights_only=True)
            import shutil
            isolated=root/'isolated';isolated.mkdir();shutil.copyfile(root/'first/last.pt',isolated/'last.pt')
            launch(root/'resumed',isolated/'last.pt');actual=torch.load(root/'resumed/best.pt',weights_only=True)
            self.assertEqual(actual['epoch'],expected['epoch'])
            self.assertTrue(all(torch.equal(value,actual['model'][key]) for key,value in expected['model'].items()))
            self.assertTrue((root/'resumed/validation_metrics.json').is_file());self.assertTrue((root/'resumed/predictions.csv').is_file())


if __name__=='__main__':unittest.main()
