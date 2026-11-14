"""Frozen image features preserve IDs, targets, and direct encoder predictions."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import numpy as np
from PIL import Image
from artfae.features.frozen import extract_frozen,FeatureRows

class FrozenTests(unittest.TestCase):
    def test_cached_matches_direct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[]
            for index in range(3):
                path=root/f'{index}.png';Image.new('RGB',(8,8),(index*40,50,90)).save(path);rows.append({'path':path,'painting':str(index),'target':np.ones(9,dtype=np.float32)/9})
            transform=lambda image:torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
            encoder=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(192,5)).eval()
            state=extract_frozen(rows,transform,encoder,torch.device('cpu'),batch_size=2)
            with Image.open(rows[0]['path']) as image:expected=encoder(transform(image)[None]).detach()[0]
            self.assertTrue(torch.allclose(expected,state['features'][0],atol=1e-6));self.assertEqual(state['ids'],['0','1','2']);self.assertEqual(len(FeatureRows(state)),3);self.assertEqual(state['targets'].shape,(3,9))
if __name__=='__main__':unittest.main()
