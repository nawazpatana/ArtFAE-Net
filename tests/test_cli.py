"""End-to-end synthetic cached training and epoch resume without GPU or artworks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest,tempfile,json
from pathlib import Path
from argparse import Namespace
import torch
import yaml
from artfae import EMOTIONS,MODEL_ID
from artfae.features.aesthetic import VERSION
from artfae.cli import train,device_for

class CLITests(unittest.TestCase):
    def test_booking_guard(self):
        with self.assertRaises(ValueError):device_for(Namespace(device='cuda',booking_start=None,booking_end=None))
    def test_interrupted_cache_recovery(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import numpy as np
        from PIL import Image
        from artfae.cli import cache as build_cache
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir();image=data/'image.png';Image.new('RGB',(8,8)).save(image)
            rows={}
            for split in ('train','val'):
                (data/f'{split}_artworks.csv').write_text(split)
                rows[split]=[{'painting':split,'path':image,'target':np.ones(9,dtype=np.float32)/9}]
            model=SimpleNamespace(config=SimpleNamespace(_commit_hash='synthetic'),to=lambda device:None)
            processor=SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda:{}))
            args=Namespace(data_dir=data,output=root/'cache',device='cpu',booking_start=None,booking_end=None,batch_size=1,resume_cache=False)
            encoded=(torch.ones(1,12),torch.ones(1,28),torch.ones(1,10),torch.ones(1,4,11))
            with patch('artfae.cli.read_manifest',side_effect=lambda p,d:rows[p.name.split('_')[0]]),patch('artfae.cli.backbone',return_value=(model,processor)),patch('artfae.cli.encode',side_effect=[encoded,InterruptedError('test interruption')]):
                with self.assertRaises(InterruptedError):build_cache(args)
            shard=args.output/'train_000000.pt';before=hashlib.sha256(shard.read_bytes()).hexdigest();self.assertFalse((args.output/'metadata.json').exists())
            args.resume_cache=True
            with patch('artfae.cli.read_manifest',side_effect=lambda p,d:rows[p.name.split('_')[0]]),patch('artfae.cli.backbone',return_value=(model,processor)),patch('artfae.cli.encode',return_value=encoded) as encoder:
                build_cache(args);self.assertEqual(encoder.call_count,1)
            self.assertEqual(before,hashlib.sha256(shard.read_bytes()).hexdigest());self.assertTrue((args.output/'metadata.json').exists())
    def test_cached_training_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir();meta={'classes':list(EMOTIONS),'backbone':MODEL_ID,'feature_version':VERSION,'splits':{}}
            for split in ('train','val'):
                n=9;part={'g':torch.randn(n,12),'a':torch.randn(n,28),'t':torch.randn(n,10),'patches':torch.randn(n,4,11),'targets':torch.eye(9),'ids':[f'{split}_{i}' for i in range(n)]}
                torch.save(part,cache/f'{split}.pt');meta['splits'][split]={'chunks':[f'{split}.pt']}
            (cache/'metadata.json').write_text(json.dumps(meta));cfg=root/'config.yaml';cfg.write_text(yaml.safe_dump({'variant':'M1','epochs':2,'batch_size':4,'cpu_threads':2,'architecture':{'hidden':8},'patience':5}))
            args=Namespace(config=cfg,cache=cache,output=root/'run',resume=None,device='cpu',booking_start=None,booking_end=None)
            train(args);self.assertTrue((args.output/'best.pt').exists());self.assertEqual(len(json.loads((args.output/'history.json').read_text())),2)
            selected=torch.load(args.output/'best.pt',weights_only=True)
            # A new-format last checkpoint must remain self-contained even if
            # copied away from the original directory and its sibling best.pt.
            import shutil
            isolated=root/'isolated';isolated.mkdir();shutil.copyfile(args.output/'last.pt',isolated/'last.pt')
            # Same configuration can resume at its completed epoch without replay.
            args.resume=isolated/'last.pt';args.output=root/'resume';train(args)
            state=torch.load(args.output/'last.pt',weights_only=True);self.assertEqual(state['epoch'],2)
            self.assertEqual(state['emotions'],list(EMOTIONS))
            restored=torch.load(args.output/'best.pt',weights_only=True)
            self.assertEqual(restored['epoch'],selected['epoch'])
            self.assertTrue(all(torch.equal(value,restored['model'][key]) for key,value in selected['model'].items()))
            self.assertTrue((args.output/'validation_metrics.json').is_file())
            self.assertEqual(len(json.loads((args.output/'history.json').read_text())),2)
if __name__=='__main__':unittest.main()
