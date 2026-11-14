"""Interrupted epoch replay must match uninterrupted CPU optimization exactly."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from argparse import Namespace
import json
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml
from artfae import EMOTIONS,MODEL_ID
from artfae.features.aesthetic import VERSION
from artfae.cli import train,report_predictions


class TrainingRecoveryTests(unittest.TestCase):
    def test_epoch_replay_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir();metadata={'classes':list(EMOTIONS),'backbone':MODEL_ID,'feature_version':VERSION,'splits':{}}
            generator=torch.Generator().manual_seed(123)
            for split in ('train','val'):
                part={'g':torch.randn(9,12,generator=generator),'a':torch.randn(9,28,generator=generator),'t':torch.randn(9,10,generator=generator),'patches':torch.randn(9,4,11,generator=generator),'targets':torch.eye(9),'ids':[f'{split}_{i}' for i in range(9)]}
                torch.save(part,cache/f'{split}.pt');metadata['splits'][split]={'chunks':[f'{split}.pt']}
            (cache/'metadata.json').write_text(json.dumps(metadata));config=root/'config.yaml';config.write_text(yaml.safe_dump({'variant':'M1','epochs':3,'batch_size':4,'cpu_threads':2,'architecture':{'hidden':8},'patience':5}))
            args=Namespace(config=config,cache=cache,output=root/'uninterrupted',resume=None,device='cpu',booking_start=None,booking_end=None)
            train(args);expected=torch.load(args.output/'last.pt',weights_only=True)
            args.output=root/'interrupted';calls=[0]
            def interrupt_second_validation(*positional,**keywords):
                calls[0]+=1
                if calls[0]==2:raise InterruptedError('Synthetic interruption after epoch-two optimization')
                return report_predictions(*positional,**keywords)
            with patch('artfae.cli.report_predictions',side_effect=interrupt_second_validation):
                with self.assertRaises(InterruptedError):train(args)
            checkpoint=args.output/'last.pt';self.assertEqual(torch.load(checkpoint,weights_only=True)['epoch'],1)
            args.output=root/'resumed';args.resume=checkpoint;train(args);actual=torch.load(args.output/'last.pt',weights_only=True)
            self.assertEqual(actual['epoch'],3)
            self.assertTrue(all(torch.equal(value,actual['model'][key]) for key,value in expected['model'].items()))
            self.assertTrue(torch.equal(expected['rng'],actual['rng']))
            self.assertEqual(expected['scheduler'],actual['scheduler'])
            self.assertEqual([r['train_soft_ce'] for r in expected['history']],[r['train_soft_ce'] for r in actual['history']])


if __name__=='__main__':unittest.main()
