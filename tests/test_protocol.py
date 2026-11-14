"""Synthetic locked-protocol guards; project held-out data are never opened."""
import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from artfae import EMOTIONS
from artfae.evaluation.protocol import validate_protocol,sha256

class ProtocolTests(unittest.TestCase):
    def test_draft_rejected_before_any_dataset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'draft.json';path.write_text('{"status":"draft"}')
            with self.assertRaisesRegex(ValueError,'locked'):validate_protocol(path,Path('/missing/checkpoint'),Path('/missing/data'))
    def test_lock_binds_selection_and_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'selected.pt';checkpoint.write_bytes(b'synthetic fixture')
            evidence=root/'validation.json';evidence.write_text(json.dumps({'macro_f1':.4,'classes':list(EMOTIONS),'checkpoint_sha256':sha256(checkpoint)}))
            hashes={}
            for split in ('train','val','test'):
                path=root/f'{split}_artworks.csv';path.write_text(f'synthetic {split}');hashes[split]=sha256(path)
            protocol={'status':'locked','selection_split':'val','emotions':list(EMOTIONS),'checkpoint_sha256':sha256(checkpoint),'validation_evidence':{'path':str(evidence),'sha256':sha256(evidence)},'manifest_sha256':hashes,'selection_rationale':'Synthetic test','limitations':'Not a research result'}
            path=root/'protocol.json';path.write_text(json.dumps(protocol));self.assertEqual(validate_protocol(path,checkpoint,root)['status'],'locked')
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint'):validate_protocol(path,checkpoint,root)

class FinalEvaluatorTests(unittest.TestCase):
    def test_synthetic_heldout_pipeline(self):
        """Run the complete evaluator against temporary synthetic fixtures only."""
        import importlib.util
        from unittest.mock import patch
        from types import SimpleNamespace
        import numpy as np
        import torch
        from PIL import Image
        from artfae import MODEL_ID
        from artfae.models.network import ArtFAENet
        from artfae.features.aesthetic import fit_normalizer
        source=Path(__file__).resolve().parents[1]/'scripts/evaluate_locked_test.py'
        spec=importlib.util.spec_from_file_location('artfae_test_evaluator',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'image.png';Image.new('RGB',(8,8)).save(image)
            dims={'visual_dim':12,'attribute_dim':28,'text_dim':10,'patch_dim':11}
            config={'variant':'M1','scope':'research','dimensions':dims,'architecture':{'hidden':8}}
            model=ArtFAENet(**dims,hidden=8);checkpoint=root/'best.pt'
            torch.save({'model':model.state_dict(),'emotions':list(EMOTIONS),'backbone':MODEL_ID,'config':config,'normalizer':fit_normalizer(np.zeros((3,28),dtype=np.float32))},checkpoint)
            evidence=root/'validation.json';evidence.write_text(json.dumps({'macro_f1':.4,'classes':list(EMOTIONS),'checkpoint_sha256':sha256(checkpoint)}));hashes={};rows={}
            for split in ('train','val','test'):
                path=root/f'{split}_artworks.csv';path.write_text('synthetic fixture');hashes[split]=sha256(path)
                rows[split]=[{'painting':f'{split}_{i}','path':image,'target':np.eye(9,dtype=np.float32)[i]} for i in range(2)]
            protocol={'status':'locked','selection_split':'val','emotions':list(EMOTIONS),'checkpoint_sha256':sha256(checkpoint),'validation_evidence':{'path':str(evidence),'sha256':sha256(evidence)},'manifest_sha256':hashes,'selection_rationale':'Synthetic test','limitations':'Synthetic fixtures only'}
            protocol_file=root/'protocol.json';protocol_file.write_text(json.dumps(protocol));output=root/'output'
            def encoded(_encoder,_processor,images,_device):
                n=len(images);return torch.ones(n,12),torch.zeros(n,28),torch.ones(n,10),torch.ones(n,4,11)
            argv=['evaluate_locked_test','--protocol',str(protocol_file),'--checkpoint',str(checkpoint),'--data-dir',str(root),'--output',str(output)]
            with patch.object(sys,'argv',argv),patch.object(module,'read_manifest',side_effect=lambda p,d:rows[p.name.split('_')[0]]),patch.object(module,'backbone',return_value=(SimpleNamespace(to=lambda device:None),None)),patch.object(module,'encode',side_effect=encoded),patch('builtins.print'):
                module.main()
            result=json.loads((output/'test_metrics.json').read_text());self.assertEqual(result['split'],'test');self.assertEqual(result['artworks'],2);self.assertTrue((output/'test_predictions.csv').is_file());self.assertFalse((output/'validation_metrics.json').exists())

if __name__=='__main__':unittest.main()
