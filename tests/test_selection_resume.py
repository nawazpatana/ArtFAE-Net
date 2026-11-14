"""Selection recovery must retain matching model and optimizer state, not only a score."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import tempfile
import unittest
import torch
from artfae.training.checkpoint import selection_for_resume


class SelectionResumeTests(unittest.TestCase):
    def state(self,epoch=2):
        return {'best':.4,'config':{'variant':'M1'},'emotions':['fixed order'],'backbone':'synthetic','epoch':epoch,'model':{'weight':torch.tensor([float(epoch)])},'optimizer':{'step':epoch}}

    def test_legacy_requires_actual_selection_and_validates_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'last.pt';state=self.state(3)
            with self.assertRaisesRegex(ValueError,'sibling'):selection_for_resume(state,source)
            torch.save(self.state(2),source.parent/'best.pt')
            selected=selection_for_resume(state,source)
            self.assertEqual(selected['epoch'],2);self.assertEqual(selected['optimizer']['step'],2)
            torch.save(self.state(4),source.parent/'best.pt')
            with self.assertRaisesRegex(ValueError,'does not match'):selection_for_resume(state,source)

    def test_embedded_selection_is_self_contained(self):
        state=self.state(3);state['selection_state']=self.state(1)
        selected=selection_for_resume(state,Path('/nonexistent/last.pt'))
        self.assertEqual(selected['epoch'],1);self.assertEqual(selected['optimizer']['step'],1)
        self.assertIsNone(selection_for_resume({'best':-1},Path('/nonexistent/last.pt')))


if __name__=='__main__':unittest.main()
