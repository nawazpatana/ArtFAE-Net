"""Synthetic evidence-study sampling, batching, artwork bootstrap, and recovery."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import tempfile
import unittest
import numpy as np
from PIL import Image
from artfae.evaluation.faithfulness import evaluate_evidence
from artfae.evaluation.study import select_artworks,aggregate_reports,run_study


def result(painting: str='example') -> dict:
    rows=[]
    for method in ('mean','blur'):
        rows.append({'method':method,'budget':.25,'actual_area':.25,'actual_image_area':.2,'original_probability':.8,'deletion_probability':.4,'retention_probability':.7,'deletion_drop':.4,'random_deletion_drop':.2,'deletion_advantage':.2,'retention_advantage':.1})
    return {'painting':painting,'class_index':0,'rows':rows}


class StudyTests(unittest.TestCase):
    def test_sampling_ignores_order_and_annotations(self):
        rows=[{'painting':str(i),'target':'unused'} for i in range(20)]
        selected=select_artworks(rows,5,42)
        changed=[{'painting':r['painting'],'target':'changed label'} for r in reversed(rows)]
        self.assertEqual([r['painting'] for r in selected],[r['painting'] for r in select_artworks(changed,5,42)])
        with self.assertRaises(ValueError):select_artworks(rows,21,42)

    def test_scalar_and_batch_interventions_match(self):
        pixels=np.zeros((32,32,3),dtype=np.uint8);pixels[:16,:16]=255;image=Image.fromarray(pixels)
        def predict(im):
            p=np.asarray(im)[:16,:16].mean()/255
            return np.array([p,*([float(1-p)/8]*8)])
        kwargs={'budgets':[.25,.5],'repeats':3,'seed':7}
        scalar=evaluate_evidence(image,np.array([[1.,0.],[0.,0.]]),0,predict,**kwargs)
        batched=evaluate_evidence(image,np.array([[1.,0.],[0.,0.]]),0,predict,predict_batch=lambda images:np.stack([predict(im) for im in images]),**kwargs)
        self.assertEqual(scalar,batched)
        with self.assertRaisesRegex(ValueError,'probability'):
            evaluate_evidence(image,np.ones((2,2)),0,lambda im:np.zeros(9),repeats=1)

    def test_artwork_bootstrap_and_inconsistent_groups(self):
        summary=aggregate_reports([result('a'),result('b')],bootstrap=30)
        self.assertEqual(summary['bootstrap_unit'],'artwork')
        interval=summary['rows'][0]['bootstrap']['deletion_advantage']['ci95']
        self.assertTrue(np.allclose(interval,[.2,.2]))
        bad=result('c');bad['rows'].pop()
        with self.assertRaisesRegex(ValueError,'Inconsistent'):aggregate_reports([result('a'),bad])

    def test_interrupted_study_reuses_only_completed_artworks(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'study';rows=[{'painting':'a'},{'painting':'b'}];settings={'seed':42,'scope':'synthetic'}
            calls=[]
            def interrupted(row):
                calls.append(row['painting'])
                if row['painting']=='b':raise InterruptedError('synthetic interruption')
                return result()
            with self.assertRaises(InterruptedError):run_study(rows,settings,output,interrupted,bootstrap=20)
            self.assertEqual(json.loads((output/'status.json').read_text())['completed_artworks'],1)
            before={p.name:p.read_bytes() for p in (output/'artworks').glob('*.json')};calls.clear()
            def resumed(row):calls.append(row['painting']);return result()
            summary=run_study(rows,settings,output,resumed,resume=True,bootstrap=20)
            self.assertEqual(calls,['b']);self.assertEqual(summary['artworks'],2)
            self.assertEqual(json.loads((output/'status.json').read_text())['status'],'complete')
            self.assertTrue(all((output/'artworks'/name).read_bytes()==content for name,content in before.items()))
            with self.assertRaisesRegex(ValueError,'changed'):run_study(rows,{'seed':43},output,resumed,resume=True,bootstrap=20)


if __name__=='__main__':unittest.main()
