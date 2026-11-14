"""Verify inverse mapping against installed CLIP resize/crop preprocessing."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor
from artfae.explanations.geometry import CLIPGeometry

class GeometryTests(unittest.TestCase):
    def test_landscape_and_portrait_preprocessing(self):
        processor=CLIPImageProcessor()
        for width,height in ((400,200),(99,301),(256,256)):
            image=Image.fromarray(np.random.default_rng(3).integers(0,256,(height,width,3),dtype=np.uint8));geometry=CLIPGeometry.from_processor(image,processor)
            manual=image.resize(geometry.resized_size,processor.resample).crop((geometry.left,geometry.top,geometry.left+224,geometry.top+224))
            actual=processor(images=image,return_tensors='np')['pixel_values'][0].transpose(1,2,0)
            rgb=np.clip(actual*np.array(processor.image_std)+np.array(processor.image_mean),0,1)
            self.assertTrue(np.array_equal(np.asarray(manual),(rgb*255).round().astype('uint8')))
            self.assertEqual(geometry.map_values(np.ones((7,7))).shape,(height,width))
    def test_visible_crop_and_patch_boxes(self):
        processor=CLIPImageProcessor();geometry=CLIPGeometry.from_processor(Image.new('RGB',(400,200)),processor)
        self.assertEqual(geometry.resized_size,(448,224));self.assertEqual(geometry.left,112)
        self.assertAlmostEqual(geometry.visible_mask().mean(),.5)
        self.assertEqual(geometry.patch_box(0,(1,1)),[100.,0.,300.,200.])
        self.assertFalse(geometry.map_mask(np.ones((7,7)))[:,0].any())
if __name__=='__main__':unittest.main()
