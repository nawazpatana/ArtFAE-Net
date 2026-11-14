"""CPU-only scientific protocol checks, no model loading or inference."""
from engine import *
from transformers import CLIPProcessor
proc=CLIPProcessor.from_pretrained(P['encoder'],revision=P['encoder_revision'],use_fast=False,local_files_only=True)
rng=np.random.default_rng(275)
for w,h in [(640,480),(480,640),(336,336)]:
 im=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8));original=np.asarray(im).copy();op=Intervention(im,proc)
 for method in ['mean','blur']:
  assert np.array_equal(np.asarray(op.apply(np.ones((336,336),np.float32),method)),original),'Identity mask changed pixels'
  zero=np.asarray(op.apply(np.zeros((336,336),np.float32),method))
  if w>h:assert np.array_equal(zero[:,0,:],original[:,0,:]),'Outside horizontal crop changed'
  if h>w:assert np.array_equal(zero[0,:,:],original[0,:,:]),'Outside vertical crop changed'
 assert np.array_equal(np.asarray(im),original),'Source object mutated'
for k in P['mask_counts']:
 a=cells_keep(np.arange(49),k);r=cells_keep(np.arange(49),k,True)
 assert int((1-a).sum())==k*48*48 and np.array_equal(a+r,np.ones_like(a))
x=np.array(P['mask_fractions']);width=x[-1]-x[0]
assert abs(np.trapezoid(np.full(4,.3),x)/width-.3)<1e-12
assert abs(np.trapezoid(x,x)/width-(x[0]+x[-1])/2)<1e-12
r={'identity_masks_preserve_source_pixels':True,'source_objects_unchanged':True,'outside_visible_crop_preserved':True,'deletion_retention_complements_correct':True,'mask_counts_match_declared_fractions':True,'constant_and_linear_partial_AUC_analytic_checks':True,'CPU_only':True,'no_model_loaded':True}
(O/'protocol_checks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
