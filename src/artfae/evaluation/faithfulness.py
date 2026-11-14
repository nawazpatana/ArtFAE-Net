"""Image-space deletion and retention with matched random region controls.

Predictor must recompute all image-derived features after each intervention.
An optional inverse geometry maps crop rankings onto the original artwork before perturbation.
"""
from collections.abc import Callable,Sequence
import numpy as np
from PIL import Image,ImageFilter


def perturb(image: Image.Image, mask: np.ndarray, method: str) -> Image.Image:
    if mask.shape != (image.height,image.width): raise ValueError('Mask geometry mismatch')
    rgb=np.asarray(image.convert('RGB')).astype(np.float32)
    if method=='mean': replacement=np.broadcast_to(rgb.mean((0,1)),rgb.shape)
    elif method=='blur':replacement=np.asarray(image.filter(ImageFilter.GaussianBlur(radius=max(image.size)/16)))
    else:raise ValueError('Unknown perturbation')
    return Image.fromarray(np.clip(np.where(mask[...,None],replacement,rgb),0,255).astype(np.uint8))


def region_mask(scores: np.ndarray, size: tuple[int,int], budget: float, rng=None) -> np.ndarray:
    if scores.ndim!=2 or not np.isfinite(scores).all() or not 0<budget<1:raise ValueError('Invalid region scores or budget')
    k=max(1,min(scores.size-1,round(scores.size*budget)))
    if scores.size<2:raise ValueError('Need at least two regions')
    order=np.argsort(-scores.ravel(),kind='stable') if rng is None else rng.permutation(scores.size)
    mask=np.zeros(scores.size,dtype=np.uint8);mask[order[:k]]=255
    return np.asarray(Image.fromarray(mask.reshape(scores.shape)).resize(size,Image.Resampling.NEAREST))>0


def evaluate_evidence(image: Image.Image,scores: np.ndarray,class_index: int,predict: Callable[[Image.Image],np.ndarray],budgets: Sequence[float]=(.1,.2,.3,.5),repeats: int=10,seed: int=42,geometry=None,predict_batch: Callable | None=None) -> list[dict]:
    if not 0<=class_index<9 or repeats<1:raise ValueError('Invalid class or random repeats')
    if geometry is not None and geometry.original_size!=image.size:raise ValueError('Inverse geometry/image size mismatch')
    def checked(values,expected):
        values=np.asarray(values)
        if values.shape!=(expected,9) or not np.isfinite(values).all() or (values<0).any() or not np.allclose(values.sum(1),1,atol=1e-5):
            raise ValueError('Predictor must return finite nine-class probability distributions')
        return values
    original=float(checked(np.asarray(predict(image))[None],1)[0,class_index]);rng=np.random.default_rng(seed);rows=[]
    def masks(budget,random=None):
        crop=region_mask(scores,geometry.crop_size if geometry is not None else image.size,budget,random)
        return crop,geometry.map_mask(crop) if geometry is not None else crop
    for method in ('mean','blur'):
        for budget in budgets:
            crop,top=masks(budget)
            interventions=[perturb(image,top,method),perturb(image,~top,method)]
            for _ in range(repeats):
                _,control=masks(budget,rng)
                interventions.extend([perturb(image,control,method),perturb(image,~control,method)])
            probabilities=checked(predict_batch(interventions) if predict_batch is not None else np.stack([predict(im) for im in interventions]),len(interventions))[:,class_index]
            deletion,retention=map(float,probabilities[:2]);random_delete=probabilities[2::2];random_retain=probabilities[3::2]
            rows.append({'method':method,'budget':budget,'actual_area':float(crop.mean()),'actual_image_area':float(top.mean()),'original_probability':original,'deletion_probability':deletion,'retention_probability':retention,'deletion_drop':original-deletion,'random_deletion_drop':original-float(np.mean(random_delete)),'deletion_advantage':float(np.mean(random_delete))-deletion,'retention_advantage':retention-float(np.mean(random_retain)),'random_deletion_std':float(np.std(random_delete)),'random_retention_std':float(np.std(random_retain)),'random_repeats':repeats,'seed':seed})
    return rows
