"""Versioned deterministic image measurements, not subjective aesthetic judgments."""
import numpy as np
from PIL import Image
VERSION = 'rgb256-v1'
NAMES = ('red','green','blue','hue_sin','hue_cos','saturation','saturation_var','value','warm','cool','luminance','luminance_std','dark','bright','contrast_p95_p05','edge_density','edge_tl','edge_tr','edge_bl','edge_br') + tuple(f'hue_bin_{i}' for i in range(8))


def extract(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize((256,256), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32)/255
    hsv = np.asarray(image.convert('HSV'), dtype=np.float32)/255
    h,s,v = hsv.transpose(2,0,1)
    y = rgb @ np.array([.2126,.7152,.0722])
    dy,dx = np.gradient(y); edge = np.hypot(dx,dy) > .08
    chromatic = s > .15
    warm = (((h < .17) | (h > .92)) & chromatic).mean()
    cool = ((h >= .33) & (h <= .75) & chromatic).mean()
    hist = np.histogram(h, bins=8, range=(0,1), weights=s)[0]; hist = hist/max(hist.sum(),1e-8)
    return np.asarray([*rgb.mean((0,1)), np.sin(h*2*np.pi).mean(),np.cos(h*2*np.pi).mean(),s.mean(),s.var(),v.mean(),warm,cool,y.mean(),y.std(),(y<.2).mean(),(y>.8).mean(),np.percentile(y,95)-np.percentile(y,5),edge.mean(),edge[:128,:128].mean(),edge[:128,128:].mean(),edge[128:,:128].mean(),edge[128:,128:].mean(),*hist],dtype=np.float32)


def describe(a: np.ndarray) -> str:
    d = dict(zip(NAMES,a))
    brightness = 'Low' if d['luminance'] < .33 else 'High' if d['luminance'] > .67 else 'Moderate'
    contrast = 'low' if d['luminance_std'] < .12 else 'high' if d['luminance_std'] > .25 else 'moderate'
    colors = 'predominantly warm' if d['warm'] > d['cool']+.1 else 'predominantly cool' if d['cool'] > d['warm']+.1 else 'mixed or neutral'
    return f'{brightness} overall brightness, {contrast} luminance contrast, and {colors} colors.'


def fit_normalizer(train: np.ndarray) -> dict:
    if not np.isfinite(train).all(): raise ValueError('Nonfinite attributes')
    return {'mean':train.mean(0).tolist(),'std':np.maximum(train.std(0),1e-6).tolist(),'version':VERSION,'names':list(NAMES),'fit_split':'train'}


def normalize(a: np.ndarray, state: dict) -> np.ndarray:
    if state['version'] != VERSION or state['names'] != list(NAMES) or state['fit_split'] != 'train': raise ValueError('Normalizer provenance mismatch')
    return ((a-np.asarray(state['mean']))/np.asarray(state['std'])).astype(np.float32)

GROUPS = {'color': list(range(0,10))+list(range(20,28)), 'brightness': [10,12,13], 'contrast': [11,14], 'complexity': list(range(15,20))}
