"""Artwork-level paired bootstrap, with fixed class order handled by score callback."""
from collections.abc import Callable
import numpy as np


def paired_bootstrap(targets: np.ndarray,first: np.ndarray,second: np.ndarray,score: Callable,repeats: int=1000,seed: int=42) -> dict:
    if len(targets)==0 or len(targets)!=len(first) or len(first)!=len(second):raise ValueError('Unpaired predictions')
    rng=np.random.default_rng(seed);differences=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(targets),len(targets));differences.append(score(targets[idx],first[idx])-score(targets[idx],second[idx]))
    return {'difference':float(score(targets,first)-score(targets,second)),'ci95':np.quantile(differences,[.025,.975]).tolist(),'bootstrap_repeats':repeats,'seed':seed,'unit':'artwork'}


def macro_f1_soft_targets(targets: np.ndarray,probabilities: np.ndarray) -> float:
    """Fixed-nine-class macro F1 after removing tied human top votes."""
    mask=np.isclose(targets,targets.max(1,keepdims=True),atol=1e-5).sum(1)==1
    if not mask.any():return 0.
    true=targets[mask].argmax(1);pred=probabilities[mask].argmax(1)
    cm=np.bincount(true*9+pred,minlength=81).reshape(9,9);tp=cm.diagonal();den=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*tp,den,out=np.zeros(9,dtype=float),where=den>0).mean())
