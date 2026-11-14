"""Fixed-order metrics with unique-top hard-label filtering."""
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from artfae import EMOTIONS
from artfae.datasets.manifests import validate_targets


def evaluate(logits: torch.Tensor, targets: torch.Tensor, bins: int=15) -> dict:
    y=targets.detach().cpu().numpy(); validate_targets(y)
    if logits.shape != targets.shape or not torch.isfinite(logits).all(): raise ValueError('Invalid logits')
    logp=logits.detach().cpu().double().log_softmax(-1).numpy(); p=np.exp(logp)
    mask=np.isclose(y,y.max(1,keepdims=True),atol=1e-5).sum(1)==1
    hard=y[mask].argmax(1); pred=p[mask].argmax(1)
    precision,recall,f1,support=precision_recall_fscore_support(hard,pred,labels=list(range(9)),zero_division=0)
    ece=0.; confidence=p[mask].max(1) if mask.any() else np.array([])
    for i in range(bins):
        selected=(confidence>=i/bins)&(confidence<(i+1)/bins if i<bins-1 else confidence<=1)
        if selected.any(): ece+=selected.sum()/len(hard)*abs((hard[selected]==pred[selected]).mean()-confidence[selected].mean())
    return {'classes':list(EMOTIONS),'artworks':len(y),'unique_top':int(mask.sum()),'excluded_ties':int((~mask).sum()),'macro_f1':float(f1.mean()) if mask.any() else None,'accuracy':float(accuracy_score(hard,pred)) if mask.any() else None,'balanced_accuracy':float(recall[support>0].mean()) if mask.any() else None,'precision':precision.tolist(),'recall':recall.tolist(),'f1':f1.tolist(),'support':support.tolist(),'confusion_matrix':confusion_matrix(hard,pred,labels=list(range(9))).tolist(),'soft_ce':float(-(y*logp).sum(1).mean()),'kl':float((np.where(y>0,y*np.log(np.maximum(y,1e-30)),0)-y*logp).sum(1).mean()),'brier':float(((y-p)**2).sum(1).mean()),'ece_unique_top':float(ece) if mask.any() else None}
