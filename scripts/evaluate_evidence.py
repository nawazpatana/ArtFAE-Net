#!/usr/bin/env python3
"""Run image-space evidence interventions on a supplied non-test artwork."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from artfae.cli import backbone,construct,encode
import numpy as np
import torch
from artfae.datasets.manifests import open_image
from artfae.features.aesthetic import normalize
from artfae.training.checkpoint import load
from artfae.evaluation.faithfulness import evaluate_evidence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--random-repeats',type=int,default=10);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    torch.set_num_threads(4);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True);model=construct(state['config']);load(args.checkpoint,model);model.eval()
    if getattr(model,'ecsa',None) is None:raise ValueError('Checkpoint has no class-conditioned evidence module')
    encoder,processor=backbone();image=open_image(args.image)
    from artfae.explanations.geometry import CLIPGeometry
    geometry=CLIPGeometry.from_processor(image,processor.image_processor)
    def predict(im,attention=False):
        g,a,t,patches=encode(encoder,processor,[im],torch.device('cpu'));a=torch.from_numpy(normalize(a.numpy(),state['normalizer']))
        with torch.no_grad():
            logits,maps=model(g,a,t,patches,return_attention=True)
        probs=logits.softmax(-1)[0].numpy()
        return (probs,maps) if attention else probs
    probs,maps=predict(image,True);label=int(probs.argmax());scores=maps[0,label].numpy();side=int(scores.size**.5)
    if side*side!=scores.size:raise ValueError('Unsupported non-square patch grid')
    rows=evaluate_evidence(image,scores.reshape(side,side),label,predict,repeats=args.random_repeats,seed=args.seed,geometry=geometry)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'faithfulness.json').write_text(json.dumps({'scope':state['config'].get('scope','research'),'image':str(args.image),'coordinate_system':'Original decoded artwork; inverse CLIP crop mapping; all image-derived features recomputed after every perturbation','geometry':geometry.to_dict(),'class_index':label,'rows':rows,'limitations':'Perturbation sensitivity does not prove semantic correctness.'},indent=2)+'\n')
    from artfae.evaluation.plots import render_evidence
    render_evidence(args.output)
if __name__=='__main__':main()
