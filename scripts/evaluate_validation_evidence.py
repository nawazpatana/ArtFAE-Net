#!/usr/bin/env python3
"""Prespecify and resume a validation-only study of class-conditioned evidence."""
import argparse
from pathlib import Path
import hashlib
import signal
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from artfae import EMOTIONS,MODEL_ID
from artfae.cli import backbone,construct,encode,device_for
from artfae.datasets.manifests import read_manifest,open_image
from artfae.features.aesthetic import normalize
from artfae.explanations.geometry import CLIPGeometry
from artfae.evaluation.faithfulness import evaluate_evidence
from artfae.evaluation.study import select_artworks,run_study,atomic_json
from artfae.evaluation.protocol import sha256
from artfae.evaluation.plots import render_evidence
from artfae.utils.reproducibility import capture


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path('data'));parser.add_argument('--sample-size',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--random-repeats',type=int,default=10)
    parser.add_argument('--budgets',type=float,nargs='+',default=[.1,.2,.3,.5]);parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--bootstrap',type=int,default=1000);parser.add_argument('--resume',action='store_true');parser.add_argument('--allow-smoke',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--booking-start');parser.add_argument('--booking-end')
    args=parser.parse_args()
    if min(args.batch_size,args.bootstrap,args.random_repeats)<1 or not args.budgets or any(not 0<b<1 for b in args.budgets) or len(set(args.budgets))!=len(args.budgets):parser.error('Need positive counts and distinct budgets strictly between zero and one')
    device=device_for(args);torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state.get('emotions')!=list(EMOTIONS) or state.get('backbone')!=MODEL_ID:raise ValueError('Unsupported checkpoint provenance')
    scope=state['config'].get('scope','research')
    if 'smoke' in scope and (not args.allow_smoke or args.sample_size>4 or args.random_repeats>2):raise ValueError('Smoke checkpoints require --allow-smoke, at most four artworks, and at most two random repeats')
    model=construct(state['config']);model.load_state_dict(state['model']);model.to(device).eval()
    if getattr(model,'ecsa',None) is None:raise ValueError('Selected checkpoint has no class-conditioned attention')
    # This command deliberately has no arbitrary split parameter.
    manifest=args.data_dir/'val_artworks.csv';rows=select_artworks(read_manifest(manifest,args.data_dir),args.sample_size,args.seed)
    settings={'scope':scope,'seed':args.seed,'checkpoint_sha256':sha256(args.checkpoint),'validation_manifest_sha256':sha256(manifest),'budgets':args.budgets,'random_repeats':args.random_repeats,'batch_size':args.batch_size,'bootstrap':args.bootstrap,'device':str(device),'class_selection':'predicted emotion on the unperturbed original artwork','images':{r['painting']:[r['path'].stat().st_size,r['path'].stat().st_mtime_ns] for r in rows},'source_sha256':capture(str(device))['source_sha256'],'entrypoint_sha256':sha256(Path(__file__))}
    encoder,processor=backbone();encoder.to(device);stopping=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    def guard():
        device_for(args)
        if stopping[0]:raise InterruptedError('Evidence study interrupted; use --resume')
    def predict_many(images,attention=False):
        probabilities=[];maps=[]
        for start in range(0,len(images),args.batch_size):
            guard();g,a,t,p=encode(encoder,processor,images[start:start+args.batch_size],device)
            a=torch.from_numpy(normalize(a.numpy(),state['normalizer']))
            with torch.no_grad():logits,spatial=model(g.to(device),a.to(device),t.to(device),p.to(device),return_attention=True)
            probabilities.append(logits.softmax(-1).cpu().numpy())
            if attention:maps.append(spatial.cpu().numpy())
        values=np.concatenate(probabilities)
        return (values,np.concatenate(maps)) if attention else values
    def evaluate_one(row):
        guard();image=open_image(row['path']);probabilities,maps=predict_many([image],True);label=int(probabilities[0].argmax())
        geometry=CLIPGeometry.from_processor(image,processor.image_processor);patch=int(encoder.config.vision_config.patch_size)
        shape=(geometry.crop_size[1]//patch,geometry.crop_size[0]//patch);scores=maps[0,label]
        if np.prod(shape)!=scores.size:raise ValueError('Patch output and configured preprocessing geometry disagree')
        local_seed=int.from_bytes(hashlib.sha256(f'{args.seed}:{row["painting"]}'.encode()).digest()[:4],'big')
        results=evaluate_evidence(image,scores.reshape(shape),label,lambda im:predict_many([im])[0],budgets=args.budgets,repeats=args.random_repeats,seed=local_seed,geometry=geometry,predict_batch=predict_many)
        return {'class_index':label,'probabilities':probabilities[0].tolist(),'geometry':geometry.to_dict(),'rows':results}
    run_study(rows,settings,args.output,evaluate_one,resume=args.resume,bootstrap=args.bootstrap)
    atomic_json(args.output/'reproducibility.json',capture(str(device)));render_evidence(args.output)
    print(f'Evidence study complete: {args.output}; {len(rows)} validation artworks; scope={scope}')
if __name__=='__main__':main()
