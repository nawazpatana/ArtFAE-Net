#!/usr/bin/env python3
"""Evaluate held-out images only after a frozen validation-selection protocol passes.

This command is intentionally separate from development validation. It has not
been executed against the project test set. Draft protocols are rejected before
loading test targets or pretrained encoders.
"""
import argparse,csv,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from artfae import EMOTIONS,MODEL_ID
from artfae.cli import backbone,encode,construct,device_for
from artfae.datasets.manifests import read_manifest,open_image,validate_disjoint
from artfae.features.aesthetic import normalize,NAMES
from artfae.models.network import FusionBaseline,ResNetBaseline
from artfae.evaluation.metrics import evaluate
from artfae.evaluation.protocol import validate_protocol,sha256
from artfae.utils.reproducibility import capture


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path('data'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--booking-start');parser.add_argument('--booking-end');args=parser.parse_args()
    if args.batch_size<1:parser.error('batch-size must be positive')
    protocol=validate_protocol(args.protocol,args.checkpoint,args.data_dir)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state.get('emotions')!=list(EMOTIONS):raise ValueError('Checkpoint class order mismatch')
    if 'smoke' in state.get('config',{}).get('scope','research'):raise ValueError('Smoke checkpoints cannot support final testing')
    if args.output.exists():raise FileExistsError(args.output)
    device=device_for(args);torch.set_num_threads(4)
    # Only now, after immutable protocol validation, parse the held-out targets.
    test=read_manifest(args.data_dir/'test_artworks.csv',args.data_dir)
    train=read_manifest(args.data_dir/'train_artworks.csv',args.data_dir)
    val=read_manifest(args.data_dir/'val_artworks.csv',args.data_dir);validate_disjoint(train,val,test)
    resnet=state.get('backbone','').startswith('torchvision/resnet50')
    if resnet:model=ResNetBaseline();model.load_state_dict(state['model']);processor=model.preprocess;encoder=None
    elif 'head_state_dict' in state:
        if state.get('model_id')!=MODEL_ID:raise ValueError('Unsupported CLIP backbone')
        model=FusionBaseline(512,len(NAMES),512,'B2');model.head.load_state_dict(state['head_state_dict']);encoder,processor=backbone()
    else:
        if state.get('backbone')!=MODEL_ID:raise ValueError('Unsupported CLIP backbone')
        model=construct(state['config']);model.load_state_dict(state['model']);encoder,processor=backbone()
    model.to(device).eval()
    if encoder is not None:encoder.to(device)
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();logits=[];targets=[];ids=[]
    for offset in range(0,len(test),args.batch_size):
        device_for(args);batch=test[offset:offset+args.batch_size];images=[open_image(r['path']) for r in batch]
        with torch.inference_mode():
            if resnet:prediction=model(torch.stack([processor(image) for image in images]).to(device))
            else:
                g,a,t,patches=encode(encoder,processor,images,device)
                if 'normalizer' in state:a=torch.from_numpy(normalize(a.numpy(),state['normalizer']))
                prediction=model(g.to(device),a.to(device),t.to(device),patches.to(device))
        logits.append(prediction.cpu());targets.extend(r['target'].tolist() for r in batch);ids.extend(r['painting'] for r in batch)
    prediction=torch.cat(logits);metrics=evaluate(prediction,torch.tensor(targets));metrics.update(split='test',protocol_sha256=sha256(args.protocol),checkpoint_sha256=sha256(args.checkpoint),runtime_seconds=time.monotonic()-start)
    (args.output/'test_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.output/'reproducibility.json').write_text(json.dumps(capture(str(device)),indent=2)+'\n')
    with (args.output/'test_predictions.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['painting',*EMOTIONS]);writer.writerows([name,*probs] for name,probs in zip(ids,prediction.softmax(-1).tolist()))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
