#!/usr/bin/env python3
"""Independently evaluate the preserved CLIP head on validation only, on CPU."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from transformers import AutoImageProcessor,CLIPVisionModelWithProjection
from artfae import EMOTIONS,MODEL_ID
from artfae.datasets.manifests import read_manifest,open_image
from artfae.evaluation.metrics import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path('data'));p.add_argument('--checkpoint',type=Path,default=Path('results/clip_b32_full/best_head.pt'));p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    if args.batch_size<1 or args.threads<1:p.error('batch-size and threads must be positive')
    torch.set_num_threads(args.threads);args.output.mkdir(parents=True,exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state['emotions']!=list(EMOTIONS) or state['feature_dim']!=512 or state['model_id']!=MODEL_ID:raise ValueError('Incompatible checkpoint')
    head=torch.nn.Linear(512,9);head.load_state_dict(state['head_state_dict']);head.eval()
    model=CLIPVisionModelWithProjection.from_pretrained(MODEL_ID,local_files_only=True).eval()
    processor=AutoImageProcessor.from_pretrained(MODEL_ID,local_files_only=True,use_fast=False)
    rows=read_manifest(args.data_dir/'val_artworks.csv',args.data_dir);logits=[];targets=[];features=[];start=time.monotonic()
    with torch.inference_mode():
        for offset in range(0,len(rows),args.batch_size):
            batch=rows[offset:offset+args.batch_size];images=[open_image(r['path']) for r in batch]
            pixels=processor(images=images,return_tensors='pt')['pixel_values']
            embedding=torch.nn.functional.normalize(model(pixel_values=pixels).image_embeds.float(),dim=-1)
            logits.append(head(embedding));features.append(embedding);targets.extend(r['target'].tolist() for r in batch)
            if offset%(args.batch_size*10)==0:print(f'validation {offset+len(batch)}/{len(rows)}; {time.monotonic()-start:.1f}s',flush=True)
    logits=torch.cat(logits);targets=torch.tensor(targets);metrics=evaluate(logits,targets)
    metrics.update(split='validation',device='cpu',runtime_seconds=time.monotonic()-start,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((args.data_dir/'val_artworks.csv').read_bytes()).hexdigest(),historical_selection_metric='validation soft cross entropy',compute_precision='float32; historical feature extraction used CUDA fp16')
    (args.output/'validation_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    torch.save({'g':torch.cat(features),'targets':targets,'ids':[r['painting'] for r in rows],'model_id':MODEL_ID,'emotions':list(EMOTIONS)},args.output/'validation_features.pt')
    with (args.output/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['painting',*EMOTIONS]);w.writerows([r['painting'],*prob] for r,prob in zip(rows,logits.softmax(-1).tolist()))
    print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':main()
