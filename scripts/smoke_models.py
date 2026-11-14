#!/usr/bin/env python3
"""Tiny CPU training smoke tests using real CLIP features; never research results."""
import argparse,json,sys
from pathlib import Path
from argparse import Namespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import yaml
from artfae import EMOTIONS,MODEL_ID
from artfae.cli import backbone,encode,train
from artfae.datasets.manifests import read_manifest,open_image
from artfae.features.aesthetic import VERSION


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path('data'));p.add_argument('--output',type=Path,required=True);p.add_argument('--resnet',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);args.output.mkdir(parents=True,exist_ok=False);cache=args.output/'features';cache.mkdir();encoder,processor=backbone();metadata={'classes':list(EMOTIONS),'backbone':MODEL_ID,'feature_version':VERSION,'scope':'smoke','splits':{}}
    for split,size in [('train',32),('val',16)]:
        rows=read_manifest(args.data_dir/f'{split}_artworks.csv',args.data_dir)[:size]
        images=[open_image(r['path']) for r in rows];g,a,t,patches=encode(encoder,processor,images,torch.device('cpu'))
        torch.save({'g':g,'a':a,'t':t,'patches':patches,'targets':torch.from_numpy(np.stack([r['target'] for r in rows])),'ids':[r['painting'] for r in rows]},cache/f'{split}.pt');metadata['splits'][split]={'chunks':[f'{split}.pt'],'ids':[r['painting'] for r in rows]}
    (cache/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');del encoder
    for variant in ('B3','B4','B5','M1'):
        config={'variant':variant,'epochs':2,'batch_size':8,'cpu_threads':2,'seed':42,'scope':'smoke','architecture':{'hidden':32}}
        path=args.output/f'{variant}.yaml';path.write_text(yaml.safe_dump(config));train(Namespace(config=path,cache=cache,output=args.output/variant,resume=None,device='cpu',booking_start=None,booking_end=None))
    if args.resnet:
        from artfae.models.network import ResNetBaseline
        from artfae.training.losses import soft_cross_entropy
        model=ResNetBaseline()
        for param in model.network.parameters():param.requires_grad_(False)
        for param in model.network.fc.parameters():param.requires_grad_(True)
        model.eval();model.network.fc.train();optimizer=torch.optim.AdamW(model.network.fc.parameters(),lr=.0001)
        images=[model.preprocess(im) for im in images[:2]];logits=model(torch.stack(images));loss=soft_cross_entropy(logits,torch.ones(2,9)/9);loss.backward();optimizer.step()
        (args.output/'resnet_smoke.json').write_text(json.dumps({'scope':'smoke','device':'cpu','forward_shape':list(logits.shape),'loss_finite':bool(torch.isfinite(loss)),'optimizer_step_completed':True,'preprocessing':str(model.preprocess)},indent=2)+'\n')
    print('CPU_MODEL_SMOKES_PASSED; these are implementation checks, not research metrics.')
if __name__=='__main__':main()
