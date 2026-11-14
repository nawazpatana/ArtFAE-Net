#!/usr/bin/env python3
"""B1 ResNet50 training with pretrained-weight preprocessing and soft labels."""
import argparse,json,sys,signal,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from artfae import EMOTIONS
from artfae.cli import device_for
from artfae.datasets.manifests import read_manifest,validate_disjoint,open_image
from artfae.models.network import ResNetBaseline
from artfae.training.losses import soft_cross_entropy
from artfae.evaluation.metrics import evaluate
from artfae.training.checkpoint import save_state,selection_for_resume

class Images(torch.utils.data.Dataset):
    def __init__(self,rows,transform):self.rows=rows;self.transform=transform
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];return self.transform(open_image(row['path'])),torch.tensor(row['target']),row['painting']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path('data'));p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=.0001);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--booking-start');p.add_argument('--booking-end');p.add_argument('--freeze-backbone',action='store_true');p.add_argument('--cache-features',action='store_true');args=p.parse_args()
    if args.cache_features and not args.freeze_backbone:p.error('--cache-features requires --freeze-backbone')
    if args.epochs<1 or args.batch_size<1:p.error('epochs and batch-size must be positive')
    device=device_for(args);torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    train=read_manifest(args.data_dir/'train_artworks.csv',args.data_dir);val=read_manifest(args.data_dir/'val_artworks.csv',args.data_dir);validate_disjoint(train,val)
    model=ResNetBaseline().to(device)
    if args.freeze_backbone:
        for param in model.network.parameters():param.requires_grad_(False)
        for param in model.network.fc.parameters():param.requires_grad_(True)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','output','booking_start','booking_end')};first=0;best=-1.;history=[];stale=0;selected_checkpoint=None
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        if state['emotions']!=list(EMOTIONS) or state['config']!=config:raise ValueError('Resume class/config mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler']);torch.set_rng_state(state['rng']);first=state['epoch'];best=state['best'];history=state['history'];stale=state['stale']
        if state.get('cuda_rng') and torch.cuda.is_initialized():torch.cuda.set_rng_state_all(state['cuda_rng'])
        selected_checkpoint=selection_for_resume(state,args.resume)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    if selected_checkpoint is not None:save_state(args.output/'best.pt',selected_checkpoint)
    stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    def checkpoint(name,epoch):
        nonlocal selected_checkpoint
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],'epoch':epoch,'best':best,'history':history,'stale':stale,'config':config,'emotions':list(EMOTIONS),'backbone':'torchvision/resnet50/IMAGENET1K_V2','preprocessing':str(model.preprocess),'is_selected_checkpoint':name=='best.pt'}
        if name!='best.pt':state['selection_state']=selected_checkpoint
        save_state(args.output/name,state)
        if name=='best.pt':selected_checkpoint=torch.load(args.output/name,map_location='cpu',weights_only=True)
    checkpoint('last.pt',first)
    trainset=Images(train,model.preprocess);valset=Images(val,model.preprocess)
    if args.cache_features:
        from artfae.features.frozen import extract_frozen,FeatureRows
        trunk=torch.nn.Sequential(*list(model.network.children())[:-1]).to(device).eval()
        def guard():
            device_for(args)
            if stopping[0]:raise InterruptedError('Frozen feature extraction interrupted')
        try:
            train_features=extract_frozen(train,model.preprocess,trunk,device,args.batch_size,guard)
            val_features=extract_frozen(val,model.preprocess,trunk,device,args.batch_size,guard)
        except (InterruptedError,ValueError):
            (args.output/'interruption.json').write_text(json.dumps({'resume_from':str(args.output/'last.pt'),'phase':'feature extraction'})+'\n')
            return
        trainset=FeatureRows(train_features);valset=FeatureRows(val_features)
        torch.save({'train':train_features,'val':val_features,'backbone':'torchvision/resnet50/IMAGENET1K_V2','preprocessing':str(model.preprocess),'emotions':list(EMOTIONS)},args.output/'frozen_features.pt')
    predict=lambda pixels:model.network.fc(pixels) if args.cache_features else model(pixels)
    if args.resume:
        # Feature extraction must not advance the restored training RNG stream.
        torch.set_rng_state(state['rng'])
        if state.get('cuda_rng') and torch.cuda.is_initialized():torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(first,args.epochs):
        if stale>=5:break
        model.train()
        if args.freeze_backbone:model.network.eval();model.network.fc.train()
        total=0.;count=0
        loader=torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed+epoch))
        for pixels,y,_ in loader:
            try:device_for(args)
            except ValueError:stopping[0]=True
            if stopping[0]:break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):loss=soft_cross_entropy(predict(pixels.to(device)).float(),y.to(device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);scaler.step(optimizer);scaler.update();total+=loss.item()*len(y);count+=len(y)
        if stopping[0]:break
        model.eval();pred=[];ys=[];ids=[]
        with torch.no_grad():
            for pixels,y,names in torch.utils.data.DataLoader(valset,batch_size=args.batch_size):
                try:device_for(args)
                except ValueError:stopping[0]=True;break
                if stopping[0]:break
                pred.append(predict(pixels.to(device)).cpu());ys.append(y);ids.extend(names)
        if stopping[0]:break
        logits=torch.cat(pred);metrics=evaluate(logits,torch.cat(ys));score=metrics['macro_f1']
        if score is None:raise ValueError('No unique-top validation artworks')
        scheduler.step();history.append({'epoch':epoch+1,'train_soft_ce':total/count,**metrics});improved=score>best;best=max(best,score);stale=0 if improved else stale+1
        if improved:
            checkpoint('best.pt',epoch+1)
            from artfae.evaluation.protocol import sha256
            metrics.update(split='validation',checkpoint_sha256=sha256(args.output/'best.pt'))
            (args.output/'validation_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
            import csv
            with (args.output/'predictions.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['painting',*EMOTIONS]);w.writerows([name,*prob] for name,prob in zip(ids,logits.softmax(-1).tolist()))
        checkpoint('last.pt',epoch+1)
        (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(f'epoch={epoch+1} macro_f1={score:.6f}',flush=True)
        if stale>=5:break
    if selected_checkpoint is not None and not stopping[0]:
        device_for(args);model.load_state_dict(selected_checkpoint['model']);model.eval();pred=[];ys=[];ids=[]
        with torch.no_grad():
            for pixels,y,names in torch.utils.data.DataLoader(valset,batch_size=args.batch_size):
                device_for(args)
                if stopping[0]:break
                pred.append(predict(pixels.to(device)).cpu());ys.append(y);ids.extend(names)
        if not stopping[0]:
            from artfae.evaluation.protocol import sha256
            logits=torch.cat(pred);metrics=evaluate(logits,torch.cat(ys));metrics.update(split='validation',checkpoint_sha256=sha256(args.output/'best.pt'))
            (args.output/'validation_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
            import csv
            with (args.output/'predictions.csv').open('w',newline='') as file:
                writer=csv.writer(file);writer.writerow(['painting',*EMOTIONS]);writer.writerows([name,*prob] for name,prob in zip(ids,logits.softmax(-1).tolist()))
    from artfae.utils.reproducibility import capture
    from artfae.evaluation.plots import render_run
    (args.output/'reproducibility.json').write_text(json.dumps(capture(str(device)),indent=2)+'\n')
    render_run(args.output)
    if stopping[0]:(args.output/'interruption.json').write_text(json.dumps({'resume_from':str(args.output/'last.pt'),'policy':'Replay incomplete epoch from last complete checkpoint.'})+'\n')
if __name__=='__main__':main()
