#!/usr/bin/env python3
"""Reproducible ArtFAE-Net commands; GPU access requires a dated booking window."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime,timezone
import numpy as np
import torch
from artfae import EMOTIONS,MODEL_ID
from artfae.datasets.manifests import read_manifest,validate_disjoint,open_image
from artfae.features.aesthetic import extract,describe,fit_normalizer,normalize,NAMES,VERSION
from artfae.models.network import ArtFAENet,FusionBaseline
from artfae.evaluation.metrics import evaluate
from artfae.training.losses import soft_cross_entropy
from artfae.training.checkpoint import save,load,save_state,selection_for_resume


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def device_for(args):
    if args.device=='cuda':
        if not args.booking_start or not args.booking_end: raise ValueError('CUDA requires ISO booking-start and booking-end with timezone')
        start=datetime.fromisoformat(args.booking_start); end=datetime.fromisoformat(args.booking_end)
        if start.tzinfo is None or end.tzinfo is None or not start<=datetime.now(timezone.utc)<end: raise ValueError('Outside authorized booking')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    return torch.device(args.device)


def audit(args):
    """Audit IDs/counts across splits; test labels and utterance content are withheld."""
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    splits={}; report={'test_policy':'Test IDs, paths, counts only; no test probabilities, utterances, or predictions inspected.','splits':{},'issues':[]}
    for name in ('train','val','test'):
        with (args.data_dir/f'{name}_artworks.csv').open() as f:
            reader=csv.DictReader(f)
            if name=='test': rows=[{k:r[k] for k in ('painting','filename','n_annotations','split')} for r in reader]
            else: rows=read_manifest(args.data_dir/f'{name}_artworks.csv',args.data_dir)
        splits[name]=rows
        report['splits'][name]={'artworks':len(rows),'annotations':sum(int(r['n_annotations']) for r in rows)}
        for r in rows:
            path=args.data_dir/'images'/Path(r['filename']).name
            if not path.is_file(): report['issues'].append({'painting':r['painting'],'issue':'missing image'})
            if args.verify_images and name!='test':
                try:
                    with open_image(path) as image: image.load()
                except Exception as exc: report['issues'].append({'painting':r['painting'],'issue':str(exc)})
    validate_disjoint(*splits.values())
    from collections import Counter
    counts=Counter(); labels={}; bad=0
    allowed={r['painting']:s for s,rows in splits.items() for r in rows}
    with (args.data_dir/'selected_human_annotations.csv').open() as f:
        for row in csv.DictReader(f):
            painting=row['painting']; split=row['split']; counts[split]+=1
            if allowed.get(painting)!=split: bad+=1
            if split!='test':
                emotion={'something else':'something_else'}.get(row['emotion'],row['emotion'])
                if emotion not in EMOTIONS: bad+=1
                labels.setdefault(painting,Counter())[emotion]+=1
    report['annotation_rows_by_split']=dict(counts);report['annotation_id_or_label_issues']=bad
    for split in ('train','val'):
        for r in splits[split]:
            votes=np.array([labels.get(r['painting'],{}).get(e,0) for e in EMOTIONS])
            if votes.sum()!=int(r['n_annotations']) or not np.allclose(votes/max(votes.sum(),1),r['target'],atol=1e-5): report['issues'].append({'painting':r['painting'],'issue':'annotation distribution mismatch'})
    report['artwork_disjoint']=True
    report['annotation_label_mapping']={'something else':'something_else'}
    write(out/'audit.json',report); print(json.dumps({k:v if k!='issues' else {'count':len(v),'examples':v[:5]} for k,v in report.items()},indent=2))


def backbone():
    from transformers import AutoProcessor,CLIPModel
    model=CLIPModel.from_pretrained(MODEL_ID,local_files_only=True).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,AutoProcessor.from_pretrained(MODEL_ID,local_files_only=True,use_fast=False)


def encode(model,processor,images,device):
    attrs=np.stack([extract(im) for im in images]);texts=[describe(a) for a in attrs]
    inputs=processor(text=texts,images=images,return_tensors='pt',padding=True,truncation=True)
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        vision=model.vision_model(pixel_values=inputs['pixel_values'])
        g=torch.nn.functional.normalize(model.visual_projection(vision.pooler_output),dim=-1)
        text=model.text_model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
        t=torch.nn.functional.normalize(model.text_projection(text.pooler_output),dim=-1)
    return g.cpu(),torch.from_numpy(attrs),t.cpu(),vision.last_hidden_state[:,1:].cpu()


def cache(args):
    """Cache immutable artwork-indexed shards with explicit interrupted-run recovery."""
    device=device_for(args); torch.set_num_threads(4)
    resumed=getattr(args,'resume_cache',False)
    if resumed:
        if not (args.output/'progress.json').exists() or (args.output/'metadata.json').exists(): raise ValueError('Resume requires an incomplete cache with progress metadata')
    else: args.output.mkdir(parents=True,exist_ok=False)
    splits={name:read_manifest(args.data_dir/f'{name}_artworks.csv',args.data_dir) for name in ('train','val')}
    validate_disjoint(*splits.values())
    provenance={'backbone':MODEL_ID,'feature_version':VERSION,'classes':list(EMOTIONS),'batch_size':args.batch_size,'manifests':{name:hashlib.sha256((args.data_dir/f'{name}_artworks.csv').read_bytes()).hexdigest() for name in splits},'images':{r['painting']:[r['path'].stat().st_size,r['path'].stat().st_mtime_ns] for rows in splits.values() for r in rows}}
    if resumed and json.loads((args.output/'progress.json').read_text())!=provenance: raise ValueError('Cache provenance changed; use a new directory')
    if not resumed: write(args.output/'progress.json',provenance)
    model,processor=backbone(); model.to(device)
    metadata={'backbone':MODEL_ID,'backbone_revision':getattr(model.config,'_commit_hash',None),'feature_version':VERSION,'classes':list(EMOTIONS),'splits':{},'image_processor':processor.image_processor.to_dict(),'complete':True}
    for split,rows in splits.items():
        chunks=[]; ids=[]
        for start in range(0,len(rows),args.batch_size):
            if args.device=='cuda': device_for(args)
            batch=rows[start:start+args.batch_size]; name=f'{split}_{start:06d}.pt'; path=args.output/name
            expected_ids=[r['painting'] for r in batch]
            if resumed and path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if saved['ids']!=expected_ids or not torch.equal(saved['targets'],torch.from_numpy(np.stack([r['target'] for r in batch]))): raise ValueError('Interrupted shard mismatch')
            else:
                images=[open_image(r['path']) for r in batch]
                g,a,t,patches=encode(model,processor,images,device)
                temporary=path.with_suffix('.tmp')
                torch.save({'g':g,'a':a,'t':t,'patches':patches,'targets':torch.from_numpy(np.stack([r['target'] for r in batch])),'ids':expected_ids},temporary)
                temporary.replace(path)
            chunks.append(name);ids.extend(expected_ids)
            if start%(args.batch_size*10)==0: print(f'{split}: {min(start+args.batch_size,len(rows))}/{len(rows)}',flush=True)
        metadata['splits'][split]={'chunks':chunks,'ids':ids,'manifest_sha256':provenance['manifests'][split]}
    write(args.output/'metadata.json',metadata)


class CachedDataset(torch.utils.data.Dataset):
    """Load one small shard at a time rather than retaining the patch corpus in RAM."""
    def __init__(self,root,split,normalizer=None,spatial=True):
        self.root=root; self.meta=json.loads((root/'metadata.json').read_text()); self.files=self.meta['splits'][split]['chunks'];self.index=[]; self.current=None;self.shard=None;self.normalizer=normalizer;self.spatial=spatial;self.ids=[]
        for file in self.files:
            part=torch.load(root/file,map_location='cpu',weights_only=True,mmap=True)
            self.index.extend((file,i) for i in range(len(part['ids'])));self.ids.extend(part['ids'])
        if len(self.ids)!=len(set(self.ids)): raise ValueError('Repeated IDs in feature cache')
        if self.meta['splits'][split].get('ids',self.ids)!=self.ids: raise ValueError('Cache ID ordering mismatch')
    def __len__(self): return len(self.index)
    def __getitem__(self,index):
        file,i=self.index[index]
        if file!=self.current: self.shard=torch.load(self.root/file,map_location='cpu',weights_only=True,mmap=True);self.current=file
        s=self.shard;a=s['a'][i]
        if self.normalizer: a=torch.from_numpy(normalize(a.numpy(),self.normalizer))
        return s['g'][i],a,s['t'][i],s['patches'][i] if self.spatial else torch.empty(0,0),s['targets'][i],s['ids'][i]


def construct(config):
    dims=config['dimensions']
    if config['variant'] in ('B2','B3','B4','B5'): return FusionBaseline(dims['visual_dim'],dims['attribute_dim'],dims['text_dim'],config['variant'])
    if config['variant']!='M1': raise ValueError('Unsupported model variant; optional M2 training is not implemented')
    return ArtFAENet(**dims,**config.get('architecture',{}))


def report_predictions(model,dataset,device,batch_size,output,guard=None):
    model.eval(); predictions=[];targets=[];ids=[]
    with torch.no_grad():
        for g,a,t,p,y,names in torch.utils.data.DataLoader(dataset,batch_size=batch_size):
            if guard is not None: guard()
            predictions.append(model(g.to(device),a.to(device),t.to(device),p.to(device)).cpu());targets.append(y);ids.extend(names)
    logits=torch.cat(predictions); y=torch.cat(targets); metrics=evaluate(logits,y)
    write(output/'validation_metrics.json',metrics)
    with (output/'predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['painting',*EMOTIONS]);writer.writerows([name,*probs] for name,probs in zip(ids,logits.softmax(-1).tolist()))
    return metrics


def train(args):
    import random,signal,time
    import yaml
    device=device_for(args);config=yaml.safe_load(args.config.read_text());seed=config.get('seed',42)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(config.get('cpu_threads',4))
    meta=json.loads((args.cache/'metadata.json').read_text())
    if meta['backbone']!=MODEL_ID or meta['classes']!=list(EMOTIONS) or meta['feature_version']!=VERSION: raise ValueError('Cache provenance mismatch')
    if meta.get('scope')=='smoke' and config.get('scope')!='smoke': raise ValueError('Smoke cache requires explicit smoke config')
    raw=CachedDataset(args.cache,'train'); attrs=np.stack([raw[i][1].numpy() for i in range(len(raw))]); norm=fit_normalizer(attrs)
    training=CachedDataset(args.cache,'train',norm);validation=CachedDataset(args.cache,'val',norm)
    if set(training.ids)&set(validation.ids): raise ValueError('Train/validation cache overlap')
    sample=training[0];config['dimensions']={'visual_dim':sample[0].numel(),'attribute_dim':sample[1].numel(),'text_dim':sample[2].numel(),'patch_dim':sample[3].shape[-1]}
    config['cache_metadata_sha256']=hashlib.sha256((args.cache/'metadata.json').read_bytes()).hexdigest()
    model=construct(config).to(device)
    training.spatial=validation.spatial=isinstance(model,ArtFAENet) and model.ecsa is not None
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.0003),weight_decay=config.get('weight_decay',.01))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.get('epochs',20))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    first=0;best=-1.;history=[];stale=0;selected_checkpoint=None
    if args.resume:
        state=load(args.resume,model,optimizer)
        if state['config']!=config: raise ValueError('Resume config/cache mismatch')
        norm=state['normalizer'];training.normalizer=norm;validation.normalizer=norm
        first=state['epoch'];best=state['best'];history=state['history'];stale=state['stale'];scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        selected_checkpoint=selection_for_resume(state,args.resume)
    args.output.mkdir(parents=True,exist_ok=False);write(args.output/'config.json',config)
    write(args.output/'history.json',history)
    if selected_checkpoint is not None:save_state(args.output/'best.pt',selected_checkpoint)
    from artfae.utils.reproducibility import capture
    write(args.output/'reproducibility.json',capture(str(device)))
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    batch_size=config.get('batch_size',32);accumulate=config.get('accumulation',1)
    if accumulate<1: raise ValueError('accumulation must be positive')
    # Seeded shard and in-shard permutations keep memory bounded and avoid fixed artwork order.
    start_time=time.monotonic()
    save(args.output/'last.pt',model,optimizer,first,config,norm,best=best,history=history,stale=stale,scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),selection_state=selected_checkpoint,is_selected_checkpoint=False)
    for epoch in range(first,config.get('epochs',20)):
        if stale>=config.get('patience',5):break
        model.train(); total=0.;count=0;optimizer.zero_grad(set_to_none=True)
        generator=torch.Generator().manual_seed(seed+epoch)
        blocks={}
        for index,(file,_) in enumerate(training.index): blocks.setdefault(file,[]).append(index)
        ordered=[]
        keys=list(blocks)
        for block_idx in torch.randperm(len(keys),generator=generator).tolist():
            block=blocks[keys[block_idx]]
            ordered.extend(block[i] for i in torch.randperm(len(block),generator=generator).tolist())
        loader=torch.utils.data.DataLoader(torch.utils.data.Subset(training,ordered),batch_size=batch_size,shuffle=False)
        for index,(g,a,t,p,y,_) in enumerate(loader):
            if device.type=='cuda':
                try: device_for(args)
                except ValueError: stop[0]=True
            if stop[0]: break
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                logits=model(g.to(device),a.to(device),t.to(device),p.to(device));loss=soft_cross_entropy(logits.float(),y.to(device))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            group_size=min(accumulate,len(loader)-(index//accumulate)*accumulate)
            scaler.scale(loss/group_size).backward()
            if (index+1)%accumulate==0 or index+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),config.get('clip_grad',1.),error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            total+=loss.item()*len(y);count+=len(y)
        if stop[0]:
            # Epoch replay: retain the prior epoch checkpoint, avoiding a misleading epoch-complete resume.
            write(args.output/'interruption.json',{'epoch_incomplete':epoch+1,'resume_from':str(args.output/'last.pt') if (args.output/'last.pt').exists() else str(args.resume) if args.resume else None,'message':'Partial epoch discarded; resume last complete epoch.'})
            break
        metrics=report_predictions(model,validation,device,batch_size,args.output,lambda:device_for(args));score=metrics['macro_f1']
        if score is None: raise ValueError('No unique-top validation artworks')
        scheduler.step();history.append({'epoch':epoch+1,'train_soft_ce':total/count,**metrics,'runtime_seconds':time.monotonic()-start_time,'cuda_peak_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0})
        improved=score>best;best=max(best,score);stale=0 if improved else stale+1
        extra={'best':best,'history':history,'stale':stale,'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict()}
        if improved:
            save(args.output/'best.pt',model,optimizer,epoch+1,config,norm,is_selected_checkpoint=True,**extra)
            selected_checkpoint=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
        save(args.output/'last.pt',model,optimizer,epoch+1,config,norm,selection_state=selected_checkpoint,is_selected_checkpoint=False,**extra)
        write(args.output/'history.json',history);print(f'epoch={epoch+1} macro_f1={score:.6f}',flush=True)
        if stale>=config.get('patience',5):break
    if (args.output/'best.pt').exists() and not stop[0]:
        device_for(args)
        load(args.output/'best.pt',model);report_predictions(model,validation,device,batch_size,args.output,lambda:device_for(args))
        from artfae.evaluation.plots import render_run
        from artfae.evaluation.protocol import sha256
        metrics=json.loads((args.output/'validation_metrics.json').read_text())
        metrics.update(split='validation',checkpoint_sha256=sha256(args.output/'best.pt'))
        write(args.output/'validation_metrics.json',metrics)
        render_run(args.output)


def baseline(args):
    device=device_for(args);args.output.mkdir(parents=True,exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state['emotions']!=list(EMOTIONS) or state['model_id']!=MODEL_ID or state['feature_dim']!=512: raise ValueError('Incompatible baseline checkpoint')
    model=FusionBaseline(512,len(NAMES),512,'B2');model.head.load_state_dict(state['head_state_dict']);model.to(device)
    dataset=CachedDataset(args.cache,'val')
    if dataset.meta['backbone']!=MODEL_ID or dataset.meta['classes']!=list(EMOTIONS): raise ValueError('Cache mismatch')
    metrics=report_predictions(model,dataset,device,args.batch_size,args.output,lambda:device_for(args));print(json.dumps(metrics,indent=2))


def infer(args):
    from artfae.explanations.report import explain
    device=device_for(args);args.output.mkdir(parents=True,exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if 'head_state_dict' in state:
        if state['emotions']!=list(EMOTIONS) or state['model_id']!=MODEL_ID: raise ValueError('Incompatible baseline checkpoint')
        model=FusionBaseline(512,len(NAMES),512,'B2');model.head.load_state_dict(state['head_state_dict'])
    else:
        model=construct(state['config']);load(args.checkpoint,model)
    model.to(device).eval()
    encoder,processor=backbone();encoder.to(device);image=open_image(args.image);g,a,t,p=encode(encoder,processor,[image],device)
    raw=a[0].numpy()
    if 'normalizer' in state: a=torch.from_numpy(normalize(a.numpy(),state['normalizer']))
    with torch.no_grad():
        if isinstance(model,ArtFAENet): logits,maps=model(g.to(device),a.to(device),t.to(device),p.to(device),return_attention=True)
        else:logits=model(g.to(device),a.to(device),t.to(device));maps=None
    regions=[]
    if maps is not None:
        from PIL import Image
        from artfae.explanations.geometry import CLIPGeometry
        geometry=CLIPGeometry.from_processor(image,processor.image_processor)
        selected=maps[0,logits.argmax(-1).item()].cpu();side=int(selected.numel()**.5)
        if side*side!=selected.numel(): raise ValueError('Non-square patch grid requires explicit geometry')
        heat=selected.reshape(side,side).numpy();heat=(heat-heat.min())/max(float(heat.max()-heat.min()),1e-8)
        heatmap=Image.fromarray((heat*255).astype('uint8')).resize((224,224))
        # CLIP resizes then center-crops: align evidence to that exact processor view.
        pixels=processor(images=image,return_tensors='pt')['pixel_values'][0].numpy().transpose(1,2,0)
        rgb=np.clip(pixels*np.array(processor.image_processor.image_std)+np.array(processor.image_processor.image_mean),0,1)
        overlay=rgb*.6+np.stack([np.asarray(heatmap)/255,np.zeros((224,224)),np.zeros((224,224))],-1)*.4
        Image.fromarray((overlay*255).astype('uint8')).save(args.output/'overlay.png');heatmap.save(args.output/'heatmap.png')
        selected_mask=torch.zeros_like(selected);selected_mask[selected.argsort(descending=True)[:max(1,int(.2*len(selected)))]]=1
        Image.fromarray((selected_mask.reshape(side,side).numpy()*255).astype('uint8')).resize((224,224),Image.Resampling.NEAREST).save(args.output/'regions.png')
        full_heat=geometry.map_values(heat);visible=geometry.visible_mask();full_heat[~visible]=0
        original_rgb=np.asarray(image,dtype=np.float32)/255
        tint=np.stack([full_heat,np.zeros_like(full_heat),np.zeros_like(full_heat)],-1)
        alpha=visible[...,None].astype(np.float32)*.4
        full_overlay=original_rgb*(1-alpha)+tint*alpha
        Image.fromarray((np.clip(full_overlay,0,1)*255).astype('uint8')).save(args.output/'overlay_original.png')
        Image.fromarray((full_heat*255).astype('uint8')).save(args.output/'heatmap_original.png')
        Image.fromarray((geometry.map_mask(selected_mask.reshape(side,side).numpy()>0)*255).astype('uint8')).save(args.output/'regions_original.png')
        write(args.output/'geometry.json',geometry.to_dict())
        regions=[{'patch_index':int(i),'attention':float(selected[i]),'class_index':int(logits.argmax(-1).item()),'coordinate_system':'Original decoded artwork pixels','box_xyxy':geometry.patch_box(int(i),(side,side))} for i in selected.argsort(descending=True)[:max(1,int(.2*len(selected)))]]
    report=explain(logits.softmax(-1)[0].cpu().numpy(),raw,regions,image_size=image.size)
    report['checkpoint_scope']=state.get('config',{}).get('scope','research')
    if report['checkpoint_scope']=='smoke': report['limitations']+=' Tiny smoke checkpoint: predictions are implementation checks, not research findings.'
    report['attention_available']=maps is not None
    if maps is None: report['limitations']+=' This global-only baseline has no class-conditioned spatial evidence.'
    write(args.output/'inference.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for name,fn in [('audit',audit),('cache',cache),('train',train),('baseline',baseline),('infer',infer)]:
        p=sub.add_parser(name,help=fn.__doc__);p.set_defaults(fn=fn)
        p.add_argument('--output',type=Path,required=True);p.add_argument('--data-dir',type=Path,default=Path('data'))
        p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--booking-start');p.add_argument('--booking-end');p.add_argument('--batch-size',type=int,default=32)
        if name=='audit':p.add_argument('--verify-images',action='store_true')
        if name=='cache':p.add_argument('--resume-cache',action='store_true')
        if name in ('train','baseline'):p.add_argument('--cache',type=Path,required=True)
        if name=='train':p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',type=Path)
        if name in ('baseline','infer'):p.add_argument('--checkpoint',type=Path,required=True)
        if name=='infer':p.add_argument('--image',type=Path,required=True)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch-size must be positive')
    output_existed=args.output.exists()
    try:
        args.fn(args)
    except Exception as exc:
        if not output_existed and args.output.exists(): write(args.output/'error.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
if __name__=='__main__':main()
