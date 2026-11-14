#!/usr/bin/env python3
"""Booking-gated real-image CUDA smoke: frozen CLIP, AGF/ECSA, backward, checkpoint."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from artfae.cli import backbone,encode,device_for
from artfae.datasets.manifests import read_manifest,open_image
from artfae.features.aesthetic import fit_normalizer,normalize
from artfae.models.network import ArtFAENet
from artfae.training.losses import soft_cross_entropy
from artfae.training.checkpoint import save,load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path('data'));p.add_argument('--output',type=Path,required=True);p.add_argument('--booking-start',required=True);p.add_argument('--booking-end',required=True);p.set_defaults(device='cuda');args=p.parse_args()
    device=device_for(args);torch.manual_seed(42);args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    rows=read_manifest(args.data_dir/'train_artworks.csv',args.data_dir)[:4];encoder,processor=backbone();encoder.to(device)
    g,a,t,patches=encode(encoder,processor,[open_image(r['path']) for r in rows],device);del encoder
    normalizer=fit_normalizer(a.numpy());a=torch.from_numpy(normalize(a.numpy(),normalizer));dims={'visual_dim':g.shape[1],'attribute_dim':a.shape[1],'text_dim':t.shape[1],'patch_dim':patches.shape[-1]};config={'variant':'M1','scope':'gpu_smoke','seed':42,'dimensions':dims,'architecture':{'hidden':128}}
    model=ArtFAENet(**dims).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003);target=torch.from_numpy(np.stack([r['target'] for r in rows])).to(device)
    for _ in range(3):
        device_for(args);optimizer.zero_grad(set_to_none=True);logits=model(g.to(device),a.to(device),t.to(device),patches.to(device));loss=soft_cross_entropy(logits,target)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite smoke loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    save(args.output/'smoke.pt',model,optimizer,1,config,normalizer);load(args.output/'smoke.pt',model,optimizer)
    torch.cuda.synchronize();report={'scope':'smoke only','device':torch.cuda.get_device_name(device),'optimizer_steps':3,'finite_loss':bool(torch.isfinite(loss)),'cuda_peak_bytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-start,'checkpoint_roundtrip':True};(args.output/'gpu_smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
