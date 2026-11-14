"""Exploratory validation-only inference. Never trains or opens holdout features."""
from pathlib import Path
import sys,hashlib,json,time,itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image,ImageFilter
W=Path(__file__).resolve().parents[1];O=W/'remaining_experiments'
sys.path[:0]=[str(W),str(W/'src'),str(W/'runs/revision_B/code')]
from train_abl import build
from artfae_balance.extract_features import load_image,compute_attributes_and_text
from artfae.features.aesthetic import normalize
E=['amusement','awe','contentment','excitement','anger','disgust','fear','sadness','something_else']
P=json.loads((O/'protocol.json').read_text());V=pd.read_csv(W/'data/artemis_balanced/val.csv');SUB=pd.read_csv(O/'validation_subset.csv')
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def head(run,device='cuda'):
 assert 'runs/revision_B/step1/' in run
 record=next(x for k in P['head_checkpoints'].values() for x in k if x['run']==run)
 assert sha(W/run/'best.pt')==record['checkpoint_sha256']
 ck=torch.load(W/run/'best.pt',map_location='cpu',weights_only=False);cfg=ck['config'];m=build(cfg['condition'],cfg['dims']);m.load_state_dict(ck['model'],strict=True);m.to(device).eval().requires_grad_(False)
 return m,ck['normalizer']

def probabilities(z):
 z=np.asarray(z,dtype=np.float64);x=np.exp(z-z.max(-1,keepdims=True));return x/x.sum(-1,keepdims=True)

def metrics(p,rows=None):
 df=V if rows is None else V.iloc[rows];y=df[['p_'+x for x in E]].to_numpy(float);u=df.unique_top.astype(bool).to_numpy();pred=p.argmax(1);target=y.argmax(1)
 cm=np.bincount(target[u]*9+pred[u],minlength=81).reshape(9,9);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(9),where=den>0)
 return {'softCE':float(-(y*np.log(np.maximum(p,1e-12))).sum(1).mean()),'macroF1':float(f.mean()),'Brier':float(((p-y)**2).sum(1).mean())}

class Pipeline:
 def __init__(self,kind='R1',batch=16,text_cache=True):
  from transformers import CLIPModel,CLIPProcessor
  self.kind=kind;self.batch=batch;self.text_cache=text_cache;self.run=P['head_checkpoints'][kind][0]['run'];self.net,self.norm=head(self.run)
  self.enc=CLIPModel.from_pretrained(P['encoder'],revision=P['encoder_revision'],local_files_only=True).cuda().eval().requires_grad_(False)
  self.proc=CLIPProcessor.from_pretrained(P['encoder'],revision=P['encoder_revision'],use_fast=False,local_files_only=True)
  self.text={}
  if text_cache and kind=='R1':
   descriptions=[f'{b} overall brightness, {c} luminance contrast, and {x} colors.' for b,c,x in itertools.product(['Low','Moderate','High'],['low','moderate','high'],['predominantly warm','predominantly cool','mixed or neutral'])]
   with torch.inference_mode():
    toks=self.proc(text=descriptions,padding=True,truncation=True,return_tensors='pt').to('cuda');t=self.enc.get_text_features(**toks);t=t/t.norm(dim=-1,keepdim=True)
   self.text={s:t[i].detach() for i,s in enumerate(descriptions)}
 @torch.inference_mode()
 def forward(self,imgs,attention=False):
  px=self.proc(images=imgs,return_tensors='pt')['pixel_values'].cuda();vo=self.enc.vision_model(pixel_values=px);g=self.enc.visual_projection(vo.pooler_output);g=g/g.norm(dim=-1,keepdim=True)
  if self.kind=='PROBE':return self.net(g,None,None,None).float().cpu().numpy(),None
  attrs,descs=zip(*(compute_attributes_and_text(im) for im in imgs));a=torch.from_numpy(normalize(np.stack(attrs),self.norm)).cuda()
  if self.text_cache:t=torch.stack([self.text[s] for s in descs])
  else:
   toks=self.proc(text=list(descs),padding=True,truncation=True,return_tensors='pt').to('cuda');t=self.enc.get_text_features(**toks);t=t/t.norm(dim=-1,keepdim=True)
  raw=vo.last_hidden_state[:,1:,:];patch=F.adaptive_avg_pool2d(raw.transpose(1,2).reshape(-1,1024,24,24),7).flatten(2).transpose(1,2).half().float()
  if attention:z,m=self.net(g,a,t,patch,return_attention=True);return z.float().cpu().numpy(),m.float().cpu().numpy()
  return self.net(g,a,t,patch).float().cpu().numpy(),None
 def predict(self,imgs):
  out=[]
  for s in range(0,len(imgs),self.batch):out.append(self.forward(imgs[s:s+self.batch])[0])
  return probabilities(np.concatenate(out))

class Intervention:
 def __init__(self,image,proc):
  self.image=image;self.rgb=np.asarray(image).astype(np.float32);self.h,self.w=self.rgb.shape[:2]
  # Same PIL resize operation used by the pinned slow CLIP processor.
  resized=proc.image_processor.resize(np.asarray(image),size=proc.image_processor.size,resample=proc.image_processor.resample)
  self.rh,self.rw=resized.shape[:2];self.left=(self.rw-336)//2;self.top=(self.rh-336)//2
  crop=resized[self.top:self.top+336,self.left:self.left+336];self.mean=crop.mean((0,1),dtype=np.float64).astype(np.float32)
  scale=336/min(self.h,self.w);self.blur=np.asarray(image.filter(ImageFilter.GaussianBlur(12/scale))).astype(np.float32)
 def apply(self,keep,method='mean',smooth=False):
  # Source pixels outside the encoder-visible crop stay unchanged, including for retention.
  visible=np.ones((self.rh,self.rw),np.float32);visible[self.top:self.top+336,self.left:self.left+336]=keep
  mapped=np.asarray(Image.fromarray(visible).resize((self.w,self.h),Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST))
  replacement=self.mean[None,None,:] if method=='mean' else self.blur
  out=self.rgb*mapped[:,:,None]+replacement*(1-mapped[:,:,None]);return Image.fromarray(np.uint8(np.clip(np.rint(out),0,255)))

def cells_keep(order,count,retention=False):
 v=np.zeros(49,np.float32);v[np.asarray(order)[:count]]=1
 if not retention:v=1-v
 return np.repeat(np.repeat(v.reshape(7,7),48,0),48,1)

def pilot():
 pipe=Pipeline();idx=SUB.validation_row.to_numpy()[:4];images=[load_image(W/V.iloc[i].image_path) for i in idx]
 z,m=pipe.forward(images,True);saved=np.load(W/pipe.run/'val_logits.npy')[idx];delta=float(np.abs(z-saved).max());print('Original end-to-end vs saved logits max abs:',delta,flush=True)
 assert delta<.001 and np.array_equal(z.argmax(1),saved.argmax(1)),delta
 t=time.perf_counter();p=pipe.predict([Intervention(images[0],pipe.proc).apply(cells_keep(np.arange(49),k),'mean') for k in [5,10,15,24]*4]);torch.cuda.synchronize();sec=time.perf_counter()-t
 result={'max_original_logit_difference':delta,'target_classes_match':True,'modified_images':16,'modified_batch_seconds':sec,'images_per_second':16/sec,'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(),'protocol_sha256':sha(O/'protocol.json')}
 (O/'pilot.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':pilot()
