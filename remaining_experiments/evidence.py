"""Exploratory validation interventions; fixed protocol, no optimizer and no holdout inference."""
from engine import *
from skimage.transform import resize
D=O/'faithfulness';D.mkdir(exist_ok=True)
def main():
 pipe=Pipeline();idx=SUB.validation_row.to_numpy();saved=np.load(W/pipe.run/'val_logits.npy');orig=[];maps=[]
 for s in range(0,len(idx),16):
  zz,mm=pipe.forward([load_image(W/V.iloc[i].image_path) for i in idx[s:s+16]],True);orig.extend(zz);maps.extend(mm)
 orig=np.asarray(orig);maps=np.asarray(maps);diff=float(np.abs(orig-saved[idx]).max());assert diff<.001,diff
 assert np.array_equal(orig.argmax(1),saved[idx].argmax(1)),'Fresh vs cached predicted target differs; stop before interventions'
 np.savez_compressed(D/'originals.npz',rows=idx,logits=orig,maps=maps)
 receipt={'protocol_sha256':sha(O/'protocol.json'),'engine_sha256':sha(O/'engine.py'),'script_sha256':sha(__file__),'checkpoint_sha256':P['evidence_checkpoint']['checkpoint_sha256'],'max_original_saved_logit_difference':diff,'targets_match_saved':True,'rise_reference':'https://github.com/eclique/RISE/blob/master/explanations.py','status':'running'}
 (D/'run.json').write_text(json.dumps(receipt,indent=2)+'\n');start=time.perf_counter()
 for j,i in enumerate(idx):
  dest=D/f'artwork_{i:04d}.npz'
  if dest.exists():continue
  target=int(orig[j].argmax());im=load_image(W/V.iloc[i].image_path);op=Intervention(im,pipe.proc);rng=np.random.default_rng(2752026+int(i));randoms=[rng.permutation(49) for _ in range(10)]
  # Original RISE sampling: Bernoulli coarse grids, reflect bilinear resize, random shifts.
  grids=(rng.random((256,7,7))<.5).astype(np.float32);masks=[];mask7=[]
  for grid in grids:
   x,y=rng.integers(0,48,size=2);mask=resize(grid,(384,384),order=1,mode='reflect',anti_aliasing=False)[x:x+336,y:y+336].astype(np.float32)
   masks.append(mask);mask7.append(mask.reshape(7,48,7,48).mean((1,3)).ravel())
  rp=[]
  for s in range(0,256,16):rp.extend(pipe.predict([op.apply(m,'mean',True) for m in masks[s:s+16]])[:,target])
  sal=np.asarray(rp)@np.asarray(mask7)/(256*.5)
  orders=np.stack([np.argsort(-maps[j,target],kind='stable'),*randoms,np.argsort(-sal,kind='stable')])
  responses=np.empty((2,12,2,4,9),np.float32)
  for fill,method in enumerate(['mean','blur']):
   jobs=[];loc=[]
   for ranking,order in enumerate(orders):
    for direction in range(2):
     for b,k in enumerate(P['mask_counts']):
      jobs.append(op.apply(cells_keep(order,k,bool(direction)),method));loc.append((ranking,direction,b))
      if len(jobs)==16:
       values=pipe.predict(jobs)
       for pos,p in zip(loc,values):responses[(fill,*pos)]=p
       jobs=[];loc=[]
   if jobs:
    for pos,p in zip(loc,pipe.predict(jobs)):responses[(fill,*pos)]=p
  np.savez_compressed(dest,validation_row=i,painting=V.iloc[i].painting,target=target,original_probability=probabilities(orig[j:j+1])[0],responses=responses,rankings=orders,ecsa_map=maps[j,target],rise_map=sal,rise_response=np.asarray(rp),rise_grid=grids,seed=2752026+int(i))
  print(f'{j+1}/200; row={i}; elapsed={time.perf_counter()-start:.1f}s; original_target={E[target]}',flush=True)
 receipt.update(status='complete',artworks=200,wall_seconds=time.perf_counter()-start,images_encoded=89800)
 (D/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
