"""Task B: fixed-parameter numerical/text pair permutations on cached validation features only."""
from engine import *
D=O/'permutation';D.mkdir(exist_ok=True)
def main():
 rng=np.random.default_rng(2752026);perms=[]
 for _ in range(20):
  while True:
   ix=rng.permutation(len(V))
   if np.all(ix!=np.arange(len(V))):break
  perms.append(ix)
 perms=np.stack(perms);np.save(D/'permutations.npy',perms)
 g=torch.from_numpy(np.load(W/'features_l14/val/global.npy')).cuda();t=torch.from_numpy(np.load(W/'features_l14/val/attr_text.npy')).cuda();patch=torch.from_numpy(np.load(W/'features_l14/val/patches.npy').astype(np.float32)).cuda();attrs=np.load(W/'features_l14/val/attributes.npy');probs=[];rows=[];repros=[]
 with torch.inference_mode():
  for seed,record in zip(range(42,47),P['head_checkpoints']['R1']):
   net,norm=head(record['run']);a=torch.from_numpy(normalize(attrs,norm)).cuda();seedprobs=[]
   for rep,ix in enumerate([np.arange(len(V)),*perms]):
    ix_t=torch.as_tensor(ix,device='cuda');z=[]
    for b in range(0,len(V),128):z.append(net(g[b:b+128],a[ix_t[b:b+128]],t[ix_t[b:b+128]],patch[b:b+128]).cpu().numpy())
    z=np.concatenate(z);p=probabilities(z);seedprobs.append(p.astype(np.float32))
    if rep==0:
     err=float(np.abs(z-np.load(W/record['run']/'val_logits.npy')).max());assert err<.0001,err;repros.append(err);base=p
    rows.append(dict(seed=seed,permutation=rep,**metrics(p),top1_flip_fraction=float((p.argmax(1)!=base.argmax(1)).mean()),mean_total_variation=float((np.abs(p-base).sum(1)/2).mean())))
   probs.append(seedprobs);print('Permutation seed complete:',seed,flush=True)
 np.savez_compressed(D/'predictions.npz',probabilities=np.asarray(probs),validation_rows=np.arange(len(V)))
 pd.DataFrame(rows).to_csv(D/'metrics.csv',index=False);d=pd.DataFrame(rows);baseline=d[d.permutation==0];shuffled=d[d.permutation>0]
 y=V[['p_'+e for e in E]].to_numpy(float);allp=np.asarray(probs,dtype=float);ce=-(y[None,None,:,:]*np.log(np.maximum(allp,1e-12))).sum(-1);delta=(ce[:,1:]-ce[:,0:1]).mean((0,1));rng=np.random.default_rng(2752026);boot=np.array([delta[rng.integers(0,len(V),len(V))].mean() for _ in range(2000)])
 from artfae.features.aesthetic import describe
 desc=np.asarray([describe(a) for a in attrs])
 result={'same_template_fraction_after_pair_permutation':float((desc[perms]==desc[None,:]).mean()),'scope':'EXPLORATORY development validation; five fixed checkpoints, 20 paired derangements; no encoder, training or holdout inference','baseline':{m:float(baseline[m].mean()) for m in ['softCE','macroF1']},'paired_permuted':{m:float(shuffled[m].mean()) for m in ['softCE','macroF1','top1_flip_fraction','mean_total_variation']},'delta_softCE':float(delta.mean()),'delta_softCE_paired_artwork_CI95':np.quantile(boot,[.025,.975]).tolist(),'delta_macroF1':float(shuffled.macroF1.mean()-baseline.macroF1.mean()),'max_original_saved_logit_difference':max(repros),'CI_scope':'Conditional on these five checkpoints and 20 fixed permutations; not training-seed or permutation-population uncertainty','protocol_sha256':sha(O/'protocol.json')}
 (D/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
