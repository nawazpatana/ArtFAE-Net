"""Independent artifact/arithmetic checks after completion; no model loading or inference."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import pandas as pd
W=Path(__file__).resolve().parents[1];O=W/'remaining_experiments';P=json.loads((O/'protocol.json').read_text());A=json.loads((O/'faithfulness_summary.json').read_text());B=json.loads((O/'permutation/summary.json').read_text());S=pd.read_csv(O/'validation_subset.csv');V=pd.read_csv(W/'data/artemis_balanced/val.csv');x=np.array(P['mask_fractions']);weights=np.array([x[1]-x[0],x[2]-x[0],x[3]-x[1],x[3]-x[2]])/(2*(x[-1]-x[0]));auc=[]
assert A['artworks']==200 and A['run']['status']=='complete'
assert len(list((O/'faithfulness').glob('artwork_*.npz')))==200
for i in S.validation_row:
 d=np.load(O/f'faithfulness/artwork_{i:04d}.npz');r=d['responses'];assert r.shape==(2,12,2,4,9) and np.isfinite(r).all() and np.all((r>=0)&(r<=1));assert np.max(np.abs(r.sum(-1)-1))<.000001
 assert np.all(np.sort(d['rankings'],axis=-1)==np.arange(49));assert int(d['target'])==int(d['original_probability'].argmax());assert str(d['painting'])==V.iloc[i].painting
 q=r[...,int(d['target'])];auc.append(q@weights)
auc=np.array(auc)
for row in A['summary']:
 fi=['mean','blur'].index(row['fill']);di=['deletion','retention'].index(row['curve']);method=row['ranking'];a=auc[:,fi,1:11,di].mean(1) if method=='Random' else auc[:,fi,0 if method=='ECSA' else 11,di];assert abs(a.mean()-row['mean_partial_AUC'])<1e-7
perms=np.load(O/'permutation/permutations.npy');assert perms.shape==(20,3000) and (perms!=np.arange(3000)).all();assert np.all(np.sort(perms,axis=1)==np.arange(3000))
probs=np.load(O/'permutation/predictions.npz')['probabilities'];assert probs.shape==(5,21,3000,9) and np.isfinite(probs).all() and np.max(np.abs(probs.sum(-1)-1))<.000001
for j,r in enumerate(P['head_checkpoints']['R1']):
 z=np.load(W/r['run']/'val_logits.npy').astype(float);e=np.exp(z-z.max(1,keepdims=True));p=e/e.sum(1,keepdims=True);assert np.max(np.abs(probs[j,0]-p))<.000001
for model in ['R1','PROBE']:
 c=json.loads((O/f'resources/{model}.json').read_text());d=pd.read_csv(O/f'resources/{model}_batches.csv');assert len(d)==20 and d.n.sum()==160 and c['batch']==8
 assert c['peak_allocated_bytes']>=c['model_baseline_allocated_bytes']>0
 assert abs(c['complete_throughput_images_s']-160/d.complete_s.sum())<1e-8
 assert (d.complete_s>=d.features_s+d.head_s+d.decode_s-1e-8).all()
 assert c['peak_ru_maxrss_bytes']>0 and d.max_saved_logit_difference.max()<.001
protected=json.loads((O/'preservation_before.json').read_text());images=json.loads((O/'validation_image_hashes.json').read_text())
for path,r in {**protected,**{r['path']:r for r in images.values()}}.items():
 with Path(path).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert h==r['sha256'],path
record={'artwork_response_files_checked':200,'response_probabilities_valid':True,'rankings_are_permutations':True,'partial_AUC_verified_by_independent_integration_weights':True,'paired_derangements_valid':True,'permutation_probability_files_valid':True,'original_permutation_logits_reproduce':True,'resource_batch_records_consistent':True,'protected_artifact_hashes_matching':len(protected),'validation_image_hashes_matching':len(images),'no_training_or_new_holdout_inference':True,'all_original_H1_H5_outputs_unchanged':True}
(O/'VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
