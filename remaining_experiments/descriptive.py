"""Task D: descriptive arithmetic on saved holdout logits; no model or image is loaded."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
W=Path(__file__).resolve().parents[1];O=W/'remaining_experiments';E=['amusement','awe','contentment','excitement','anger','disgust','fear','sadness','something_else']
h=pd.read_csv(W/'runs/revision_B/fresh_holdout/holdout.csv');y=h[['p_'+e for e in E]].to_numpy(float);u=h.unique_top.astype(bool).to_numpy();ent=-(y*np.log(np.maximum(y,1e-12))).sum(1)
groups=[('population','all',np.ones(len(h),bool)),('top','unique',u),('top','tied',~u),('artist','seen_in_train',h.artist_in_43k_train.astype(bool).to_numpy()),('artist','absent_from_train',~h.artist_in_43k_train.astype(bool).to_numpy())]
for style,ids in h.groupby('art_style').groups.items():
 mask=np.zeros(len(h),bool);mask[np.asarray(ids)]=True;groups.append(('style',str(style),mask))
rows=[];inputs={}
for model,prefix in [('R1','R1_R1_s'),('PROBE','PROBE_probe100_lr1e-3_s')]:
 for seed in range(42,47):
  f=W/f'runs/final_eval_v3/{prefix}{seed}_holdout_logits.npy';inputs[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();z=np.load(f);z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
  ce=-(y*np.log(np.maximum(p,1e-12))).sum(1);br=((p-y)**2).sum(1);pred=p.argmax(1);target=y.argmax(1)
  for kind,label,mask in groups:
   elig=mask&u;cm=np.bincount(target[elig]*9+pred[elig],minlength=81).reshape(9,9);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(9),where=den>0).mean() if elig.any() else np.nan
   rows.append(dict(model=model,seed=seed,group=kind,label=label,n=int(mask.sum()),unique_top=int(elig.sum()),tied=int((mask&~u).sum()),entropy=float(ent[mask].mean()),softCE=float(ce[mask].mean()),KL=float((ce-ent)[mask].mean()),Brier=float(br[mask].mean()),macroF1=f1,display_supported=bool(mask.sum()>=100)))
d=pd.DataFrame(rows);d.to_csv(O/'descriptive_per_seed.csv',index=False)
summary=d.groupby(['model','group','label'],sort=False).agg(n=('n','first'),unique_top=('unique_top','first'),tied=('tied','first'),entropy=('entropy','first'),softCE=('softCE','mean'),softCE_seed_SD=('softCE','std'),KL=('KL','mean'),Brier=('Brier','mean'),macroF1=('macroF1','mean'),macroF1_seed_SD=('macroF1','std'),display_supported=('display_supported','first')).reset_index();summary.to_csv(O/'descriptive_summary.csv',index=False)
J=json.loads((W/'runs/final_eval_v3/results.json').read_text())
for model in ['R1','PROBE']:
 value=summary[(summary.model==model)&(summary.label=='all')].iloc[0].softCE;assert abs(value-J['primary']['per_seed_mean (PRIMARY contrast)']['models'][model]['metrics']['softCE'])<.00000051
report=['# Saved-result generalization analysis (descriptive only)','','No model was loaded. Existing holdout logits and existing style/artist membership columns were read; no new holdout predictions, metadata mapping or split was generated. Seed means match the locked main CE values. Groups are not new confirmatory hypotheses. Artwork styles and artist overlap differ in annotation entropy and composition, so metric gaps are not causal effects of style or artist novelty. Tiny groups are saved but not promoted as reliable estimates.','','| Model | Group | N | Unique-top | CE | KL | Macro-F1 |','|---|---|---:|---:|---:|---:|---:|']
for r in summary.itertuples():
 if r.group!='style' or (r.n>=500 and r.model=='R1'):report.append(f'| {r.model} | {r.group}: {r.label} | {r.n} | {r.unique_top} | {r.softCE:.4f} | {r.KL:.4f} | {r.macroF1:.4f} |')
report+=['','Complete per-style and per-seed tables are in the CSVs. The historical 32,868-training-artist experiment remains a separate protocol, not evidence newly obtained on the main training setup. No external collection or corruption inference was conducted.'];(O/'GENERALIZATION_RESULTS.md').write_text('\n'.join(report).replace('| nan |','| not applicable |')+'\n');(O/'descriptive_input_hashes.json').write_text(json.dumps(inputs,indent=2)+'\n');print('Saved-only descriptive groups:',len(summary))
