"""Aggregate completed validation studies; no model inference. Paired artwork bootstrap and plots."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def markdown(frame):
 def fmt(v):
  if pd.isna(v):return "--"
  return f"{v:.4f}" if isinstance(v,float) else str(v)
 return "| "+" | ".join(map(str,frame.columns))+" |\n|"+"|".join(["---"]*len(frame.columns))+"|\n"+"\n".join("| "+" | ".join(fmt(v).replace("|","/") for v in row)+" |" for row in frame.itertuples(index=False,name=None))

W=Path(__file__).resolve().parents[1];O=W/'remaining_experiments';D=O/'faithfulness';P=json.loads((O/'protocol.json').read_text())

def main():
 run=json.loads((D/'run.json').read_text());assert run['status']=='complete'
 subset=pd.read_csv(O/'validation_subset.csv');v=pd.read_csv(W/'data/artemis_balanced/val.csv');x=np.array(P['mask_fractions']);curves=[];aucs=[];rows=[];digests={}
 for i in subset.validation_row:
  f=D/f'artwork_{i:04d}.npz';d=np.load(f);assert int(d['validation_row'])==i;target=int(d['target']);p=d['responses'][...,target];a=np.trapezoid(p,x,axis=-1)/(x[-1]-x[0]);curves.append(p);aucs.append(a);digests[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
  for fi,fill in enumerate(['mean','blur']):
   for rank,label in [(0,'ECSA'),(11,'RISE'),(None,'Random')]:
    for direction,curve in enumerate(['deletion','retention']):
     response=p[fi,1:11,direction].mean(0) if rank is None else p[fi,rank,direction]
     value=float(a[fi,1:11,direction].mean()) if rank is None else float(a[fi,rank,direction])
     rows.append({'validation_row':i,'painting':str(d['painting']),'target_class':target,'fill':fill,'ranking':label,'curve':curve,'partial_AUC':value,**{f'p_{k}_of_49':float(q) for k,q in zip(P['mask_counts'],response)}})
 curves=np.array(curves);aucs=np.array(aucs);pd.DataFrame(rows).to_csv(O/'faithfulness_per_artwork.csv',index=False)
 rng=np.random.default_rng(2752026);draws=rng.integers(0,len(subset),size=(2000,len(subset)));summary=[];contrasts=[]
 for fi,fill in enumerate(['mean','blur']):
  for di,curve in enumerate(['deletion','retention']):
   values={'ECSA':aucs[:,fi,0,di],'RISE':aucs[:,fi,11,di],'Random':aucs[:,fi,1:11,di].mean(1)}
   for label,a in values.items():
    ci=np.quantile(a[draws].mean(1),[.025,.975]);summary.append({'fill':fill,'curve':curve,'ranking':label,'mean_partial_AUC':float(a.mean()),'CI95_low':float(ci[0]),'CI95_high':float(ci[1])})
   for label in ['ECSA','RISE']:
    adv=(values['Random']-values[label]) if di==0 else (values[label]-values['Random']);ci=np.quantile(adv[draws].mean(1),[.025,.975]);contrasts.append({'fill':fill,'curve':curve,'contrast':label+'_advantage_over_random','mean':float(adv.mean()),'CI95_low':float(ci[0]),'CI95_high':float(ci[1])})
   adv=(values['RISE']-values['ECSA']) if di==0 else (values['ECSA']-values['RISE']);ci=np.quantile(adv[draws].mean(1),[.025,.975]);contrasts.append({'fill':fill,'curve':curve,'contrast':'ECSA_advantage_over_RISE','mean':float(adv.mean()),'CI95_low':float(ci[0]),'CI95_high':float(ci[1])})
 pd.DataFrame(summary).to_csv(O/'faithfulness_summary.csv',index=False);pd.DataFrame(contrasts).to_csv(O/'faithfulness_paired_contrasts.csv',index=False)
 c=pd.read_csv(W/'runs/revision_B/duplicates/classification.csv');dup=set(c.loc[(c.eval_side_splits=='val')&(c.classification!='different'),'eval_side']);selected=set(subset.painting);known=sorted(selected&dup)
 run['wall_seconds_scope']='Intervention loop only; excludes model loading and the original 200 encodings.'
 report={'scope':'EXPLORATORY development validation, fixed R1 seed42, no human-region or causal claim','artworks':200,'unique_top':int(v.iloc[subset.validation_row].unique_top.sum()),'known_training_reproductions_in_subset':known,'known_reproduction_count':len(known),'original_predicted_class_counts':{str(k):int(n) for k,n in pd.DataFrame(rows).drop_duplicates('validation_row').target_class.value_counts().items()},'partial_auc_convention':P['auc'],'summary':summary,'contrasts':contrasts,'bootstrap':'2000 paired artwork draws, ten random rankings averaged within each artwork. Pointwise unadjusted intervals, conditional on checkpoint/subset/mask draws. RISE finite-mask uncertainty not separately resampled. Not H1–H5.','checkpoint_sha256':run['checkpoint_sha256'],'protocol_sha256':run['protocol_sha256'],'run':run,'no_holdout_inference':True}
 (O/'faithfulness_summary.json').write_text(json.dumps(report,indent=2)+'\n');(O/'faithfulness_artifact_hashes.json').write_text(json.dumps(digests,indent=2)+'\n')
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42});fig,axes=plt.subplots(2,2,figsize=(7,4.2),sharex=True)
 for fi,fill in enumerate(['Mean fill','Blur']):
  for di,direction in enumerate(['Deletion: lower better','Retention: higher better']):
   ax=axes[di,fi]
   for rank,label,color,style in [(0,'ECSA','#235e93','-'),(11,'RISE-style','#96502f','-'),(None,'Matched random','#666666','--')]:
    vv=curves[:,fi,1:11,di].mean(1) if rank is None else curves[:,fi,rank,di]
    ax.plot(x,vv.mean(0),style,marker='o',markersize=3,color=color,label=label)
   ax.set_title(fill+' | '+direction,fontsize=9);ax.grid(alpha=.2);ax.set_xlabel('Observed masking fraction');ax.set_ylabel('Fixed-target probability')
 axes[0,1].legend(fontsize=7);fig.tight_layout();fig.savefig(O/'faithfulness_curves.pdf',bbox_inches='tight');fig.savefig(O/'faithfulness_curves.png',dpi=180,bbox_inches='tight');plt.close(fig)
 (O/'FAITHFULNESS_RESULTS.md').write_text('# Exploratory validation perturbation results\n\n'+markdown(pd.DataFrame(summary))+'\n\nPaired advantages (positive favours stated ranking):\n\n'+markdown(pd.DataFrame(contrasts))+'\n\nNo endpoints were measured or imputed. This is raw-probability partial AUC over [5/49,24/49], normalized by range width. Intervals are pointwise, paired-artwork and conditional on one fixed checkpoint, selected validation images and generated mask rankings. Whole-pipeline features are recomputed; a perturbation result does not isolate causal ECSA or human-semantic evidence. RISE-style uses 256 masks and is not claimed converged.\n')
 print(pd.DataFrame(contrasts).to_string(index=False),flush=True)
if __name__=='__main__':main()
