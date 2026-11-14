#!/usr/bin/env python3
"""Consolidate completed validation metrics; explicitly mark uncompleted variants."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from artfae.evaluation.plots import render_run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--output',type=Path,required=True);p.add_argument('--plots',action='store_true');p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--data-dir',type=Path,default=Path('data'));args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[];groups={}
    for file in sorted(args.results.rglob('validation_metrics.json')):
        if any('smoke' in part for part in file.relative_to(args.results).parts): continue
        metrics=json.loads(file.read_text());config_path=file.parent/'config.json';config=json.loads(config_path.read_text()) if config_path.exists() else {}
        if config.get('scope')=='smoke':continue
        variant=config.get('variant','B2' if file.parent.name.startswith('clip_validation') else 'B1' if 'freeze_backbone' in config else 'unknown')
        row={'run':str(file.parent.relative_to(args.results)),'variant':variant,'status':'validation completed','seed':config.get('seed',42),'macro_f1':metrics['macro_f1'],'accuracy':metrics['accuracy'],'soft_ce':metrics['soft_ce'],'test_status':'not evaluated'};rows.append(row)
        # A tuning trial is a distinct recipe even when architecture is identical.
        # Only seed and display-only notebook labels may vary within a seed summary.
        recipe={key:value for key,value in config.items() if key not in ('seed','notebook_label','notebook_trial')}
        key=json.dumps({'variant':variant,'architecture':config.get('architecture',{}),'training':recipe},sort_keys=True);groups.setdefault(key,[]).append(row)
        if args.plots:render_run(file.parent)
    for variant in ('B1','B2','B3','B4','B5','M1','M2'):
        if not any(r['variant']==variant for r in rows):rows.append({'run':'','variant':variant,'status':'pending','seed':'','macro_f1':'','accuracy':'','soft_ce':'','test_status':'not evaluated'})
    with (args.output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import numpy as np
    summaries=[]
    for key,runs in groups.items():
        by_seed={}
        for run in runs:by_seed.setdefault(run['seed'],run)
        values=[r['macro_f1'] for r in by_seed.values() if r['macro_f1'] is not None];summaries.append({'group':json.loads(key),'runs':[r['run'] for r in runs],'selected_runs_by_seed':{str(seed):run['run'] for seed,run in by_seed.items()},'macro_f1_mean':float(np.mean(values)) if values else None,'macro_f1_sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None,'independent_seed_count':len(set(r['seed'] for r in runs)),'warning':'Repeated runs of one seed are not independent seeds; the first lexicographic run per recipe/seed is used.'})
    comparisons=[]
    complete=[r for r in rows if r['status']=='validation completed']
    if args.bootstrap>0 and len(complete)>1:
        from artfae.datasets.manifests import read_manifest
        from artfae.evaluation.statistics import paired_bootstrap,macro_f1_soft_targets
        targets={r['painting']:r['target'] for r in read_manifest(args.data_dir/'val_artworks.csv',args.data_dir)}
        prediction_sets={}
        for row in complete:
            with (args.results/row['run']/'predictions.csv').open() as f:
                from artfae import EMOTIONS
                prediction_sets[row['run']]={r['painting']:np.array([float(r[e]) for e in EMOTIONS]) for r in csv.DictReader(f)}
        reference=next((r for r in complete if r['variant']=='B2'),None)
        if reference:
            ids=sorted(targets);a=prediction_sets[reference['run']]
            for row in complete:
                if row['run']==reference['run']:continue
                b=prediction_sets[row['run']]
                if set(a)!=set(targets) or set(b)!=set(targets):raise ValueError('Prediction IDs do not match full validation set')
                result=paired_bootstrap(np.stack([targets[i] for i in ids]),np.stack([b[i] for i in ids]),np.stack([a[i] for i in ids]),macro_f1_soft_targets,repeats=args.bootstrap)
                comparisons.append({'first':row['run'],'second':reference['run'],**result,'scope':'validation; exploratory model comparison, not a final-test significance claim'})
    (args.output/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    (args.output/'seed_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
