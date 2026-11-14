"""Prespecified artwork sampling and resumable, artwork-level evidence studies."""
from collections import Counter
from collections.abc import Callable
from pathlib import Path
import hashlib
import json
import numpy as np

METRICS=('actual_area','actual_image_area','original_probability','deletion_probability','retention_probability','deletion_drop','random_deletion_drop','deletion_advantage','retention_advantage')


def select_artworks(rows: list[dict],count: int,seed: int) -> list[dict]:
    """Select by seeded artwork identity, without labels, scores, or explanations."""
    ordered=sorted(rows,key=lambda row:row['painting'])
    if len({row['painting'] for row in ordered})!=len(ordered):raise ValueError('Repeated artwork IDs')
    if not 1<=count<=len(ordered):raise ValueError('Sample size outside available artworks')
    indices=sorted(np.random.default_rng(seed).choice(len(ordered),count,replace=False).tolist())
    return [ordered[index] for index in indices]


def atomic_json(path: Path,value: dict) -> None:
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def aggregate_reports(reports: list[dict],bootstrap: int=1000,seed: int=42) -> dict:
    """Bootstrap artworks, never treating random perturbations as independent images."""
    if not reports or bootstrap<1:raise ValueError('Need completed artworks and positive bootstrap count')
    if len({r['painting'] for r in reports})!=len(reports):raise ValueError('Repeated artwork reports')
    groups={};expected=None
    for report in reports:
        if not 0<=report['class_index']<9:raise ValueError('Invalid predicted emotion class')
        keys=[(r['method'],float(r['budget'])) for r in report['rows']]
        if len(keys)!=len(set(keys)):raise ValueError('Repeated intervention group for one artwork')
        if expected is not None and set(keys)!=expected:raise ValueError('Inconsistent intervention groups across artworks')
        expected=set(keys)
        for row,key in zip(report['rows'],keys):
            values=np.array([row[name] for name in METRICS],dtype=float)
            if not np.isfinite(values).all():raise ValueError('Nonfinite intervention result')
            groups.setdefault(key,[]).append(values)
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(reports),(bootstrap,len(reports)));rows=[]
    for (method,budget),values in sorted(groups.items()):
        matrix=np.stack(values);row={'method':method,'budget':budget,'artworks':len(reports),**dict(zip(METRICS,map(float,matrix.mean(0))))}
        intervals={}
        for name in ('deletion_advantage','retention_advantage'):
            column=matrix[:,METRICS.index(name)];samples=column[indices].mean(1)
            intervals[name]={'ci95':np.quantile(samples,[.025,.975]).tolist(),'artwork_sample_std':float(column.std(ddof=1)) if len(column)>1 else None}
        row['bootstrap']=intervals;rows.append(row)
    return {'artworks':len(reports),'predicted_class_counts':dict(Counter(str(r['class_index']) for r in reports)),'rows':rows,'bootstrap_repeats':bootstrap,'bootstrap_seed':seed,'bootstrap_unit':'artwork','limitations':'Exploratory perturbation sensitivity, conditional on the selected checkpoint and predicted classes; not proof of semantic correctness.'}


def run_study(rows: list[dict],settings: dict,output: Path,evaluate_one: Callable,resume: bool=False,bootstrap: int=1000) -> dict:
    """Persist each artwork atomically and resume only an identical fixed study."""
    study={'version':1,'settings':settings,'artwork_ids':[r['painting'] for r in rows]}
    serialized=json.dumps(study,sort_keys=True,allow_nan=False);fingerprint=hashlib.sha256(serialized.encode()).hexdigest()
    if resume:
        if json.loads((output/'study.json').read_text())!=study:raise ValueError('Study selection or configuration changed')
    else:
        output.mkdir(parents=True,exist_ok=False);(output/'artworks').mkdir();atomic_json(output/'study.json',study)
    reports=[]
    try:
        for row in rows:
            painting=row['painting'];name=hashlib.sha256(painting.encode()).hexdigest()+'.json';path=output/'artworks'/name
            if path.exists():
                report=json.loads(path.read_text())
                if report.get('painting')!=painting or report.get('study_sha256')!=fingerprint:raise ValueError('Completed artwork provenance changed')
            else:
                report={**evaluate_one(row),'painting':painting,'study_sha256':fingerprint};atomic_json(path,report)
            reports.append(report)
            atomic_json(output/'status.json',{'status':'running','completed_artworks':len(reports),'total_artworks':len(rows)})
    except (Exception,KeyboardInterrupt) as exc:
        atomic_json(output/'status.json',{'status':'interrupted','completed_artworks':len(reports),'total_artworks':len(rows),'error_type':type(exc).__name__,'message':str(exc)})
        raise
    summary=aggregate_reports(reports,bootstrap,settings.get('seed',42));summary.update(scope=settings.get('scope','research'),study_sha256=fingerprint)
    atomic_json(output/'faithfulness.json',summary);atomic_json(output/'status.json',{'status':'complete','completed_artworks':len(reports),'total_artworks':len(rows)})
    return summary
