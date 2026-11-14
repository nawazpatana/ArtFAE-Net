"""Exportable Matplotlib figures for completed experiments only."""
from pathlib import Path
import json
import numpy as np


def render_run(directory: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics_path=directory/'validation_metrics.json'
    if not metrics_path.exists():return
    metrics=json.loads(metrics_path.read_text());labels=metrics['classes'];cm=np.asarray(metrics['confusion_matrix'])
    import csv
    with (directory/'per_class_metrics.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['emotion','precision','recall','f1','support'])
        writer.writerows([label,metrics['precision'][i],metrics['recall'][i],metrics['f1'][i],metrics['support'][i]] for i,label in enumerate(labels))
    with (directory/'confusion_matrix.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['true/predicted',*labels]);writer.writerows([label,*row] for label,row in zip(labels,cm.tolist()))
    fig,ax=plt.subplots(figsize=(9,8));picture=ax.imshow(cm,cmap='Blues');fig.colorbar(picture,ax=ax,label='Artwork count');ax.set(xticks=range(9),yticks=range(9),xticklabels=labels,yticklabels=labels,xlabel='Predicted emotion',ylabel='Human unique-top emotion',title=f'Validation confusion matrix (excluded ties: {metrics["excluded_ties"]})');plt.setp(ax.get_xticklabels(),rotation=55,ha='right');fig.tight_layout();fig.savefig(directory/'confusion_matrix.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(labels,metrics['f1']);ax.set(ylabel='F1',ylim=(0,1),title='Validation per-class F1');plt.setp(ax.get_xticklabels(),rotation=45,ha='right');fig.tight_layout();fig.savefig(directory/'per_class_f1.png',dpi=180);plt.close(fig)
    history_path=directory/'history.json'
    if history_path.exists():
        history=json.loads(history_path.read_text())
        if history:
            fig,axes=plt.subplots(1,2,figsize=(10,4));epochs=[r['epoch'] for r in history];axes[0].plot(epochs,[r['train_soft_ce'] for r in history],label='Train');axes[0].plot(epochs,[r['soft_ce'] for r in history],label='Validation');axes[0].legend();axes[0].set(xlabel='Epoch',ylabel='Soft cross-entropy');axes[1].plot(epochs,[r['macro_f1'] for r in history]);axes[1].set(xlabel='Epoch',ylabel='Validation macro-F1');fig.tight_layout();fig.savefig(directory/'learning_curves.png',dpi=180);plt.close(fig)


def render_evidence(directory: Path) -> None:
    """Plot intervention sensitivity with matched random controls, without causal claims."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((directory/'faithfulness.json').read_text())
    rows=report['rows'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method in sorted({r['method'] for r in rows}):
        selected=sorted((r for r in rows if r['method']==method),key=lambda r:r['budget'])
        x=[r['actual_area'] for r in selected]
        axes[0].plot(x,[r['deletion_drop'] for r in selected],marker='o',label=f'{method}: top regions')
        axes[0].plot(x,[r['random_deletion_drop'] for r in selected],linestyle='--',label=f'{method}: random')
        axes[1].plot(x,[r['retention_advantage'] for r in selected],marker='o',label=method)
    axes[0].set(xlabel='Selected crop area',ylabel='Class probability drop',title='Deletion sensitivity')
    axes[1].set(xlabel='Selected crop area',ylabel='Top minus random retention probability',title='Retention control advantage')
    for ax in axes:ax.axhline(0,color='gray',linewidth=.7);ax.legend(fontsize=8)
    fig.suptitle('Smoke implementation check' if 'smoke' in report.get('scope','') else 'Perturbation sensitivity; not proof of semantic correctness',fontsize=10)
    fig.tight_layout();fig.savefig(directory/'faithfulness_curves.png',dpi=180);plt.close(fig)
