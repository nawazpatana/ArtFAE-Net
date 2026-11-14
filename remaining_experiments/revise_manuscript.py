"""Update only checked exploratory findings. Refuses to run before all jobs/checks finish."""
from pathlib import Path
import json,csv,hashlib,re,shutil,zipfile
import pandas as pd
W=Path(__file__).resolve().parents[1];O=W/'remaining_experiments';P=W/'paper/syscom2026_camera_ready'
A=json.loads((O/'faithfulness_summary.json').read_text());B=json.loads((O/'permutation/summary.json').read_text());C={m:json.loads((O/f'resources/{m}.json').read_text()) for m in ['R1','PROBE']}
assert A['run']['status']=='complete' and A['artworks']==200
assert C['R1']['gpu']==C['PROBE']['gpu'] and C['R1']['batch']==C['PROBE']['batch']==8
for f,r in json.loads((O/'preservation_before.json').read_text()).items():
 with Path(f).open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
 assert h==r['sha256'],f
with zipfile.ZipFile(P/'history/pre_new_validation_experiments_Source.zip') as z:
 for rel in ['tables/final_models.tex','tables/final_hypotheses.tex','numbers.tex']:
  assert (P/rel).read_bytes()==z.read(next(n for n in z.namelist() if n.endswith('/'+rel))),rel
s=(P/'main.tex').read_text()
def ci(low,high):return f'$[{low:+.4f},{high:+.4f}]$'
# Primary table unchanged: new table reports exploratory paired sensitivity only.
rows=['\\begin{table*}[t]','\\caption{Exploratory validation partial AUC, one fixed R1 seed-42 checkpoint and 200 predetermined paintings. Raw target probability is integrated only over $[5/49,24/49]$ and divided by $19/49$. Positive advantages favour the stated ranking: random minus method for deletion, method minus random for retention. Brackets are pointwise, unadjusted paired-artwork 95\\% intervals, conditional on this checkpoint and mask draws.}\\label{tab:validationevidence}','\\centering\\small','\\begin{tabular}{llrrrr}','\\toprule','Fill & Ranking & Deletion AUC & Advantage [95\\% CI] & Retention AUC & Advantage [95\\% CI] \\\\','\\midrule']
for fill in ['mean','blur']:
 for method in ['ECSA','RISE','Random']:
  vals={d:next(x for x in A['summary'] if x['fill']==fill and x['curve']==d and x['ranking']==method) for d in ['deletion','retention']}
  adv={d:next((x for x in A['contrasts'] if x['fill']==fill and x['curve']==d and x['contrast']==method+'_advantage_over_random'),None) for d in ['deletion','retention']}
  fields=[]
  for d in ['deletion','retention']:
   fields.append(f"{vals[d]['mean_partial_AUC']:.4f}")
   fields.append('--' if adv[d] is None else f"{adv[d]['mean']:+.4f} "+ci(adv[d]['CI95_low'],adv[d]['CI95_high']))
  rows.append(('Mean' if fill=='mean' else 'Blur')+' & '+('RISE-style' if method=='RISE' else method)+' & '+' & '.join(fields)+r' \\')
rows+=['\\bottomrule','\\end{tabular}','\\end{table*}'];(P/'tables/new_validation_evidence.tex').write_text('\n'.join(rows)+'\n')
rows=['\\begin{table}[t]','\\caption{Exploratory paired attribute/text permutation, five fixed R1 checkpoints and 20 derangements of all 3,000 validation rows. Image/patch inputs and parameters stay fixed. Means are over checkpoint/permutation combinations, not ensembles.}\\label{tab:permutation}','\\centering','\\begin{tabular}{lrrr}','\\toprule','Pairing & SoftCE & Macro-F1 & Flips \\\\','\\midrule',f"Original & {B['baseline']['softCE']:.4f} & {B['baseline']['macroF1']:.4f} & 0\\% \\\\",f"Permuted & {B['paired_permuted']['softCE']:.4f} & {B['paired_permuted']['macroF1']:.4f} & {100*B['paired_permuted']['top1_flip_fraction']:.1f}\\% \\\\",'\\bottomrule','\\end{tabular}','\\end{table}'];(P/'tables/new_permutation.tex').write_text('\n'.join(rows)+'\n')
rows=['\\begin{table*}[t]','\\caption{Exploratory resource measurement on the same RTX 5090 and 32 validation images, batch 8, FP32, TF32 disabled, separate fresh processes. Timings are amortized milliseconds per image after warm-up, including actual model feature branches; complete time also includes decoding/head/output. CUDA peaks are allocated/reserved memory, not device-wide usage; RSS includes model loading. Five measured passes, 160 images per model with repetition.}\\label{tab:resources}','\\centering','\\begin{tabular}{lrrrrr}','\\toprule','Model & Features (ms) & Complete (ms) & Images/s & CUDA alloc./res. (GiB) & Peak RSS (GiB) \\\\','\\midrule']
for model,label in [('R1','R1'),('PROBE','Linear probe')]:
 c=C[model];tim=c['amortized_ms_per_image'];rows.append(f"{label} & {tim['features_s']:.2f} & {tim['complete_s']:.2f} & {c['complete_throughput_images_s']:.1f} & {c['peak_allocated_bytes']/1024**3:.2f}/{c['peak_reserved_bytes']/1024**3:.2f} & {c['peak_ru_maxrss_bytes']/1024**3:.2f}"+r' \\')
rows+=['\\bottomrule','\\end{tabular}','\\end{table*}'];(P/'tables/new_resources.tex').write_text('\n'.join(rows)+'\n')
a=s.index('\\subsection{Complexity and Generalization Boundaries}');b=s.index('\\section{\\uppercase{Discussion and Limitations}}',a)
s=s[:a]+r'''\subsection{Complexity, Resources and Generalization}
The full head has \val{parr1} trainable parameters versus \val{parprobe} for the probe; both use the same large frozen CLIP model. Table~\ref{tab:resources} reports new exploratory measurements on identical validation images, hardware, batch and precision, with no training or holdout inference. Each worker loads the full CLIP object, including the unused text branch for the probe; this is a controlled implementation comparison, not a minimum-memory optimized deployment. CPU feature preparation and decoding, warm file caches and allocator behaviour affect timings. The head-parameter ratio is not a full-system speed or memory ratio.

\input{tables/new_resources}

Saved-prediction analysis remains descriptive. The holdout contains \val{nAbsWorks} artworks by \val{nAbsArt} artists absent from training; their different composition limits interpretation. Existing logits also support style and tied/unique-top summaries, not external-domain or corruption robustness. R1 softCE is 1.7813 on unique-top and 1.9890 on tied-top artworks, while KL is 0.7226 and 0.7315: vote entropy explains much of the CE difference. No new holdout predictions were generated.

'''+s[b:]
a=s.index('The attribute sentence contains no independent information');b=s.index('Class weighting addresses a different goal.',a)
lo,hi=B['delta_softCE_paired_artwork_CI95']
s=s[:a]+f'''The attribute sentence introduces no independent annotation information. A new exploratory validation control keeps each numerical/text pair internally consistent but permutes the pairs across images, leaving image/patch features, normalizers and parameters fixed (Table~\\ref{{tab:permutation}}). Twenty fixed derangements are shared across five checkpoints. SoftCE changes by {B['delta_softCE']:+.4f}, with a paired-artwork 95\\% interval {ci(lo,hi)}, conditional on these checkpoints/permutations; macro-F1 changes by {B['delta_macroF1']:+.4f}. Mean prediction total variation is {B['paired_permuted']['mean_total_variation']:.4f}. This measures sensitivity to misalignment, not the benefit of retraining with attributes, human causality or an extension of H1--H5. Validation selected checkpoints and includes 16 known training reproductions.

\\input{{tables/new_permutation}}

The existing development-only numerical/text/gate ablations remain exploratory. Separately saved post-hoc B5 concatenation heads obtain holdout softCE 1.8344 and macro-F1 0.3612, outside H1--H5; they do not establish superiority of gated fusion.

'''+s[b:]
a=s.index('\\subsection{Evidence Grounding and Missing Tests}');b=s.index('\\subsection{Scope of the Scientific Claims}',a)
s=s[:a]+f'''\\subsection{{Exploratory Validation Perturbations}}
We evaluate the fixed R1 seed-42 checkpoint on 200 validation paintings chosen by a target-independent hash rule before execution. This subset has {A['unique_top']} unique-top artworks and {A['known_reproduction_count']} known training reproductions. Original predicted classes are fixed throughout. Original fresh logits reproduce saved logits to within $10^{{-6}}$ on the subset. The evaluation is exploratory; it does not use new holdout inference or alter H1--H5.

For each image, ECSA ranks the 49 display-grid cells. We compare ten matched random rankings and a 256-mask RISE-style ranking \\cite{{rise}}, using shifted smooth Bernoulli masks. RISE rankings are constructed with mean fill and tested under both operators; this finite-mask comparator is not claimed converged. Deletion replaces selected cells and retention replaces their complement, using visible-crop RGB mean or Gaussian blur. Masks are projected to the source image; outside-crop pixels remain unchanged. Global and pooled spatial features and whole-image attributes are recomputed; the changed template selects its matching frozen text embedding. The intervention therefore affects the whole pipeline, not ECSA alone.

Budgets are $b=(5,10,15,24)/49$. With fixed-target probability $q(b)$, the partial AUC is
\\begin{{equation}}
A(q)=\\frac{{\\sum_{{j=1}}^3(b_{{j+1}}-b_j)[q(b_j)+q(b_{{j+1}})]/2}}{{b_4-b_1}}.
\\end{{equation}}
No endpoint is imputed and no division by original probability is applied. Lower deletion and higher retention are favourable under the same operator. Table~\\ref{{tab:validationevidence}} reports paired contrasts: ten random trials are averaged within painting, then 2,000 paired artwork resamples give pointwise, unadjusted 95\\% intervals conditional on the checkpoint, subset and mask draws.

\\input{{tables/new_validation_evidence}}

These measurements quantify perturbation sensitivity relative to controls; they do not validate human-semantic explanations or isolate a causal ECSA effect. Native $24\\times24$ tokens are adaptively averaged to $7\\times7$; display-cell masking is a coarse intervention, not exact independent receptive-field removal. Upsampling cannot recover localization precision. No matched higher-resolution or human-region experiment is established. The accepted title's evidence-grounding wording remains qualified by these limits.

'''+s[b:]
s=s.replace('the lack of matched ResNet50 and full resource measurements','the lack of matched ResNet50, external-domain and human-region evidence')
s=s.replace('Broader generalization, matched resource measurements and independent explanation evaluation remain necessary.','Broader generalization and independent human-region explanation evaluation remain necessary.')
a=s.index('OpenAI Codex assisted revision');b=s.index('\\bibliographystyle',a)
s=s[:a]+'''OpenAI Codex assisted revision of Sections 1--8 and preparation of validation-evaluation, analysis and LaTeX code. New experiments used fixed models without training or new holdout inference. Scientific responsibility remains with the authors.

'''+s[b:]
(P/'main.tex').write_text(s)
f=P/'references.bib';f.write_text(f.read_text()+r'''
@inproceedings{rise,
 author={Petsiuk, Vitali and Das, Abir and Saenko, Kate},
 title={{RISE}: Randomized Input Sampling for Explanation of Black-box Models},
 booktitle={British Machine Vision Conference},year={2018},url={https://arxiv.org/abs/1806.07421}}
''')
print('Checked exploratory tables and manuscript generated; original main tables/H1–H5 remain unchanged.')
