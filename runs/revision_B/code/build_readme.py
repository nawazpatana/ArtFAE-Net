"""Assemble runs/revision_B/README.md (duplicate classification table + all step tables)."""
import pandas as pd
from common import OUT

c = pd.read_csv(OUT / "duplicates/classification.csv")
lines = ["## Step 0a. pHash pair classification (contact sheets in duplicates/pair_NN.jpg)", "",
         "39 distinct pairs flagged in revision_A (23 test–43K train, 16 val–43K train; the 6 test–14K train pairs "
         "are a subset of the test–43K pairs). Classified by visual inspection of each contact sheet.", "",
         "| classification | test pairs | val pairs | total |", "|---|---|---|---|"]
for k in ("same artwork", "same artwork, different reproduction", "different"):
    g = c[c.classification == k]
    lines.append(f"| {k} | {(g.eval_side_splits == 'test').sum()} | {(g.eval_side_splits == 'val').sum()} | {len(g)} |")
lines += ["", "Confirmed duplicates (not 'different'): 15 distinct test artworks and 16 val artworks, all against the "
          "43K train; none of the 6 pairs against the paper 14K train is a duplicate.", "",
          "| pair | train side | eval side | split | pHash | classification | note |", "|---|---|---|---|---|---|---|"]
for _, r in c.iterrows():
    lines.append(f"| {r.pair} | {r.train_side} | {r.eval_side} | {r.eval_side_splits} | {r.hamming} | "
                 f"{r.classification} | {r.note} |")
head = """# Revision Phase B: distribution prediction as the primary task

Generated 2026-11-03. Val only, except Step 0b, which re-reads the existing saved test logits for the requested
deduplication sensitivity analysis (no model run, nothing tuned). runs/final_eval* and runs/revision_A are
unchanged (revision_A/code/verify_untouched.py: 106 hashed artifacts match, no modified files). No
pre-registration written.

Code: code/ (contact_sheets.py, s0_dedup_sensitivity.py, extract_crop_attrs.py, train_abl.py, s1_eval.py,
s2_split.py, s2_eval.py). Runs: step1/<condition>_s<seed>/, step2/R1_s<seed>/. Logs: logs/.
"""
parts = [head, "\n".join(lines)] + [(OUT / f).read_text() for f in
         ("s0_dedup_sensitivity.md", "s1_ablations.md", "s1b_followup.md", "s2_artist_holdout.md")]
(OUT / "README.md").write_text("\n\n".join(p.strip() for p in parts) + "\n")
print("README.md written")
