"""Assemble runs/revision_A/README.md from the per-task markdown tables."""
from common import OUT

HEADER = """# Revision Phase A: analyses on existing outputs

Generated 2026-11-03 by `code/run_all.sh`. No training, no model selection, no new test inference. Existing
val/test logits (runs/*/val_logits.npy, runs/final_eval/**/test_logits, runs/final_eval_v2/test/*) were re-read.
Integrity checks: per-seed test macro-F1 and softCE(T) reproduce runs/final_eval_v2/test_per_seed.csv (25 runs), the
regenerated bootstrap reproduces runs/final_eval_v2/bootstrap.json, and all 106 artifacts hashed in
runs/final_selection_v2.txt are unchanged; nothing under runs/final_eval* was modified (logs/verify_untouched.log).

Models: R0n = M1 trained on paper 14K (B/32); B/32 R1; L/14 R1; L/14 R3 (raw); L/14 R3c = L/14 R3 + prior correction
log(val prior) − log(train prior). Test = 2,999 artworks (2,246 unique-top, 753 tied).

| item | file(s) |
|---|---|
| 1 baselines | t1_baselines.md / .csv, t1_train_priors.json |
| 2 tied vs unique-top | t2_ties.md, t2_ties_per_seed.csv |
| 3 confusion, counts, per-class F1 CIs | t3_per_class.md, t3_confusion_matrices.md / .json (all 30 matrices), t3_per_class_f1_ci.csv, t3_prediction_counts_per_seed.csv |
| 4 seen vs unseen | t4_seen_unseen.md, t4_seen_unseen_counts.csv, t4_seen_unseen_per_seed.csv |
| 5 parameter counts | t5_params.md, t5_params_all_checkpoints.csv |
| 6 subset documentation | subset_construction.md (+ t6_class_distributions.csv, t6_overlaps.csv, t6_near_duplicate_cross_split.csv) |
| 7 feature specification | feature_spec.md (+ t7_attribute_ranges_train43k.csv, t7_sentence_counts.csv, t7_attr_text_check.csv) |
| 8 TOST for comparison (d) | t8_tost.md, t8_tost.json |

Seed-averaged confusion matrices are included below; the 25 per-seed matrices are in t3_confusion_matrices.md.
"""


def seed_avg_only(md):
    out, keep = [], True
    for line in md.splitlines():
        if line.startswith("#### "):
            keep = "seed-averaged" in line
        if line.startswith("### "):
            keep = True
        if keep:
            out.append(line)
    return "\n".join(out)


parts = [HEADER]
for f in ("t1_baselines.md", "t2_ties.md", "t3_per_class.md"):
    parts.append((OUT / f).read_text())
parts.append(seed_avg_only((OUT / "t3_confusion_matrices.md").read_text()).replace(
    "## Task 3. Confusion matrices", "## Task 3. Seed-averaged confusion matrices"))
for f in ("t4_seen_unseen.md", "t5_params.md", "t8_tost.md"):
    parts.append((OUT / f).read_text())
parts.append("## Items 6 and 7\n\nSee subset_construction.md and feature_spec.md.\n")
(OUT / "README.md").write_text("\n\n".join(p.strip() for p in parts) + "\n")
print("README.md written")
