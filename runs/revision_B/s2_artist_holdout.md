## Step 2. Artist-held-out experiment (L/14 R1, seeds 42–44; no val or test used)

Split (runs/revision_B/step2/split.csv, code/s2_split.py): 10% of the 43K-train artists, sampled within artwork-count strata, form the unseen-artist holdout (all their artworks). A seen-artist control of the same size is drawn at random from the other artists' artworks (each artist keeps ≥ 1 training artwork). A 5% selection split of the remaining artworks replaces val for early stopping / checkpoint selection (min softCE). Training: 32,868 artworks, 961 artists. Recipe otherwise identical to Step 1 R1.

### Holdout composition

| holdout | artworks | artists | unique-top | mean vote entropy | mean vote share (amu awe con exc ang dis fea sad som) |
|---|---|---|---|---|---|
| holdout_unseen | 4,207 | 107 | 3,155 | 1.0831 | 0.121 0.145 0.232 0.089 0.016 0.056 0.108 0.109 0.124 |
| holdout_seen | 4,207 | 732 | 3,116 | 1.0509 | 0.102 0.172 0.297 0.085 0.013 0.039 0.078 0.098 0.116 |

### Metrics per holdout (mean ± SD over 3 seeds) and gap (unseen − seen) with artist-cluster bootstrap 95% CI

| metric | unseen artists | seen-artist control | gap (unseen − seen) | 95% CI |
|---|---|---|---|---|
| softCE | 1.8080 ± 0.0010 | 1.7383 ± 0.0012 | +0.0697 | [+0.0068, +0.1321] |
| KL | 0.7249 ± 0.0010 | 0.6874 ± 0.0012 | +0.0375 | [+0.0030, +0.0724] |
| macro-F1 | 0.3993 ± 0.0024 | 0.3482 ± 0.0077 | +0.0511 | [+0.0055, +0.0813] |
| mean anger+fear AP | 0.4704 ± 0.0040 | 0.4204 ± 0.0010 | +0.0501 | [-0.0125, +0.1022] |
| macro AP (9) | 0.6225 ± 0.0016 | 0.6034 ± 0.0015 | +0.0191 | [-0.0060, +0.0378] |

Per-seed gaps: s42: softCE +0.0709, KL +0.0387, macro-F1 +0.0599; s43: softCE +0.0688, KL +0.0366, macro-F1 +0.0487; s44: softCE +0.0694, KL +0.0372, macro-F1 +0.0448.
