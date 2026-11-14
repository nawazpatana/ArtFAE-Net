# Revision Phase A: analyses on existing outputs

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

## Task 1. Reference baselines (constant predictors)

Hard metrics on unique-top artworks; softCE, KL, Brier on all artworks, eps = 1e-12 as in artfae_balance.evaluate. The one-hot majority predictor assigns probability 0 to eight classes, so its softCE and KL are dominated by log(1e-12) ≈ −27.6 and are not meaningful as calibration numbers; they are reported only for completeness.

| split | baseline | n | n unique-top | acc | macro-F1 | balAcc | softCE | KL | Brier |
|---|---|---|---|---|---|---|---|---|---|
| val | majority class (contentment, one-hot) | 3000 | 2243 | 0.4062 | 0.0642 | 0.1111 | 19.6108 | 18.5641 | 0.8248 |
| val | mean train vote distribution, 43K | 3000 | 2243 | 0.4062 | 0.0642 | 0.1111 | 1.9963 | 0.9497 | 0.2442 |
| val | mean train vote distribution, paper 14K | 3000 | 2243 | 0.4062 | 0.0642 | 0.1111 | 1.9926 | 0.9460 | 0.2438 |
| test | majority class (contentment, one-hot) | 2999 | 2246 | 0.4056 | 0.0641 | 0.1111 | 19.7797 | 18.7262 | 0.8332 |
| test | mean train vote distribution, 43K | 2999 | 2246 | 0.4056 | 0.0641 | 0.1111 | 2.0080 | 0.9545 | 0.2436 |
| test | mean train vote distribution, paper 14K | 2999 | 2246 | 0.4056 | 0.0641 | 0.1111 | 2.0035 | 0.9501 | 0.2431 |

Mean training vote distributions (class order amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something_else):

- mean train vote distribution, 43K: 0.1014, 0.1675, 0.2931, 0.0849, 0.0127, 0.0408, 0.0825, 0.0994, 0.1177
- mean train vote distribution, paper 14K: 0.0972, 0.1695, 0.2881, 0.0808, 0.0200, 0.0529, 0.0840, 0.1020, 0.1055

## Task 2. Distribution metrics on unique-top vs tied test artworks

Test: 2246 unique-top, 753 tied (top-vote tie, isclose atol 1e-5). Mean ± SD over seeds 42–46. Integrity check passed: macro-F1 and softCE(T) reproduce runs/final_eval_v2/test_per_seed.csv for all 25 model × seed runs.

Entropy of the human vote distribution (the softCE floor, = softCE − KL): unique-top 1.0020, tied 1.2071, all 1.0535.

### T=1 (primary: raw probabilities)

| model | softCE unique-top | softCE tied | KL unique-top | KL tied | Brier unique-top | Brier tied | softCE all | KL all | Brier all |
|---|---|---|---|---|---|---|---|---|---|
| R0n | 1.8009 ± 0.0114 | 2.0103 ± 0.0181 | 0.7989 ± 0.0114 | 0.8033 ± 0.0181 | 0.2070 ± 0.0031 | 0.1697 ± 0.0049 | 1.8535 ± 0.0129 | 0.8000 ± 0.0129 | 0.1976 ± 0.0035 |
| B/32 R1 | 1.7747 ± 0.0221 | 1.9962 ± 0.0202 | 0.7727 ± 0.0221 | 0.7892 ± 0.0202 | 0.1973 ± 0.0064 | 0.1668 ± 0.0052 | 1.8303 ± 0.0216 | 0.7768 ± 0.0216 | 0.1896 ± 0.0060 |
| L/14 R1 | 1.7179 ± 0.0141 | 1.9655 ± 0.0159 | 0.7160 ± 0.0141 | 0.7584 ± 0.0159 | 0.1799 ± 0.0044 | 0.1597 ± 0.0048 | 1.7801 ± 0.0145 | 0.7266 ± 0.0145 | 0.1748 ± 0.0045 |
| L/14 R3 | 1.7274 ± 0.0028 | 1.9610 ± 0.0050 | 0.7255 ± 0.0028 | 0.7539 ± 0.0050 | 0.1842 ± 0.0009 | 0.1570 ± 0.0021 | 1.7861 ± 0.0032 | 0.7326 ± 0.0032 | 0.1773 ± 0.0006 |
| L/14 R3c | 1.7280 ± 0.0032 | 1.9603 ± 0.0055 | 0.7260 ± 0.0032 | 0.7532 ± 0.0055 | 0.1845 ± 0.0013 | 0.1574 ± 0.0024 | 1.7863 ± 0.0036 | 0.7329 ± 0.0036 | 0.1777 ± 0.0012 |

### T(val) (temperature fitted on val per seed, as in final_eval_v2)

| model | softCE unique-top | softCE tied | KL unique-top | KL tied | Brier unique-top | Brier tied | softCE all | KL all | Brier all |
|---|---|---|---|---|---|---|---|---|---|
| R0n | 1.7987 ± 0.0089 | 1.9918 ± 0.0064 | 0.7968 ± 0.0089 | 0.7847 ± 0.0064 | 0.2074 ± 0.0029 | 0.1632 ± 0.0012 | 1.8472 ± 0.0081 | 0.7937 ± 0.0081 | 0.1963 ± 0.0024 |
| B/32 R1 | 1.7716 ± 0.0179 | 1.9741 ± 0.0069 | 0.7697 ± 0.0179 | 0.7670 ± 0.0069 | 0.1982 ± 0.0063 | 0.1593 ± 0.0012 | 1.8225 ± 0.0151 | 0.7690 ± 0.0151 | 0.1884 ± 0.0050 |
| L/14 R1 | 1.7178 ± 0.0120 | 1.9500 ± 0.0036 | 0.7159 ± 0.0120 | 0.7429 ± 0.0036 | 0.1810 ± 0.0047 | 0.1542 ± 0.0011 | 1.7761 ± 0.0098 | 0.7226 ± 0.0098 | 0.1743 ± 0.0037 |
| L/14 R3 | 1.7281 ± 0.0031 | 1.9543 ± 0.0041 | 0.7261 ± 0.0031 | 0.7472 ± 0.0041 | 0.1849 ± 0.0006 | 0.1546 ± 0.0017 | 1.7849 ± 0.0030 | 0.7314 ± 0.0030 | 0.1773 ± 0.0007 |
| L/14 R3c | 1.7285 ± 0.0036 | 1.9558 ± 0.0046 | 0.7265 ± 0.0036 | 0.7488 ± 0.0046 | 0.1849 ± 0.0013 | 0.1556 ± 0.0018 | 1.7856 ± 0.0035 | 0.7321 ± 0.0035 | 0.1775 ± 0.0012 |

## Task 3. Per-class F1 (unique-top test) with 95% bootstrap CIs

Point estimate: mean (± SD) of per-seed F1 over seeds 42–46. CI: 2,000 paired artwork resamples (rng seed 0, same draws as final_eval_v2), per-class F1 averaged over seeds within each resample, 95% percentile interval.

| emotion | support | R0n | B/32 R1 | L/14 R1 | L/14 R3 | L/14 R3c |
|---|---|---|---|---|---|---|
| amusement | 189 | 0.381 ± 0.026 [0.328, 0.429] | 0.386 ± 0.023 [0.335, 0.436] | 0.444 ± 0.006 [0.389, 0.498] | 0.427 ± 0.018 [0.373, 0.479] | 0.431 ± 0.011 [0.375, 0.484] |
| awe | 357 | 0.395 ± 0.006 [0.349, 0.436] | 0.392 ± 0.020 [0.344, 0.435] | 0.454 ± 0.015 [0.408, 0.495] | 0.454 ± 0.010 [0.408, 0.495] | 0.463 ± 0.010 [0.419, 0.504] |
| contentment | 911 | 0.673 ± 0.006 [0.650, 0.695] | 0.692 ± 0.007 [0.671, 0.712] | 0.718 ± 0.006 [0.697, 0.738] | 0.714 ± 0.010 [0.693, 0.735] | 0.713 ± 0.010 [0.692, 0.735] |
| excitement | 97 | 0.275 ± 0.011 [0.198, 0.347] | 0.272 ± 0.025 [0.194, 0.344] | 0.279 ± 0.021 [0.203, 0.350] | 0.294 ± 0.024 [0.222, 0.363] | 0.294 ± 0.020 [0.215, 0.368] |
| anger | 14 | 0.000 ± 0.000 [0.000, 0.000] | 0.000 ± 0.000 [0.000, 0.000] | 0.038 ± 0.085 [0.000, 0.083] | 0.118 ± 0.088 [0.000, 0.246] | 0.118 ± 0.035 [0.000, 0.253] |
| disgust | 106 | 0.195 ± 0.014 [0.129, 0.260] | 0.100 ± 0.040 [0.049, 0.154] | 0.176 ± 0.026 [0.099, 0.252] | 0.190 ± 0.016 [0.116, 0.261] | 0.207 ± 0.021 [0.140, 0.272] |
| fear | 201 | 0.436 ± 0.026 [0.383, 0.488] | 0.457 ± 0.006 [0.398, 0.509] | 0.516 ± 0.011 [0.456, 0.571] | 0.540 ± 0.014 [0.486, 0.589] | 0.541 ± 0.006 [0.488, 0.590] |
| sadness | 232 | 0.364 ± 0.014 [0.310, 0.414] | 0.376 ± 0.017 [0.323, 0.428] | 0.480 ± 0.006 [0.422, 0.531] | 0.483 ± 0.014 [0.432, 0.531] | 0.481 ± 0.013 [0.430, 0.529] |
| something_else | 139 | 0.207 ± 0.012 [0.149, 0.269] | 0.250 ± 0.015 [0.184, 0.315] | 0.214 ± 0.004 [0.146, 0.287] | 0.229 ± 0.011 [0.170, 0.293] | 0.220 ± 0.024 [0.156, 0.288] |

### Prediction counts per class, mean ± SD over seeds (unique-top test, n = 2,246; true support in the first column)

| emotion | true support | R0n | B/32 R1 | L/14 R1 | L/14 R3 | L/14 R3c |
|---|---|---|---|---|---|---|
| amusement | 189 | 241.4 ± 26.4 | 243.8 ± 44.8 | 215.4 ± 41.1 | 211.8 ± 36.8 | 204.8 ± 29.7 |
| awe | 357 | 335.4 ± 16.9 | 276.8 ± 39.7 | 314.6 ± 46.3 | 318.0 ± 84.0 | 350.2 ± 93.8 |
| contentment | 911 | 1040.2 ± 31.1 | 1140.4 ± 59.8 | 1140.2 ± 46.1 | 971.0 ± 51.2 | 970.2 ± 58.0 |
| excitement | 97 | 77.6 ± 9.6 | 75.4 ± 17.8 | 90.8 ± 8.6 | 99.2 ± 36.6 | 77.2 ± 27.6 |
| anger | 14 | 1.6 ± 1.8 | 0.4 ± 0.9 | 2.2 ± 3.2 | 11.8 ± 7.7 | 24.2 ± 13.2 |
| disgust | 106 | 72.6 ± 9.4 | 23.6 ± 6.9 | 29.4 ± 7.1 | 45.0 ± 15.0 | 72.6 ± 19.2 |
| fear | 201 | 213.6 ± 28.2 | 196.6 ± 12.0 | 185.8 ± 22.4 | 220.4 ± 31.8 | 222.8 ± 32.8 |
| sadness | 232 | 180.4 ± 15.0 | 217.8 ± 36.4 | 199.6 ± 27.3 | 255.2 ± 47.3 | 250.4 ± 47.5 |
| something_else | 139 | 83.2 ± 9.3 | 71.2 ± 15.7 | 68.0 ± 12.6 | 113.6 ± 47.4 | 73.6 ± 35.8 |

### Prediction counts per class on all 2,999 test artworks (incl. tied), mean ± SD over seeds

| emotion | R0n | B/32 R1 | L/14 R1 | L/14 R3 | L/14 R3c |
|---|---|---|---|---|---|
| amusement | 331.8 ± 34.1 | 328.8 ± 65.1 | 302.0 ± 56.2 | 287.0 ± 49.6 | 276.4 ± 41.0 |
| awe | 463.6 ± 23.4 | 382.8 ± 56.9 | 432.4 ± 62.0 | 437.8 ± 120.1 | 485.2 ± 132.9 |
| contentment | 1341.6 ± 45.7 | 1487.4 ± 95.8 | 1482.0 ± 66.3 | 1245.4 ± 68.1 | 1243.2 ± 75.9 |
| excitement | 115.0 ± 12.1 | 112.2 ± 25.6 | 132.4 ± 12.4 | 149.4 ± 55.9 | 117.8 ± 44.3 |
| anger | 3.0 ± 2.7 | 1.0 ± 1.0 | 3.0 ± 3.4 | 13.8 ± 8.7 | 30.6 ± 15.8 |
| disgust | 102.4 ± 12.7 | 37.2 ± 13.5 | 42.6 ± 8.8 | 62.6 ± 21.9 | 102.4 ± 29.8 |
| fear | 289.6 ± 40.9 | 263.6 ± 18.2 | 241.6 ± 28.6 | 291.8 ± 40.0 | 294.8 ± 41.9 |
| sadness | 240.2 ± 21.7 | 288.0 ± 58.0 | 271.2 ± 41.5 | 352.0 ± 64.2 | 347.0 ± 66.6 |
| something_else | 111.8 ± 14.2 | 98.0 ± 19.9 | 91.8 ± 18.0 | 159.2 ± 68.7 | 101.6 ± 49.9 |

## Task 3. Seed-averaged confusion matrices, unique-top test artworks (n = 2,246)

Rows: human top emotion. Columns: predicted argmax. Last row: number of predictions per class.

### R0n

#### seed-averaged (mean of the 5 per-seed count matrices)

| true \ pred | amu | awe | con | exc | ang | dis | fea | sad | som | support |
|---|---|---|---|---|---|---|---|---|---|---|
| amusement | 82.2 | 16.2 | 44.8 | 14.6 | 0.0 | 9.0 | 10.2 | 4.4 | 7.6 | 189.0 |
| awe | 20.6 | 136.6 | 133.4 | 8.4 | 0.2 | 7.4 | 26.0 | 14.6 | 9.8 | 357.0 |
| contentment | 56.2 | 84.2 | 657.0 | 17.4 | 0.0 | 12.2 | 27.2 | 39.0 | 17.8 | 911.0 |
| excitement | 18.8 | 14.0 | 27.6 | 24.0 | 0.2 | 2.0 | 2.8 | 3.0 | 4.6 | 97.0 |
| anger | 2.0 | 3.2 | 1.0 | 0.0 | 0.0 | 0.6 | 6.0 | 0.0 | 1.2 | 14.0 |
| disgust | 18.8 | 9.4 | 24.6 | 3.2 | 0.2 | 17.4 | 17.2 | 9.6 | 5.6 | 106.0 |
| fear | 13.8 | 20.4 | 32.2 | 2.8 | 0.8 | 9.2 | 90.6 | 22.0 | 9.2 | 201.0 |
| sadness | 17.6 | 32.6 | 66.4 | 5.6 | 0.2 | 8.4 | 21.8 | 75.0 | 4.4 | 232.0 |
| something_else | 11.4 | 18.8 | 53.2 | 1.6 | 0.0 | 6.4 | 11.8 | 12.8 | 23.0 | 139.0 |
| **#pred** | 241.4 | 335.4 | 1040.2 | 77.6 | 1.6 | 72.6 | 213.6 | 180.4 | 83.2 | 2246.0 |

### B/32 R1

#### seed-averaged (mean of the 5 per-seed count matrices)

| true \ pred | amu | awe | con | exc | ang | dis | fea | sad | som | support |
|---|---|---|---|---|---|---|---|---|---|---|
| amusement | 83.8 | 13.6 | 50.8 | 14.0 | 0.0 | 4.8 | 6.0 | 11.2 | 4.8 | 189.0 |
| awe | 21.2 | 124.2 | 151.2 | 7.8 | 0.0 | 2.4 | 20.6 | 24.4 | 5.2 | 357.0 |
| contentment | 52.0 | 56.8 | 709.4 | 17.6 | 0.2 | 2.6 | 20.4 | 39.0 | 13.0 | 911.0 |
| excitement | 16.8 | 10.6 | 35.6 | 23.6 | 0.0 | 0.2 | 3.6 | 3.2 | 3.4 | 97.0 |
| anger | 2.0 | 3.8 | 1.4 | 0.0 | 0.0 | 0.0 | 5.6 | 0.2 | 1.0 | 14.0 |
| disgust | 19.4 | 8.4 | 29.0 | 3.6 | 0.0 | 6.6 | 17.8 | 13.4 | 7.8 | 106.0 |
| fear | 14.0 | 18.2 | 37.4 | 2.4 | 0.0 | 2.4 | 90.8 | 29.4 | 6.4 | 201.0 |
| sadness | 19.6 | 24.0 | 71.8 | 3.8 | 0.2 | 2.8 | 22.0 | 84.4 | 3.4 | 232.0 |
| something_else | 15.0 | 17.2 | 53.8 | 2.6 | 0.0 | 1.8 | 9.8 | 12.6 | 26.2 | 139.0 |
| **#pred** | 243.8 | 276.8 | 1140.4 | 75.4 | 0.4 | 23.6 | 196.6 | 217.8 | 71.2 | 2246.0 |

### L/14 R1

#### seed-averaged (mean of the 5 per-seed count matrices)

| true \ pred | amu | awe | con | exc | ang | dis | fea | sad | som | support |
|---|---|---|---|---|---|---|---|---|---|---|
| amusement | 89.8 | 14.2 | 45.8 | 15.4 | 0.0 | 3.0 | 5.8 | 8.0 | 7.0 | 189.0 |
| awe | 18.6 | 152.4 | 134.4 | 11.2 | 0.8 | 3.0 | 15.8 | 13.8 | 7.0 | 357.0 |
| contentment | 41.4 | 63.4 | 736.2 | 20.8 | 0.4 | 0.8 | 8.2 | 26.4 | 13.4 | 911.0 |
| excitement | 16.0 | 10.0 | 32.0 | 26.2 | 0.0 | 1.0 | 6.0 | 3.0 | 2.8 | 97.0 |
| anger | 0.2 | 3.2 | 2.2 | 0.2 | 0.4 | 1.4 | 5.4 | 0.0 | 1.0 | 14.0 |
| disgust | 16.2 | 5.8 | 32.4 | 4.4 | 0.4 | 12.0 | 17.8 | 11.8 | 5.2 | 106.0 |
| fear | 13.2 | 24.8 | 27.2 | 3.2 | 0.0 | 2.6 | 99.8 | 23.2 | 7.0 | 201.0 |
| sadness | 9.0 | 20.0 | 73.2 | 4.0 | 0.2 | 3.0 | 16.6 | 103.6 | 2.4 | 232.0 |
| something_else | 11.0 | 20.8 | 56.8 | 5.4 | 0.0 | 2.6 | 10.4 | 9.8 | 22.2 | 139.0 |
| **#pred** | 215.4 | 314.6 | 1140.2 | 90.8 | 2.2 | 29.4 | 185.8 | 199.6 | 68.0 | 2246.0 |

### L/14 R3

#### seed-averaged (mean of the 5 per-seed count matrices)

| true \ pred | amu | awe | con | exc | ang | dis | fea | sad | som | support |
|---|---|---|---|---|---|---|---|---|---|---|
| amusement | 85.8 | 17.6 | 30.0 | 15.4 | 0.8 | 6.8 | 7.4 | 12.0 | 13.2 | 189.0 |
| awe | 16.4 | 153.4 | 114.4 | 12.0 | 2.4 | 4.2 | 22.0 | 19.0 | 13.2 | 357.0 |
| contentment | 47.2 | 75.0 | 671.6 | 26.2 | 1.8 | 3.2 | 13.0 | 43.2 | 29.8 | 911.0 |
| excitement | 15.6 | 9.6 | 22.4 | 29.0 | 0.8 | 3.0 | 7.4 | 5.0 | 4.2 | 97.0 |
| anger | 0.4 | 2.8 | 0.4 | 0.6 | 1.6 | 1.4 | 3.8 | 1.4 | 1.6 | 14.0 |
| disgust | 17.8 | 5.2 | 19.4 | 4.0 | 0.4 | 14.4 | 20.0 | 15.8 | 9.0 | 106.0 |
| fear | 10.4 | 16.0 | 21.0 | 2.4 | 1.2 | 4.0 | 113.8 | 23.4 | 8.8 | 201.0 |
| sadness | 7.6 | 19.4 | 51.0 | 4.0 | 2.6 | 4.4 | 20.4 | 117.8 | 4.8 | 232.0 |
| something_else | 10.6 | 19.0 | 40.8 | 5.6 | 0.2 | 3.6 | 12.6 | 17.6 | 29.0 | 139.0 |
| **#pred** | 211.8 | 318.0 | 971.0 | 99.2 | 11.8 | 45.0 | 220.4 | 255.2 | 113.6 | 2246.0 |

### L/14 R3c

#### seed-averaged (mean of the 5 per-seed count matrices)

| true \ pred | amu | awe | con | exc | ang | dis | fea | sad | som | support |
|---|---|---|---|---|---|---|---|---|---|---|
| amusement | 85.0 | 20.0 | 30.8 | 13.2 | 1.8 | 12.0 | 7.8 | 11.8 | 6.6 | 189.0 |
| awe | 15.4 | 164.0 | 109.6 | 8.2 | 3.4 | 8.0 | 21.2 | 19.0 | 8.2 | 357.0 |
| contentment | 47.6 | 83.8 | 671.0 | 18.8 | 3.2 | 8.4 | 14.6 | 44.8 | 18.8 | 911.0 |
| excitement | 15.2 | 11.2 | 24.2 | 25.8 | 1.6 | 4.4 | 8.4 | 4.8 | 1.4 | 97.0 |
| anger | 0.0 | 3.0 | 0.4 | 0.2 | 2.2 | 1.8 | 3.8 | 1.0 | 1.6 | 14.0 |
| disgust | 15.6 | 7.0 | 21.4 | 2.8 | 3.2 | 18.6 | 18.6 | 13.4 | 5.4 | 106.0 |
| fear | 9.0 | 17.6 | 20.8 | 1.6 | 3.8 | 6.4 | 114.6 | 21.8 | 5.4 | 201.0 |
| sadness | 7.0 | 20.8 | 50.2 | 2.4 | 4.6 | 7.2 | 21.2 | 116.0 | 2.6 | 232.0 |
| something_else | 10.0 | 22.8 | 41.8 | 4.2 | 0.4 | 5.8 | 12.6 | 17.8 | 23.6 | 139.0 |
| **#pred** | 204.8 | 350.2 | 970.2 | 77.2 | 24.2 | 72.6 | 222.8 | 250.4 | 73.6 | 2246.0 |

## Task 4. Seen vs unseen artist / art_style (test)

artist = painting slug before the first underscore; training set = the run's train manifest. macro-F1 / acc on unique-top artworks of the group (macro over classes present in the group); softCE (T = 1) on all artworks of the group. Mean ± SD over seeds 42–46.

### artist

| model | train set | group | n test | n unique-top | distinct | classes | macro-F1 | acc | softCE |
|---|---|---|---|---|---|---|---|---|---|
| R0n | 14,000 | seen | 2977 | 2229 | 595 | 9 | 0.3246 ± 0.0068 | 0.4929 ± 0.0084 | 1.8534 ± 0.0128 |
| R0n | 14,000 | unseen | 22 | 17 | 21 | 8 | 0.3769 ± 0.0691 | 0.4235 ± 0.0767 | 1.8647 ± 0.0264 |
| B/32 R1 | 43,012 | seen | 2993 | 2241 | 610 | 9 | 0.3245 ± 0.0052 | 0.5115 ± 0.0096 | 1.8302 ± 0.0216 |
| B/32 R1 | 43,012 | unseen | 6 | 5 | 6 | 4 | 0.5000 ± 0.2569 | 0.5600 ± 0.2608 | 1.8669 ± 0.0686 |
| L/14 R1 | 43,012 | seen | 2993 | 2241 | 610 | 9 | 0.3678 ± 0.0086 | 0.5531 ± 0.0080 | 1.7798 ± 0.0145 |
| L/14 R1 | 43,012 | unseen | 6 | 5 | 6 | 4 | 0.5833 ± 0.0000 | 0.6000 ± 0.0000 | 1.9083 ± 0.0392 |
| L/14 R3 | 43,012 | seen | 2993 | 2241 | 610 | 9 | 0.3825 ± 0.0101 | 0.5415 ± 0.0040 | 1.7859 ± 0.0033 |
| L/14 R3 | 43,012 | unseen | 6 | 5 | 6 | 4 | 0.5417 ± 0.0932 | 0.5600 ± 0.0894 | 1.8933 ± 0.0456 |
| L/14 R3c | 43,012 | seen | 2993 | 2241 | 610 | 9 | 0.3846 ± 0.0053 | 0.5436 ± 0.0045 | 1.7861 ± 0.0036 |
| L/14 R3c | 43,012 | unseen | 6 | 5 | 6 | 4 | 0.4917 ± 0.1264 | 0.5200 ± 0.1095 | 1.8830 ± 0.0477 |

### art_style

| model | train set | group | n test | n unique-top | distinct | classes | macro-F1 | acc | softCE |
|---|---|---|---|---|---|---|---|---|---|
| R0n | 14,000 | seen | 2999 | 2246 | 27 | 9 | 0.3250 ± 0.0069 | 0.4923 ± 0.0084 | 1.8535 ± 0.0129 |
| R0n | 14,000 | unseen | 0 | 0 | 0 | – | – | – | – |
| B/32 R1 | 43,012 | seen | 2999 | 2246 | 27 | 9 | 0.3249 ± 0.0046 | 0.5116 ± 0.0093 | 1.8303 ± 0.0216 |
| B/32 R1 | 43,012 | unseen | 0 | 0 | 0 | – | – | – | – |
| L/14 R1 | 43,012 | seen | 2999 | 2246 | 27 | 9 | 0.3687 ± 0.0086 | 0.5533 ± 0.0079 | 1.7801 ± 0.0145 |
| L/14 R1 | 43,012 | unseen | 0 | 0 | 0 | – | – | – | – |
| L/14 R3 | 43,012 | seen | 2999 | 2246 | 27 | 9 | 0.3831 ± 0.0099 | 0.5416 ± 0.0041 | 1.7861 ± 0.0032 |
| L/14 R3 | 43,012 | unseen | 0 | 0 | 0 | – | – | – | – |
| L/14 R3c | 43,012 | seen | 2999 | 2246 | 27 | 9 | 0.3852 ± 0.0050 | 0.5435 ± 0.0044 | 1.7863 ± 0.0036 |
| L/14 R3c | 43,012 | unseen | 0 | 0 | 0 | – | – | – | – |

## Task 5. Parameter counts

nn.Parameters only (buffers excluded). One row per distinct architecture within a model family; `n ckpt` = number of checkpoints with exactly these counts. Heads on cached features add the frozen extractor at inference (last rows). Per-checkpoint list: t5_params_all_checkpoints.csv.

| model family | n ckpt | total params | trainable params | note |
|---|---|---|---|---|
| runs/L14_R1 | 5 | 762,249 | 762,249 |  |
| runs/L14_R3_lam0.3 | 5 | 763,410 | 763,410 |  |
| runs/R0n_paper14k | 5 | 631,177 | 631,177 |  |
| runs/R1 | 5 | 631,177 | 631,177 |  |
| runs/R2_tau0.5 | 1 | 631,177 | 631,177 |  |
| runs/R2_tau1.0 | 1 | 631,177 | 631,177 |  |
| runs/R2a_tau1.0 | 1 | 631,177 | 631,177 |  |
| runs/R2b_sampler | 5 | 631,177 | 631,177 |  |
| runs/R3_lam0.1 | 1 | 632,338 | 632,338 |  |
| runs/R3_lam0.3 | 5 | 632,338 | 632,338 |  |
| runs/R3g_gate | 5 | 632,339 | 632,339 |  |
| runs/R4a_textq | 1 | 697,874 | 697,874 |  |
| runs/R4b_textq_gate | 5 | 697,875 | 697,875 |  |
| workspace B3 | 3 | 140,809 | 140,809 |  |
| workspace B4 | 3 | 264,713 | 264,713 |  |
| workspace B5 | 3 | 271,881 | 271,881 |  |
| workspace B2 (CLIP B/32 linear head) | 1 | 4,617 | 4,617 |  |
| workspace B1 (ResNet50, frozen backbone) | 1 | 23,526,473 | 18,441 |  |
| home artfae_m1 M1 | 1 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical,text (paper M1, runs/m1_paper) |
| home artfae_m1 M1 | 8 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical,text |
| home artfae_m1 M1 | 2 | 241,865 | 241,865 | hidden 64, branches: gated,spatial,numerical,text |
| home artfae_m1 M1 | 1 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,text |
| home artfae_m1 M1 | 1 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical |
| home artfae_m1 M1 | 1 | 284,553 | 284,553 | hidden 128, branches: gated,numerical,text |
| home artfae_m1 M1 | 1 | 532,361 | 532,361 | hidden 128, branches: spatial,numerical,text |
| home baseline_simple_fusion B5 | 2 | 271,881 | 271,881 | hidden 128, branches: gated,spatial,numerical,text |
| home matched_component_runs M1 | 2 | 532,361 | 532,361 | hidden 128, branches: spatial,numerical,text |
| home matched_component_runs M1 | 2 | 284,553 | 284,553 | hidden 128, branches: gated,numerical,text |
| home matched_component_runs M1 | 2 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,text |
| home matched_component_runs M1 | 2 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical |
| home research_baselines M1 | 2 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical,text |
| home B2 (CLIP B/32 linear head) | 1 | 4,617 | 4,617 |  |
| home pipeline 14K clip_l14 | 2 | 303,202,313 | 151,179,273 | clip_l14, frozen_blocks=12/24 |
| home pipeline 14K siglip2_so400m | 2 | 428,250,953 | 228,619,065 | siglip2_so400m, frozen_blocks=13/27 |
| home pipeline 14K bundle current | 1 | 1,462,906,532 | 759,596,676 | 4 members: siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24; siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24 |
| home pipeline 14K bundle ensemble4 | 1 | 1,462,906,532 | 759,596,676 | 4 members: siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24; siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24 |
| home pipeline 14K bundle siglip_clipL | 1 | 731,453,266 | 379,798,338 | 2 members: siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24 |
| home pipeline 43K clip_l14 | 2 | 303,202,313 | 151,179,273 | clip_l14, frozen_blocks=12/24 |
| home pipeline 43K siglip2_so400m | 2 | 428,250,953 | 228,619,065 | siglip2_so400m, frozen_blocks=13/27 |
| home pipeline 43K bundle ensemble4 | 1 | 1,462,906,532 | 759,596,676 | 4 members: siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24; siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24 |
| openai/clip-vit-base-patch32 | 1 | 151,277,312 | 0 | vision+proj 87,849,216; text+proj 63,428,096 (text encodes the describe() sentence); logit_scale excluded |
| openai/clip-vit-large-patch14-336 | 1 | 427,944,192 | 0 | vision+proj 304,293,888; text+proj 123,650,304 (text encodes the describe() sentence); logit_scale excluded |

Inference-time totals for the Phase-A models (head + frozen extractor):

- R0n / B/32 R1: head 631,177 + clip-vit-base-patch32 151,277,312 = 151,908,489 (trainable 631,177 = 0.42%)
- L/14 R1: head 762,249 + clip-vit-large-patch14-336 427,944,192 = 428,706,441 (trainable 762,249 = 0.18%)
- L/14 R3 / R3c: head 763,410 + clip-vit-large-patch14-336 427,944,192 = 428,707,602 (trainable 763,410 = 0.18%)

## Task 8. Equivalence test (TOST), comparison (d) L/14 R3c − L/14 R3, macro-F1, margin ±0.01

Primary: the existing final_eval_v2 bootstrap (seed-averaged probabilities, 2,000 paired artwork resamples, rng seed 0), regenerated and checked to reproduce bootstrap.json exactly. Equivalence at α = 0.05 requires the 90% CI to lie inside (−0.01, +0.01).

| analysis | Δ macro-F1 | 90% CI | 95% CI | p (H0: Δ ≤ −0.01) | p (H0: Δ ≥ +0.01) | p TOST | equivalent? |
|---|---|---|---|---|---|---|---|
| bootstrap, 5-seed ensemble (primary) | -0.0112 | [-0.0232, +0.0007] | [-0.0260, +0.0026] | 0.4958 | 0.0010 | 0.4958 | no |
| paired t over 5 seeds (secondary, df = 4) | +0.0020 | [-0.0054, +0.0094] | – | 0.0131 | 0.0419 | 0.0419 | yes |

Share of bootstrap differences inside ±0.01: 0.504; at or below −0.01: 0.495. Per-seed differences: s42 +0.0053, s43 -0.0087, s44 -0.0000, s45 +0.0126, s46 +0.0009.

## Items 6 and 7

See subset_construction.md and feature_spec.md.
