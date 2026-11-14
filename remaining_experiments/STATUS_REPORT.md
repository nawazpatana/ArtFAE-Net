# Remaining experiments — final status

All new work is **exploratory**. The pre-execution protocol, predetermined validation rows, cost/risk report, checkpoint identities, source-image hashes, per-artwork responses, code, statistics and plots are retained here. Validation selected checkpoints and contains known training reproductions; the 200-image subset includes 2. Original results/H1–H5 are unchanged. No dataset changes, training, new holdout model inference, external messages or submission occurred.

| Task | Status | Completed evidence / limitation |
|---|---|---|
| A | COMPLETED: exploratory quantitative sensitivity | 200 fixed validation paintings; R1 seed42; ECSA, ten random rankings and 256-mask RISE-style; mean/blur; raw-probability partial AUC and paired-artwork intervals. Not causal/human-semantic explanation validation; one checkpoint, finite RISE masks, unadjusted intervals. |
| B | COMPLETED: exploratory paired permutation | Five fixed checkpoints, all 3,000 validation rows, 20 derangements; CE/F1, prediction changes and conditional paired CE interval. Tests fixed-model misalignment, not retraining benefit. |
| C | COMPLETED: matched implementation benchmark | R1/probe, same RTX5090, 32 images, batch8/FP32, component/full latency, throughput, peak CUDA allocated/reserved and process RAM. Not universal performance or minimal stripped-probe memory. |
| D | COMPLETED: descriptive saved-result analysis | 64 model/group summaries using existing logits and metadata only; style/artist/tie composition limitations. No external-domain or corruption evidence. |
| E | PROTOCOL PREPARED; BASELINE RESULT NOT ADDRESSED | No compatible ResNet50 found. BASELINE_PROTOCOL.md specifies a frozen encoder/head plan and compute/cache estimates. No training run. |
| F | VERIFIED EXPLORATORY FINDINGS ONLY | New tables/prose reflect checked results; accepted title, author order, original main table and H1–H5 preserved. Publisher layout unchanged. See paper build/PDF_QC.json for final page/build checks. |

## Measurements

### Partial AUC

| fill | curve | ranking | mean_partial_AUC | CI95_low | CI95_high |
|---|---|---|---|---|---|
| mean | deletion | ECSA | 0.2199 | 0.2065 | 0.2336 |
| mean | deletion | RISE | 0.2301 | 0.2165 | 0.2440 |
| mean | deletion | Random | 0.2519 | 0.2395 | 0.2643 |
| mean | retention | ECSA | 0.2581 | 0.2458 | 0.2713 |
| mean | retention | RISE | 0.2480 | 0.2347 | 0.2619 |
| mean | retention | Random | 0.2009 | 0.1894 | 0.2126 |
| blur | deletion | ECSA | 0.2015 | 0.1891 | 0.2139 |
| blur | deletion | RISE | 0.2148 | 0.2017 | 0.2284 |
| blur | deletion | Random | 0.2386 | 0.2270 | 0.2510 |
| blur | retention | ECSA | 0.2547 | 0.2418 | 0.2682 |
| blur | retention | RISE | 0.2401 | 0.2271 | 0.2538 |
| blur | retention | Random | 0.1967 | 0.1855 | 0.2087 |

Paired advantages, positive favouring the named ranking:

| fill | curve | contrast | mean | CI95_low | CI95_high |
|---|---|---|---|---|---|
| mean | deletion | ECSA_advantage_over_random | 0.0320 | 0.0240 | 0.0400 |
| mean | deletion | RISE_advantage_over_random | 0.0218 | 0.0149 | 0.0283 |
| mean | deletion | ECSA_advantage_over_RISE | 0.0102 | 0.0037 | 0.0171 |
| mean | retention | ECSA_advantage_over_random | 0.0572 | 0.0492 | 0.0657 |
| mean | retention | RISE_advantage_over_random | 0.0471 | 0.0397 | 0.0547 |
| mean | retention | ECSA_advantage_over_RISE | 0.0101 | 0.0024 | 0.0178 |
| blur | deletion | ECSA_advantage_over_random | 0.0372 | 0.0298 | 0.0449 |
| blur | deletion | RISE_advantage_over_random | 0.0238 | 0.0171 | 0.0302 |
| blur | deletion | ECSA_advantage_over_RISE | 0.0133 | 0.0068 | 0.0199 |
| blur | retention | ECSA_advantage_over_random | 0.0580 | 0.0498 | 0.0666 |
| blur | retention | RISE_advantage_over_random | 0.0434 | 0.0363 | 0.0502 |
| blur | retention | ECSA_advantage_over_RISE | 0.0146 | 0.0073 | 0.0219 |

### Paired numerical/text permutation

Original CE 1.7547, macro-F1 0.3576; permuted CE 1.7606, macro-F1 0.3520. CE difference +0.0058, paired-artwork 95% interval [0.004410098179547378, 0.007290444907728912]; top1 flip fraction 0.0671; mean total variation 0.0376; same-template fraction after pairing 0.1781.

### Matched resources

| model | features_ms_per_image | complete_ms_per_image | head_ms_per_image | images_per_second | peak_CUDA_allocated_GiB | peak_CUDA_reserved_GiB | peak_process_RSS_GiB |
|---|---|---|---|---|---|---|---|
| R1 | 13.0543 | 16.0173 | 0.0384 | 62.4325 | 1.9206 | 2.0625 | 2.5383 |
| PROBE | 12.5331 | 15.5190 | 0.0040 | 64.4372 | 1.9178 | 2.0566 | 2.5341 |

## Unresolved and approval items

No same-protocol ResNet50, official-split comparison, human-region validation, matched native-resolution comparison, external collection/corruption test or cross-hardware benchmark is established. Historical audit files and exact accepted/reference/review documents remain unavailable. Expanded revision, accepted authorship metadata, title wording, related-manuscript publication overlap, figure rights, AI-use disclosure and any supplementary-material submission need author/organizer decisions. New experiments do not erase overlap in the main dataset, hypotheses or results. No similarity or AI-screening score is asserted.

## Reproduction

Use the original repository and pinned original artifacts; data/checkpoints are not copied into the source ZIP. `prepare.py` is a historical preparation step and must not be rerun over this completed directory. The inference scripts were run once under the locked protocol. `descriptive.py` uses saved predictions only; `analyse.py` and `verify_results.py` require completed outputs and do not load models. All new outputs are under remaining_experiments; no modified painting was saved to the dataset. A rerun should use a separate copied experiment directory, preserve existing results and repeat hash checks. Optional supporting artifacts require organizer permission before submission.
