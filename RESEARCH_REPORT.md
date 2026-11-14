# ArtFAE-Net implementation and verified findings

## Scope and method

This implementation supports image-only emotion recognition for artworks with subjective human vote distributions. It preserves the original custom ArtEmis-20k dataset and frozen CLIP baseline. Human utterances are kept as annotation data but never used as inference or training features. Labels retain the exact nine-class baseline ordering.

The implemented pipeline extracts frozen CLIP global and spatial features, 28 deterministic image measurements, and CLIP text embeddings of threshold-based image descriptions. AGF learns projected feature gates; ECSA learns fused-representation-conditioned class queries over visual patches. Global and regional components contribute directly to logits. Neither attention nor gating is assumed to be a faithful explanation.

The package implements soft-label loss, strict split checks, train-only standardization, baseline models, new experiment directories, cached feature shards, booking-aware CUDA entry points, resumable optimization, calibration/distribution metrics, matched-control image perturbations, paired bootstrap comparisons, and conservative structured explanations. Source code is under `src/artfae/`; commands and configs are executable rather than pseudocode.

## Verified evidence

The dataset audit reproduces 20,000 artwork IDs and 112,722 annotations with the prescribed split sizes. Training/validation images decode without reported failures. The annotation spelling `something else` maps explicitly to `something_else`; no original data are changed. Test auditing is limited to structural identifiers and counts. Test predictions and test-label evaluation remain unopened.

Independent B2 CPU validation yields macro-F1 0.303034, accuracy 0.510894, CE 1.791518, KL 0.743326, multiclass Brier 0.184622, and unique-top ECE 0.169517. The original history reports final macro-F1 0.301131 and CE 1.791052 with historical GPU-fp16 features. The new CPU float32 result is close but not bitwise equal. The original checkpoint has a verified 9×512 weight matrix, nine biases, matching backbone ID and class order.

Completed single-seed CPU head training gives macro-F1 0.320339 (B3), 0.332499 (B4), and 0.325448 (B5). Their checkpoints and complete validation reports reside under `results/cpu_baselines_20261022/`. Configured maximum budgets are identical across these three runs, with early stopping explaining B5's 18 epochs. The original B2 training budget, selection metric, and linear head differ from the new baselines, so the measured differences do not establish attribute-specific causal improvements. Full metrics and exploratory paired intervals are documented in `EXPERIMENTS.md`.

Synthetic tests, real-image smoke training, baseline inference, M1 smoke heatmaps, matched perturbations, and checkpoint/resume checks have executed successfully. Smoke scores are not research findings. The actual CUDA smoke has not run because the two-day GPU booking begins October 23, 2026 Taiwan time.

## Limitations and incomplete work

B3/B4/B5 have three independent seeds (42, 43, 44), with macro-F1 means/sample SDs 0.321257/0.003911, 0.333004/0.001170, and 0.324391/0.001723. The original B2 has one seed. Class imbalance is substantial; the B2 confusion matrix shows no correctly classified anger artworks among nine unique-top examples. Majority-class accuracy cannot substitute for macro-F1 or class-specific reporting. Bootstrap intervals are exploratory validation analyses after selection, not final significance tests.

Full M1, B1 full-backbone fine-tuning, research ablations, M2, and actual held-out evaluation remain pending. Frozen B1 completed 20 CPU epochs with highest logged validation macro-F1 0.252505; its final independent checkpoint/raw-image verification was declined and remains pending. A separate immutable-protocol-gated test evaluator is implemented; its gates and complete pipeline are covered by synthetic fixtures, and no project test evaluation has run. M2 is deliberately not represented by an unverified auxiliary loss: differentiable mask geometry, evidence area constraints, ranking controls, and gradient behavior through deterministic image/text attributes still require a justified design after stable M1 training. Evaluation perturbations are implemented, but they do not prove semantic explanation correctness.

Numerical measurements summarize a resized whole-image view; spatial evidence originates in CLIP's center crop. Tested inverse resize/crop geometry now aligns maps with the decoded original artwork. Intervention evaluation recomputes all branches after perturbing inverse-mapped original-image regions. Identifier disjointness does not exclude visually identical images under different identifiers. Cache input signatures use size/mtime and manifest hashes, not pixel-authenticity guarantees. Test image corruption and test distributions are reserved for locked final evaluation.

The last completed epoch is the recovery boundary. Incomplete epochs replay rather than resuming at an exact minibatch. CPU optimizer round trips are tested; exact CUDA numerical reproducibility remains unverified. All original research artifact hashes match their starting values.

## Next authorized step

During the recorded October 23–24 booking, run the CUDA smoke in `TRAINING_GUIDE.md`, then full M1 against the completed shared feature cache. Compare M1 with B5 under compatible training budgets, run controlled ablations and independent seeds, and evaluate evidence on a prespecified validation subset. Lock final selection before test evaluation. No test metric or final novelty/improvement claim is currently justified. The initial schema preview displayed one test annotation record; it was not used for supervision, model selection, or evaluation. This protocol deviation is recorded in `docs/PHASE_STATUS.md`, and absolute test-record blindness is not claimed.
