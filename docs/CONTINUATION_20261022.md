# Continuation: additional verified progress

This record supersedes the initial phase report where later work changed the status. The previous goal turn made implementation and experimental progress. This continuation also made concrete progress; the GPU booking still has not begun.

## Three-seed baselines completed

Executed:

```bash
/workspace/.venv/bin/python scripts/run_cpu_baselines.py --cache results/features_clip_v1_20261022 --output results/cpu_replications_20261022 --seeds 43 44
/workspace/.venv/bin/python scripts/compare_experiments.py --output results/comparison_three_seeds_20261022 --bootstrap 1000
```

All six additional CPU runs completed. Together with seed 42, the results are:

| Variant | Independent seeds | Validation macro-F1 mean | Sample SD |
|---|---:|---:|---:|
| B3 | 3 | 0.3212571408 | 0.0039114103 |
| B4 | 3 | 0.3330041318 | 0.0011697735 |
| B5 | 3 | 0.3243912278 | 0.0017234030 |

These are validation results with the previously documented historical B2 training-budget/head differences. Nine completed B3/B4/B5 checkpoints were independently loaded and verified to reproduce their saved validation probability CSVs. Their validation reports now include the verified checkpoint SHA256. No test predictions were generated.

## Additional implementation

- Added `src/artfae/evaluation/protocol.py`, `scripts/evaluate_locked_test.py`, and a rejected-by-default draft protocol template. The evaluator checks a locked validation-selection protocol, exact checkpoint hash, hash-bound validation evidence, class order, and fixed manifest identities before loading held-out targets. The complete evaluator passed an end-to-end synthetic fixture test. It has not run against project test data.
- Added `features/frozen.py` and `--cache-features` to the ResNet trainer. Frozen features match direct encoder outputs in a synthetic test. Only a batch of image pixels is decoded at a time.
- Added inverse CLIP preprocessing geometry for original-artwork heatmaps, region masks, patch boxes, and perturbations. Tests match the installed image processor on square, portrait, and landscape fixtures. Real-image smoke inference and evidence evaluation completed under `results/smoke_geometry_20261022` and `results/smoke_geometry_evidence_20261022`.
- Added quantitative intervention plots. These remain perturbation-sensitivity measures, not proof of semantic correctness.
- Rejected unsupported model variant names, including M2, to prevent an ordinary classification model from being mislabeled faithfulness-aware.

The latest automated suite passed **25 tests**. Added final-evaluation CLI help also passed. No GPU computation was performed.

## Frozen B1 training completed; final independent check pending

Executed:

```bash
/workspace/.venv/bin/python scripts/train_resnet.py --output results/resnet50_frozen_cpu_20261022 --freeze-backbone --cache-features
```

The process completed 20 epochs on CPU. The highest logged validation macro-F1 was **0.252505 at epoch 16**. Features cover the fixed train/validation splits, and the trainer writes model checkpoints, histories, predictions, and plots. This is a frozen-backbone ResNet condition, not full-backbone fine-tuning.

The final command to independently verify B1's checkpoint against cached predictions and raw-image inference, then consolidate all baselines, was **declined by the user at command approval**. It was not retried. Therefore that final independent verification and the all-baseline comparison remain pending; no claim that those checks passed is made.

The previously completed three-seed comparison remains available at `results/comparison_three_seeds_20261022`. The declined combined command did not create `results/comparison_all_baselines_20261022`.

## Remaining work and exact continuation

First complete the declined independent B1 verification if subsequently authorized. Then a fresh all-baseline comparison can be generated with:

```bash
/workspace/.venv/bin/python scripts/compare_experiments.py --plots --output results/comparison_all_baselines_20261022 --bootstrap 1000
```

During the recorded October 23–24 Taiwan-time booking, run the CUDA smoke and M1 commands in `TRAINING_GUIDE.md`. Full M1, its research ablations and seed replications, stable-M1 evidence studies, optional M2, and actual locked-protocol held-out evaluation remain incomplete. B1 full-backbone fine-tuning is also a separate uncompleted experiment. Do not run actual test evaluation merely because its command now exists.

## Further continuation: evidence study and recovery correctness

The next goal turn made additional concrete progress without retrying the declined B1 verification. Implemented `scripts/evaluate_validation_evidence.py` and `evaluation/study.py`: prespecified validation artwork sampling, batched interventions, per-artwork atomic recovery, original-image geometry, and artwork-level bootstrap summaries/plots. Synthetic tests verify label/order-independent sampling, scalar/batched parity, group consistency, and interruption recovery.

Executed a real CPU smoke study:

```bash
/workspace/.venv/bin/python scripts/evaluate_validation_evidence.py --checkpoint results/smoke_models_20261022/M1/best.pt --output results/smoke_validation_study_20261022 --sample-size 1 --random-repeats 1 --budgets 0.1 0.2 --batch-size 4 --bootstrap 100 --allow-smoke
```

The command completed one validation artwork with four method/budget groups, correctly marked scope `smoke`, and generated the quantitative plot. No scientific faithfulness result is claimed from it.

Fixed a substantive resume defect: a new output directory could previously lose the earlier selected model if remaining epochs never improved. New last checkpoints embed the actual selected-model snapshot, including its matching optimizer state. Resuming preserves the selected best checkpoint and history. Legacy recovery requires a matching original sibling best checkpoint. The cached trainer and ResNet trainer received the fix; only synthetic ResNet weights were used to test it, and the declined research-checkpoint verification remains pending.

The current full automated suite passes **33 tests**. A synthetic interrupted-epoch test reproduces uninterrupted CPU weights, scheduler state, RNG state, and training losses exactly. GPU numerical recovery and full M1 training remain pending the authorized booking.

## Further continuation: explicit explanation consistency

The next requirements audit identified missing explicit attribute/region consistency validation in explanations. Implemented physical-range and proportion checks, finite measurement dimensions, selected-region class agreement, duplicate-patch rejection, region bounds, and normalized selected attention mass. The report records which checks were actually possible. Added a grounded geometric description of the highest-ranked candidate region and normalized predictive entropy/top-two probability margin, with explicit limits on interpretation.

The full suite now passes **37 tests**. A sweep over all **17,000 train/validation cached attribute vectors** passed the stricter checks; evidence is in `docs/explanation_attribute_verification.json`. Enhanced one-artwork CPU inference completed at `results/smoke_explanation_checks_20261022`. No test targets or declined ResNet checkpoint were opened by these checks. The booking remained in the future when rechecked at 2026-10-21 22:58 UTC (October 22 morning in Taiwan), so no GPU computation was launched.

## Blocking audit after available independent work

The preceding goal turn was progress: explanation checks, regression tests, and a 17,000-artwork attribute sweep changed authoritative evidence. The subsequent prerequisite-only audit produced no new research result and is not classified as a live-job wait.

At **2026-10-21 23:05:43 UTC / October 22 07:05:43 Asia/Taipei**, the authorized window remained October 23 00:00 through October 25 00:00 Taiwan time. The same future-booking restriction was revalidated across consecutive goal turns at 22:07, 22:42, 22:58, and 23:05 UTC. CPU-independent implementation and verification progressed during those turns; the remaining research dependencies now require that external time condition to change.

No full research M1 checkpoint was found among the completed model outputs. CUDA smoke, full M1, its controlled research ablations and seed replications, and stable-M1 evidence results remain pending. Optional M2 depends on stable M1. Actual held-out evaluation and the final selected-model demonstration depend on completed validation selection. The declined independent B1 verification/all-baseline consolidation was not retried and requires renewed authorization. Full-backbone ResNet fine-tuning is an optional separate condition, not a prerequisite for the already trained frozen B1 baseline.

The goal is therefore blocked rather than complete. No future job was scheduled and no long-lived process is being treated as a booking wait. Resume the goal during the authorized window, beginning with the booking-gated CUDA smoke command in `TRAINING_GUIDE.md`, then full M1. The scope remains the complete research objective, including the unfinished experiments and final reporting.
