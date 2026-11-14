# Experiments: completed versus pending

All figures below are validation results on the custom pilot split, not final test results. Every completed research run uses 3,000 validation artworks; hard metrics use 2,249 with unique highest human votes and exclude 751 ties. Distributional metrics use all 3,000. Label order is fixed throughout.

| Model | Run directory under `results/` | Epochs executed | Selected epoch | Macro-F1 | Accuracy | Soft CE |
|---|---|---:|---:|---:|---:|---:|
| B2 preserved CLIP head, independent CPU evaluation | `clip_validation_cpu_20261022` | historical 30 | historical minimum CE | 0.303034 | 0.510894 | 1.791518 |
| B3 CLIP + numerical attributes | `cpu_baselines_20261022/B3` | 20 | 18 | 0.320339 | 0.522454 | 1.776912 |
| B4 CLIP + attribute text | `cpu_baselines_20261022/B4` | 20 | 16 | 0.332499 | 0.522454 | 1.777088 |
| B5 simple three-branch fusion | `cpu_baselines_20261022/B5` | 18, early stopped | 13 | 0.325448 | 0.519787 | 1.778425 |

B3–B5 used seed 42, AdamW LR 3e-4, weight decay 0.01, batch size 32, cosine scheduling, a maximum 20 epochs, patience 5, and validation macro-F1 selection. Frozen features were extracted on CPU in float32. The shared completed cache is `results/features_clip_v1_20261022` (14,000 train and 3,000 validation artworks; full patch tokens retained). No GPU work was performed.

The historical B2 used a linear head, LR 1e-3, 30 epochs, and minimum validation CE selection; B3–B5 use MLP heads and a different optimization/selection budget. Therefore these comparisons do not isolate the causal effect of adding attributes. B2 was not retrained or overwritten. B4 has the highest observed validation macro-F1 among these completed runs; it is not a final selected test model, and repeated-seed robustness is unknown.

## Exploratory paired intervals

`results/comparison_completed_20261022/paired_comparisons.json` uses 1,000 paired artwork bootstrap resamples, seed 42, and unique-top nine-class macro-F1 within each resample.

| Difference relative to independent B2 | Observed delta | 95% percentile interval |
|---|---:|---:|
| B3 − B2 | 0.017305 | [0.002698, 0.032734] |
| B4 − B2 | 0.029465 | [0.011300, 0.049917] |
| B5 − B2 | 0.022414 | [0.005428, 0.039877] |

These are exploratory validation intervals after model selection, with one seed and different head/training protocols. They are not final-test significance claims, do not account for all model-selection uncertainty, and are not evidence of a causal benefit from any single feature branch. No across-seed standard deviation is reported for a single seed.

## Other completed checks

- `results/audit_decode_20261022/audit.json`: recorded counts reproduced, artwork IDs disjoint, no missing paths, no train/validation decoded-image failures, no train/validation annotation distribution mismatches after explicit spelling mapping.
- `results/smoke_models_20261022/`: 32-train/16-validation real-image CPU checks for B3/B4/B5/M1; pretrained ResNet50 forward/backward and head optimizer step; M1 inference/overlay/region output and intervention evaluation. These nested artifacts are excluded from research comparisons.
- `results/b2_inference_20261022/inference.json`: trained B2 image-only inference on a training artwork, with measured attributes and explicit absence of class-conditioned evidence.
- `docs/phase0/preserved_artifacts.json`: original artifact hashes; verification confirms no original script/checkpoint/history/log modifications.
- `results/comparison_completed_20261022/`: comparison CSV, independent-seed summaries, and paired validation intervals. Run directories contain JSON metrics, prediction CSVs, confusion matrices, per-class metrics, and learning curves for trained heads.

## Pending research

B1 full-backbone fine-tuning; booking-time CUDA smoke; full M1; controlled AGF/ECSA/branch and numerical-group ablations; stable-M1 evidence evaluation across a documented validation subset; optional M2 objective design/training; actual locked-protocol held-out testing; final trained M1 explanation demonstration. B3/B4/B5 seed replications are now complete, as recorded below. No paper-quality faithfulness or generalization conclusion is claimed.

The user booked two full days beginning October 23, 2026. Client timezone Asia/Taipei gives the recorded interval October 23 00:00 through October 25 00:00 (UTC+8). `configs/shoal_booking.yaml` records this interpretation. Full M1 is deferred until that interval; commands are in `TRAINING_GUIDE.md`.

## Completed continuation

Seeds 43 and 44 completed under `results/cpu_replications_20261022`. Across seeds 42/43/44, validation macro-F1 mean ± sample SD is **0.321257 ± 0.003911 (B3)**, **0.333004 ± 0.001170 (B4)**, and **0.324391 ± 0.001723 (B5)**. The completed comparison is `results/comparison_three_seeds_20261022`. Every one of these nine checkpoints was verified to reproduce its stored validation probabilities.

Frozen ResNet50 completed 20 CPU epochs in `results/resnet50_frozen_cpu_20261022`; its highest logged validation macro-F1 was 0.252505 at epoch 16. The final independent checkpoint/raw-image verification and all-baseline consolidation command was declined, so those checks remain pending. Full-backbone ResNet fine-tuning has not run.

The suite now passes 25 tests. Original-artwork inverse crop geometry and the separate locked-protocol test evaluator are implemented and synthetically tested; no actual test evaluation was performed. See `docs/CONTINUATION_20261022.md` for execution evidence, remaining limitations, and the declined action.

Further implementation increased the current suite to **33 passing tests**. `results/smoke_validation_study_20261022` verifies the prespecified, batched, resumable validation evidence-study workflow on one artwork; it remains labeled smoke. Recovery tests now prove exact CPU partial-epoch replay and selected-model preservation. No new research-training metrics or actual test results were produced by these checks.
