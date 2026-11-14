# Reproduction and booking-aware training

For interactive use in VS Code, open [notebooks/ArtFAE_Training_and_Improvement.ipynb](notebooks/ArtFAE_Training_and_Improvement.ipynb). It covers training, controlled improvement trials, seed comparisons, ablations, checkpoint resume, and visual inspection. Start with the instructions in [notebooks/README.md](notebooks/README.md); expensive execution switches default to preview mode.

Use the existing `/workspace/.venv/bin/python`. PyTorch/CUDA were not upgraded. `docs/phase0/environment.txt` captures starting versions; `docs/environment_current.txt` records the final environment. No private dataset paths are hard-coded.

1. Run unit tests and audit the dataset.
2. Evaluate the existing B2 checkpoint on validation with `scripts/evaluate_existing_clip.py` (CPU only). Do not rerun the preserved baseline trainer: its original output paths are not overwrite-protected.
3. Create complete train/validation frozen-feature shards using the cache command. Incomplete caches have no final `metadata.json` and cannot be trained. New cache creation requires a fresh directory. Caches created by the current CLI write `progress.json` before encoding and save shards atomically. Resume an interrupted cache by repeating its exact command with `--resume-cache`; manifest hashes, image size/mtime signatures, batch size, and feature version must match. The initial completed cache was created before this recovery feature and does not require recovery.
4. Train B3, B4, B5 with their YAML configurations; hold training budgets and preprocessing fixed.
5. Train M1, then compare validation results and controlled ablations. Keep original B2's cross-entropy checkpoint selection explicit; new models select by unique-top validation macro-F1.
6. Lock model selection, seed policy, hyperparameters, confidence-interval procedure, and test reporting before any held-out test evaluation. There is no test flag in development commands. The separate `scripts/evaluate_locked_test.py` validates a locked JSON protocol before loading any held-out targets.

```bash
/workspace/.venv/bin/python -m unittest discover -s tests -v
/workspace/.venv/bin/python scripts/artfae.py cache --output results/my_feature_cache --batch-size 32
/workspace/.venv/bin/python scripts/artfae.py train --cache results/my_feature_cache --config configs/B3.yaml --output results/b3_seed42
/workspace/.venv/bin/python scripts/artfae.py train --cache results/my_feature_cache --config configs/B4.yaml --output results/b4_seed42
/workspace/.venv/bin/python scripts/artfae.py train --cache results/my_feature_cache --config configs/B5.yaml --output results/b5_seed42
/workspace/.venv/bin/python scripts/artfae.py train --cache results/my_feature_cache --config configs/m1.yaml --output results/m1_seed42
/workspace/.venv/bin/python scripts/train_resnet.py --output results/b1_seed42 --freeze-backbone
```

For GPU commands, supply the actual approved interval, with timezone offsets, to `--booking-start` and `--booking-end`, and set `--device cuda`. The user supplied October 23 and two full booking days. Using the client timezone Asia/Taipei, the recorded interval is `2026-10-23T00:00:00+08:00` through `2026-10-25T00:00:00+08:00`; see `configs/shoal_booking.yaml`. Idle GPU memory does not authorize use.

The cached trainer uses AdamW, cosine scheduling, optional CUDA autocast/GradScaler, finite-loss checks, gradient clipping with nonfinite detection, configurable accumulation, patience-based early stopping, and seed-based shard/in-shard permutations. Numerical normalizers fit only training features. Shards contain one artwork per row, preserving targets and IDs. Cached patches are retained for ECSA; global-only embeddings cannot support spatial learning.

Send SIGINT (Ctrl-C) or SIGTERM for safe interruption. `last.pt` records the last complete epoch; incomplete epochs are discarded and replayed. An initial epoch-0 checkpoint is written before training begins. Output directories are unique; resume into a new one:

```bash
/workspace/.venv/bin/python scripts/artfae.py train --cache results/my_feature_cache --config configs/m1.yaml --resume results/m1_seed42/last.pt --output results/m1_seed42_resumed
```

Keep the config and cache identical on resume; configuration drift is rejected. Checkpoints retain optimizer, scheduler, scaler, CPU and initialized CUDA torch RNG, model configuration, class order, backbone ID, training normalizer, epoch, history, and selection state. Exact bitwise CUDA resume is not yet validated; do not claim it. Per-epoch peak CUDA allocation and elapsed runtime are logged by the cached trainer. Machine eviction can lose an incomplete epoch; persist checkpoints on shared storage within the booking.

## Ablations and evidence

`ablation_no_numerical.yaml`, `ablation_no_text.yaml`, `ablation_concat.yaml`, and `ablation_global_only.yaml` change one mechanism at a time. Individual numerical-group configs are `ablation_attributes_color.yaml`, `ablation_attributes_brightness.yaml`, `ablation_attributes_contrast.yaml`, and `ablation_attributes_complexity.yaml`. They hold the text branch fixed. Their research runs and M2 remain pending. Copy configs into new files to run independent seeds 42, 43, and 44. Compare completed runs:

```bash
/workspace/.venv/bin/python scripts/compare_experiments.py --plots --output results/comparison_new
/workspace/.venv/bin/python scripts/evaluate_evidence.py --checkpoint results/m1_seed42/best.pt --image clean_artemis/images/KNOWN_VALIDATION_ARTWORK.jpg --output results/evidence_new --random-repeats 10
```

> **Images (2026-11-06):** `data/images/` was moved to quarantine (`/workspace/_to_delete/ArtFAE-Net/data/images/`). Pick validation images from `clean_artemis/images/` or `/workspace/datasets/ArtEmis-20k-v2/ArtEmis-20k-GPU/images/`; see `README.md` and `PROJECT_INDEX.md`.

Evidence evaluation uses budgets 10%, 20%, 30%, 50%, mean-color and blur replacement, and matched random region counts. It reports deletion drops and random-control advantages. Images must belong to a permitted non-test demonstration set. Attention overlays alone are not faithfulness evidence. Statistical comparisons must pair artwork IDs before using `paired_bootstrap`; run means and standard deviations require genuinely independent seeds.

## Authorized CUDA smoke and M1 continuation

During the recorded two-day booking, first run:

```bash
/workspace/.venv/bin/python scripts/gpu_smoke.py --output results/gpu_smoke_20261023 --booking-start '2026-10-23T00:00:00+08:00' --booking-end '2026-10-25T00:00:00+08:00'
/workspace/.venv/bin/python scripts/artfae.py train --cache results/features_clip_v1_20261022 --config configs/m1.yaml --output results/m1_seed42_20261023 --device cuda --booking-start '2026-10-23T00:00:00+08:00' --booking-end '2026-10-25T00:00:00+08:00'
```

The smoke command runs frozen CLIP on four training images, three AGF/ECSA optimizer steps, and a checkpoint round trip. It rejects execution before the booking starts. Full M1 training and research ablations remain pending until this check passes inside the booking.

## Final held-out command (not executed yet)

`configs/final_protocol_template.json` is deliberately a rejected draft. After M1, ablations, seed replications, and validation selection are complete, write a separate immutable protocol specifying the exact checkpoint SHA256, validation evidence path/hash, all three manifest hashes, fixed class order, selection rationale, and limitations. Set its status to `locked` only when model selection is finished. `artfae.evaluation.protocol.sha256` computes file hashes without parsing labels. The selection artifact must contain fixed-order validation metrics.

```bash
/workspace/.venv/bin/python scripts/evaluate_locked_test.py --protocol results/locked_final_protocol.json --checkpoint results/SELECTED_RUN/best.pt --output results/final_test_selected
```

This command passed `--help`, synthetic guard tests, and a complete synthetic end-to-end evaluation fixture. It has not evaluated the project test set. Draft, changed-checkpoint, altered-manifest, missing-selection-evidence, and smoke-checkpoint conditions reject evaluation. Validation evidence must itself contain the selected checkpoint's exact hash. Test results are written as `test_metrics.json` and `test_predictions.csv`, separately from validation. CUDA requires the same explicit booking flags. The frozen ResNet path is distinguished from a fully fine-tuned ResNet through saved configuration.

## CPU seed replications and frozen ResNet

```bash
/workspace/.venv/bin/python scripts/run_cpu_baselines.py --cache results/features_clip_v1_20261022 --output results/new_replications --seeds 43 44
/workspace/.venv/bin/python scripts/train_resnet.py --output results/new_frozen_resnet --freeze-backbone --cache-features
```

The ResNet option computes each image feature once with the selected pretrained transforms, then trains only the linear head. It preserves raw 2,048-dimensional features, artwork IDs, and soft targets in `frozen_features.pt`. The full model state in its checkpoints remains compatible with raw-image evaluation. Full-backbone fine-tuning remains a separate experiment.

## Prespecified validation evidence study

After stable M1 training, evaluate a fixed validation-only sample. Artwork IDs are selected before inference using the manifest and seed, without human labels or model scores. The command has no test-split option. Every perturbation recomputes the image embedding, measured attributes, attribute text, and patch features. Random controls retain the same number of selected patches, and reports record both crop and original-image area.

```bash
/workspace/.venv/bin/python scripts/evaluate_validation_evidence.py --checkpoint results/m1_seed42_20261023/best.pt --output results/m1_validation_evidence_20261023 --sample-size 100 --seed 42 --random-repeats 10 --batch-size 16 --bootstrap 1000 --device cuda --booking-start '2026-10-23T00:00:00+08:00' --booking-end '2026-10-25T00:00:00+08:00'
```

The default budgets are 10%, 20%, 30%, and 50% of encoder crop patches; methods are mean-color replacement and blur. The relevant class is the model's prediction on the unperturbed original artwork. Predictions after deletion/retention are batched. Artwork-level percentile bootstrap intervals summarize top-versus-random deletion and retention advantages. They are exploratory sensitivity intervals, not proof of semantic correctness or a final-test significance claim. Cross-model comparisons must account for possibly different predicted classes.

Each completed artwork is saved atomically. Repeat the exact study command with `--resume` and the **same output directory** to recover; this differs from training, which resumes into a new directory. Checkpoint, manifest, selection, image signatures, source/configuration, and study parameters must match. Outputs include `study.json`, per-artwork JSON, completion status, aggregate `faithfulness.json`, reproducibility metadata, and a quantitative plot.

An end-to-end CPU smoke study completed at `results/smoke_validation_study_20261022`, using one validation artwork, two budgets, one random control, and an explicitly allowed smoke checkpoint. It is an implementation check, not a research evidence result. Smoke checkpoints require `--allow-smoke` and are limited to at most four artworks and two random controls.

## Selected-model preservation during resume

New `last.pt` checkpoints embed the full selected-model snapshot with its matching optimizer/scheduler/scaler state. Resuming into a new directory preserves its `best.pt` even if no later epoch improves. Existing older runs remain supported when their original sibling `best.pt` is available and matches the interrupted run; a missing or mismatched legacy selection is rejected instead of silently replacing it with the latest weights. An already early-stopped run is not given extra epochs merely by resuming.

CPU regression tests verify that discarding a partial epoch and replaying it from the last complete checkpoint reproduces uninterrupted weights, RNG state, scheduler state, and training losses exactly. Equivalent selected-model preservation is tested for the ResNet trainer with synthetic weights. Exact CUDA recovery remains pending the booking-time check.

Separate model notebooks for **B1, B2, B3, B4, B5, and M1** are indexed in [notebooks/README.md](notebooks/README.md#separate-notebooks-by-model). B2 reuses its trained checkpoint; the other five provide model-specific training workflows.
