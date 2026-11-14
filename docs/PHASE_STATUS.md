# Phase completion and continuation record

Later progress supersedes some initial statuses below: see [CONTINUATION_20261022.md](CONTINUATION_20261022.md) for three-seed replications, frozen B1 training, 25 passing tests, original-artwork spatial alignment, the gated final-test implementation, and the declined final B1 verification command.

Session date: October 22, 2026, Asia/Taipei. No CUDA computation was launched. The user supplied an October 23 start and two full booking days; the client timezone interpretation is recorded in `configs/shoal_booking.yaml`.

## Phase 0 — repository and environment audit: completed within the no-test-evaluation boundary

Inspected the existing baseline before adding code. Inventoried dataset manifests, annotations, saved head/history, installed packages, disk, and GPU status. The existing head is 9×512, has the correct class order and backbone, and contains classifier weights rather than the full encoder. Recorded hashes of original script, checkpoints, histories, and log; final verification matches all hashes.

Created `docs/phase0/environment.txt`, `inventory.txt`, `preserved_artifacts.json`, and current dependency record `docs/environment_current.txt`. The working environment remains `/workspace/.venv`; Matplotlib 3.11.2 and its plotting dependencies were added. Official CLIP tokenizer files and pretrained torchvision ResNet50 weights were downloaded; no artwork upload or environment replacement occurred.

Problem: the sandbox cannot start because unprivileged user namespaces are disabled. Shell operations used approved escalation. Manifest `image_path` values are stale account-specific paths; new code resolves configured dataset filenames. Annotation spelling differs for `something else`; the audit applies an explicit in-memory mapping. The first audit's spelling-related false positives are retained and superseded by the corrected audit.

Protocol note: the initial schema preview inadvertently displayed one test annotation record because the combined annotation file began with a test row. It was not used for training, model selection, debugging predictions, or evaluation. Subsequent checks use only test IDs, paths, split metadata, and counts. No test predictions or test metrics were generated. Do not describe the development process as absolutely blind to every test record.

Continue: the structural audit is reproducible with `/workspace/.venv/bin/python scripts/artfae.py audit --verify-images --output results/new_audit`.

## Phase 1 — modular implementation and tests: completed for the implemented scope

Created `src/artfae/{datasets,features,models,training,evaluation,explanations,utils}`, package CLI, YAML configurations, synthetic tests, `pyproject.toml`, `.gitignore`, and research documentation. Added strict soft-label validation, ID disjointness, deterministic measurements, train-only normalizers, image-derived descriptions, atomic checkpoints, partial-cache recovery, and unique output directories.

Final test command: `/workspace/.venv/bin/python -m unittest discover -s tests -v` — 19 tests passed. All nine script entry points passed `--help`. Synthetic two-epoch training, checkpoint/optimizer restoration, cache interruption/recovery, group exclusion, AGF/ECSA gradients, and matched perturbations passed. Full train/validation decoding reported zero image failures in `results/audit_decode_20261022/audit.json`.

Completed feature cache: `results/features_clip_v1_20261022`. Verified 14,000 train and 3,000 validation rows, per-artwork shapes 512 global / 28 numerical / 512 text / 49×768 patches / 9 targets. Global validation embeddings match independent baseline extraction exactly (maximum absolute difference 0.0). Test features were not extracted.

Continue: use this cache for new head and M1 experiments; it does not need regeneration.

## Phase 2 — existing CLIP baseline evaluation: completed

Executed `/workspace/.venv/bin/python scripts/evaluate_existing_clip.py --output results/clip_validation_cpu_20261022` on CPU. Saved metrics, predictions, feature tensors, confusion matrix, and per-class plots/tables. Verified macro-F1 0.303034, accuracy 0.510894, CE 1.791518. Hard metrics use 2,249 artworks and exclude 751 ties. Original historical GPU-fp16 scores are retained separately.

Continue: reuse the original checkpoint; do not retrain or overwrite it.

## Phase 3 — additional baselines: partially completed

Implemented B1/B3/B4/B5. Real-image CPU smoke tests passed for all four, including pretrained ResNet forward/backward and head optimization. Executed `/workspace/.venv/bin/python scripts/smoke_models.py --output results/smoke_models_20261022 --resnet`.

Executed `/workspace/.venv/bin/python scripts/run_cpu_baselines.py --cache results/features_clip_v1_20261022 --output results/cpu_baselines_20261022`. B3 and B4 ran 20 epochs; B5 early-stopped after 18. Their validation macro-F1 values are 0.320339 / 0.332499 / 0.325448. Complete checkpoints, configurations, histories, predictions, metrics, logs, plots, and normalizers are saved. B1 full research training remains pending.

Executed comparison/plotting with 1,000 paired bootstrap replicates; latest consolidated outputs are `results/comparison_final_20261022`. These are single-seed exploratory validation comparisons, with historical B2 budget/head differences explicitly documented.

Continue for B1 during booking: `/workspace/.venv/bin/python scripts/train_resnet.py --output results/b1_seed42_20261023 --device cuda --booking-start '2026-10-23T00:00:00+08:00' --booking-end '2026-10-25T00:00:00+08:00'`. Decide and record full versus frozen ResNet budget before comparing it with frozen CLIP heads.

## Phase 4 — proposed model: implementation and CPU smoke completed; full training pending

Implemented AGF/ECSA with regional logits, optional attention outputs, branch ablations, concatenation/global-only controls, and numerical-group controls. Real-image two-epoch M1 smoke, checkpoint restore, crop-aligned heatmaps/regions, and image-space evidence evaluation passed. Artifacts under `results/smoke_models_20261022` are explicitly not research evidence. Actual trained B2 and B4 inference reports are under `results/b2_inference_20261022` and `results/b4_inference_20261022`.

Exact next commands, inside booking only:

```bash
/workspace/.venv/bin/python scripts/gpu_smoke.py --output results/gpu_smoke_20261023 --booking-start '2026-10-23T00:00:00+08:00' --booking-end '2026-10-25T00:00:00+08:00'
/workspace/.venv/bin/python scripts/artfae.py train --cache results/features_clip_v1_20261022 --config configs/m1.yaml --output results/m1_seed42_20261023 --device cuda --booking-start '2026-10-23T00:00:00+08:00' --booking-end '2026-10-25T00:00:00+08:00'
```

Resume an interrupted M1 into a fresh directory:

```bash
/workspace/.venv/bin/python scripts/artfae.py train --cache results/features_clip_v1_20261022 --config configs/m1.yaml --resume results/m1_seed42_20261023/last.pt --output results/m1_seed42_resumed --device cuda --booking-start '2026-10-23T00:00:00+08:00' --booking-end '2026-10-25T00:00:00+08:00'
```

## Phase 5 — optional M2: pending prerequisite

Stable full M1 has not been trained. No M2 loss, checkpoint, or successful finding is claimed. Image-space perturbation evaluation is implemented and is not represented as a differentiable training objective.

## Phase 6 — final evaluation: pending protocol lock

No test evaluation was run. Final model selection, repeated seeds, research evidence subset, ablation results, and held-out protocol must be locked first. Automated test-report generation is intentionally not exposed without this prerequisite.

## Phase 7 — documentation and reporting: current-session report completed; final research report pending experiments

Created README, DATASET, MODEL_ARCHITECTURE, TRAINING_GUIDE, EXPERIMENTS, COLLABORATION_GUIDE, RESEARCH_REPORT, and this phase record. Saved actual metrics and exploratory paired intervals, while marking B1/M1/M2, repeated seeds, research ablations, and final test results pending. Reviewed baseline confusion-matrix rendering and smoke overlay geometry. Original artwork images remain in their existing directory and are ignored by version control.
