# ArtFAE-Net

Attribute-guided, evidence-grounded emotion recognition in visual art. Research project at National Taichung University of Education (NTCU), Taiwan.

The framework takes an artwork image, measures numerical visual properties, encodes deterministic label-free descriptions with CLIP, and combines them through Attribute-Guided Fusion (AGF). Emotion-Conditioned Spatial Attention (ECSA) adds class-specific regional contributions directly to the logits. Attention remains candidate evidence; deletion/retention tests assess perturbation sensitivity.

## Verified status

The original baseline, checkpoint, history, and dataset are preserved. Independent CPU validation of the existing CLIP head gives macro-F1 0.303034, accuracy 0.510894, and soft CE 1.791518. This uses 2,249 unique-top artworks for hard metrics and all 3,000 for distributional metrics. The original GPU-fp16 history reports 0.301131 macro-F1; the new feature extraction uses CPU float32. These are validation findings, not held-out test results.

The audit verifies 14,000/3,000/3,000 artwork partitions and 78,795/16,959/16,968 annotations. B3/B4/B5 now have three independent seeds: validation macro-F1 mean ± sample SD is 0.321257 ± 0.003911 / 0.333004 ± 0.001170 / 0.324391 ± 0.001723. Frozen ResNet50 completed 20 CPU epochs, with a highest logged validation macro-F1 of 0.252505; its final independent verification was declined and remains pending. Proposed M1 passed CPU smoke checks; full training is pending the booking. Detailed completion is tracked in [EXPERIMENTS.md](EXPERIMENTS.md) and the [continuation record](docs/CONTINUATION_20261022.md). GPU booking: two full days beginning October 23, 2026, interpreted in the client timezone Asia/Taipei as October 23 00:00 through October 25 00:00 (UTC+8); see `configs/shoal_booking.yaml`. The current implementation session precedes that window.

## Environment and commands

Use `/workspace/.venv/bin/python`; no replacement environment is needed. The commands below run from the repository root. Scripts add `src` to their import path. Exact starting package versions and original artifact hashes are in `docs/phase0/`. Matplotlib was added for plots; no PyTorch or CUDA upgrades are required.

```bash
/workspace/.venv/bin/python -m unittest discover -s tests -v
/workspace/.venv/bin/python scripts/artfae.py --help
/workspace/.venv/bin/python scripts/artfae.py audit --verify-images --output results/my_audit
/workspace/.venv/bin/python scripts/evaluate_existing_clip.py --output results/my_clip_validation
/workspace/.venv/bin/python scripts/artfae.py cache --output results/my_features
/workspace/.venv/bin/python scripts/artfae.py train --cache results/my_features --config configs/B3.yaml --output results/my_b3
/workspace/.venv/bin/python scripts/artfae.py train --cache results/my_features --config configs/m1.yaml --output results/my_m1
/workspace/.venv/bin/python scripts/artfae.py infer --checkpoint results/my_m1/best.pt --image clean_artemis/images/ARTWORK.jpg --output results/my_prediction
/workspace/.venv/bin/python scripts/evaluate_evidence.py --checkpoint results/my_m1/best.pt --image clean_artemis/images/ARTWORK.jpg --output results/my_evidence
/workspace/.venv/bin/python scripts/compare_experiments.py --plots --output results/my_comparison
```

> **Images (2026-11-06):** `data/images/` (the v1 20K image copy) was moved to quarantine in `/workspace/_to_delete/ArtFAE-Net/data/images/` (see `/workspace/_to_delete/MOVES.log`). Use `clean_artemis/images/` (49,011 cleaned images; symlink to `archive/images/`) or the v2 package `/workspace/datasets/ArtEmis-20k-v2/ArtEmis-20k-GPU/images/`. The 4 v1 images that are not in `clean_artemis/images/` are backed up in `data/images_unique_backup/` (also present in the v2 package).

Use a train or validation artwork for demonstrations. Replace `ARTWORK.jpg` with an actual filename; do not select test images during development. Inference writes probabilities, measurements, structured explanations, and ECSA crop-aligned visualizations. The selected trained M1 checkpoint must exist before its inference command can run. A pretrained, untrained proposed model must never be presented as a research result.

All output directories must be new. For CUDA, append `--device cuda --booking-start ISO_TIMESTAMP --booking-end ISO_TIMESTAMP` to supported commands. Both timestamps require timezone offsets and must bound the current time. See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for interruption and resume.

## Repository

- `src/artfae/`: dataset validation, deterministic features, neural models, training, metrics, evidence evaluation, explanations, CLI.
- `scripts/`: entry points, including preserved `artfae_clip_baseline.py`.
- `configs/`: B2–B5, M1, and controlled AGF/ECSA/branch ablations.
- `tests/`: synthetic CPU tests, including training/resume; no test artwork inputs.
- `data/`: original read-only dataset; never committed or copied into output artifacts.
- `results/`: separate run directories, feature shards, audit reports, predictions, checkpoints, and plots.

Read [DATASET.md](DATASET.md), [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md), [RESEARCH_REPORT.md](RESEARCH_REPORT.md), and [COLLABORATION_GUIDE.md](COLLABORATION_GUIDE.md). Artwork images are not licensed for automatic redistribution by this project. No dataset publishing or external upload is performed.

A phase-by-phase execution record, saved artifacts, problems, and exact continuation commands are in [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md).

The current suite passes 38 tests, including exact CPU epoch-replay recovery, selected-checkpoint preservation, explanation consistency, and separation of tuning recipes in seed summaries. All 17,000 cached train/validation attribute vectors pass the raw-measurement checks. A resumable, prespecified validation evidence-study command is documented in [TRAINING_GUIDE.md](TRAINING_GUIDE.md); its one-artwork CPU smoke check completed, while the stable-M1 research study remains pending.

## Jupyter training notebook

Open [ArtFAE_Training_and_Improvement.ipynb](notebooks/ArtFAE_Training_and_Improvement.ipynb) to train the models yourself, compare controlled learning-rate/width trials, repeat seeds, run ablations, resume checkpoints, and inspect validation plots and explanations. Select the **Python (ArtFAE .venv)** kernel. See [notebook instructions](notebooks/README.md) for the first run. Default switches preview commands; GPU execution follows your recorded Shoal booking.

Separate model notebooks for **B1, B2, B3, B4, B5, and M1** are indexed in [notebooks/README.md](notebooks/README.md#separate-notebooks-by-model). B2 reuses its trained checkpoint; the other five provide model-specific training workflows.
# ArtAFE
# ArtAFE
