# ResNet50 gap — reviewers 6, 10 and 12

**NOT ADDRESSED — No same-protocol ResNet50 evaluation exists in the searched saved artifacts.**

Located `results/resnet50_frozen_cpu_20261022/`: best/last checkpoints, `frozen_features.pt`, predictions, class/confusion reports, history, configuration and reproducibility record; also `results/notebook_validation/separate_models/B1_ResNet50.ipynb` and smoke output. The configuration uses `data_dir=data`, seed 42, frozen/cache CPU mode and learning rate 0.0001. Its validation result has 3,000 artworks and **2,249** unique-top cases, whereas current development validation has **2,243**. It is a different earlier dataset and selection recipe, not five-seed evaluation on the 30,280 holdout. Its scores are not inserted into the paper. The feature cache was located but not loaded as a compatible cache; a file's existence does not establish row identity.

The current frozen CLIP linear probe remains a valid same-feature/same-population comparison. The discovered five-seed L/14 B5 concatenation MLP is compatible as an existing **post-hoc** holdout comparator: softCE 1.8344 and macro-F1 0.3612. `runs/final_eval_v3_posthoc/baselines/summary.json`, per-seed CSV and saved logits support it; maximum saved validation-logit reproduction discrepancy is zero. It is not ResNet50, is outside H1–H5, and is not added to the fixed main table. A brief descriptive comparison was added in Section 7.1.

## Minimal scientifically valid ResNet plan — not run

1. Pin an ImageNet-pretrained ResNet50 checkpoint and encoder-specific preprocessing, freeze the encoder and cache pooled 2,048-dimensional features for the **unchanged** 43,012 training and 3,000 validation identities. Record image IDs/cache order/hashes. Do not reuse the old cache without proving exact row membership and preprocessing.
2. Train an affine 2,048→9 head (18,441 parameters) with soft targets and the same unweighted CE objective, five seeds 42–46, and validation-softCE selection. Use validation to choose a sufficient probe budget; identical short budgets can undertrain a linear probe. Predeclare any small LR search and an upper budget, for example 100 epochs/patience 10, rather than retroactively picking a holdout winner.
3. Use all-artwork CE/Brier and unique-top nine-class macro-F1, raw probabilities, no arbitrary tie-breaking. Report parameter counts, seed variability and selection protocol. Validation is a selection split, not independent final evidence.
4. A same-holdout comparison would require a separately authorized **new exploratory evaluation** of its fixed checkpoints on the unchanged holdout; it cannot be retrospectively added to the internally locked H1–H5 family. Under current restrictions this step cannot be executed.

## Resource estimate, not a benchmark

Training+validation extraction: 46,012 encoder forwards. With batch 32 separately by split: 1,345 train + 94 validation = 1,439 batches. FP32 features: 46,012×2,048×4 = 376,930,304 bytes (359.47 MiB), excluding encoder weights, activations and I/O. An optional authorized holdout pass adds 30,280 forwards, 947 batches and 248,053,760 bytes (236.56 MiB). One 100-epoch head fit has at most 134,500 train updates; five seeds at most 672,500, excluding an LR search and early stopping. CPU extraction is possible in principle but no wall-time/RAM/VRAM guarantee is established. No training, feature extraction or holdout evaluation was run.

Action F4; manuscript Sections 5.1 and 7.3.
