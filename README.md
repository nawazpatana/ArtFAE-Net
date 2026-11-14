# ArtFAE-Net

**Do measurable attributes and spatial attention help? A pre-registered study of artwork emotion distribution prediction with frozen CLIP.**

ArtFAE-Net predicts the distribution of emotions that viewers report for a visual artwork (the nine ArtEmis classes: amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something else). It combines a frozen CLIP ViT-L/14@336 image embedding with numerical visual attributes and label-free attribute text through Attribute-Guided Fusion (AGF), and adds class-specific regional evidence with Emotion-Conditioned Spatial Attention (ECSA).

The main question was whether these components help. All architecture choices were made on validation data, then five hypotheses (H1–H5) were pre-registered (`runs/final_selection_v3.txt`, with SHA-256 hashes of every checkpoint, manifest and script) and tested once on a fresh holdout that no model had seen.

## Results: pre-registered holdout

Fresh holdout: 30,280 artworks (22,351 with a unique top emotion). Five seeds (42–46) per model; the primary contrast uses the per-seed mean. Metric: soft cross-entropy (softCE) against the human vote distribution, lower is better. Equivalence margin ±0.005. 95% bootstrap intervals; Holm-corrected p-values. Source: `runs/final_eval_v3/report.md`.

| Hypothesis | Contrast (A − B) | Δ softCE | 95% CI | Holm p | Supported |
|---|---|---|---|---|---|
| H1: the attribute inputs are redundant (equivalence) | NO_ATTR − R1 | +0.0017 | [+0.0012, +0.0021] | 0.0025 | yes |
| H2: AGF-only is better than R1 | AGF_ONLY − R1 | +0.0036 | [+0.0024, +0.0047] | 1.0000 | no |
| H3: ECSA trades softCE for macro-F1 | R1 − NO_ECSA | −0.0023 | [−0.0034, −0.0013] | 1.0000 | no |
| H4: class weighting trades fidelity for minority recognition | WEIGHTED − R1 | +0.0019 (Δ macro-F1 +0.0176) | [+0.0011, +0.0027] | 0.0025 | yes |
| H5: a linear probe matches the full model (equivalence) | PROBE − R1 | +0.0002 | [−0.0009, +0.0013] | 0.0025 | yes |

H1 and H5 also use 90% intervals for equivalence: H1 [+0.0013, +0.0021], H5 [−0.0007, +0.0011].

Per-seed means on the full holdout:

| Model | softCE | KL | Brier | macro-F1 | mean anger+fear AP | neg-sentiment F1 |
|---|---|---|---|---|---|---|
| R1 (ArtFAE-Net, 762,249 parameters) | 1.8357 | 0.7249 | 0.1702 | 0.3607 | 0.4522 | 0.6072 |
| NO_ATTR | 1.8374 | 0.7266 | 0.1707 | 0.3620 | 0.4538 | 0.6030 |
| AGF_ONLY | 1.8393 | 0.7285 | 0.1706 | 0.3624 | 0.4512 | 0.6052 |
| NO_ECSA | 1.8380 | 0.7273 | 0.1707 | 0.3633 | 0.4530 | 0.6077 |
| WEIGHTED | 1.8376 | 0.7268 | 0.1703 | 0.3783 | 0.4517 | 0.6362 |
| PROBE (linear, 6,921 parameters) | 1.8359 | 0.7252 | 0.1704 | 0.3558 | 0.4522 | 0.5861 |
| MLP | 1.8363 | 0.7255 | 0.1704 | 0.3599 | 0.4522 | 0.6014 |
| R3 (balanced sampler + presence head) | 1.8449 | 0.7342 | 0.1731 | 0.3695 | 0.4495 | 0.6366 |

In short: a linear probe on the frozen CLIP embedding predicts the emotion distribution as well as the full model, the attribute inputs add almost nothing, and the two architecture effects that looked beneficial on validation (H2, H3) reversed on the holdout. Class weighting buys macro-F1 at a measurable softCE cost. Ensemble results, the artist-holdout subset (1,245 artworks by artists absent from training) and seed variability are in the same report; post-hoc analyses are in `runs/final_eval_v3_posthoc/`.

## Data (not included)

This repository contains **no images, no ArtEmis annotations and no derived label files**. Obtain them yourself and accept their terms:

- **ArtEmis** (Achlioptas et al., CVPR 2021): emotion labels and explanations for WikiArt artworks. Download `artemis_dataset_release_v0.csv` from <https://www.artemisdataset.org/> after accepting its terms of use (non-commercial research use). Place it at `artemis_official_data/official_data/`.
- **WikiArt images**: the artworks referenced by ArtEmis. Many are still under copyright and belong to their rights holders. WikiArt's terms of use apply. Use the image collection referenced by the ArtEmis instructions. Put the images under `archive/images/` (`clean_artemis/images` is a symlink to it).

Splits and manifests are rebuilt from these sources (see Reproduce). Image file names, SHA-256 hashes and perceptual hashes in `runs/` let you check that your copy matches ours.

## Reproduce

Python 3.12. Use `HF_HUB_OFFLINE=1` after the first CLIP download (`openai/clip-vit-large-patch14-336`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .                      # src/artfae (torch, transformers, scikit-learn, …)
python -m unittest discover -s tests  # synthetic CPU tests, no data needed
```

1. **Dataset cleaning and balancing**: `build_balanced_artemis.py`, `artfae_balance/`.
2. **Frozen CLIP features**: `scripts/run_extract_l14.sh` (L/14@336) and `runs/revision_B/code/extract_holdout_features.sh` for the holdout.
3. **Controlled ablations (validation)**: `runs/revision_B/code/train_abl.py --step 1 --conditions R1 probe … --seeds 42 43 44 45 46`; checkpoints are selected by validation softCE.
4. **Fresh holdout**: `runs/revision_B/code/build_fresh_holdout.py` (deduplication against all previously used artworks with `phash.py`).
5. **Pre-registered evaluation**: `runs/revision_B/code/eval_holdout.py --selection runs/final_selection_v3.txt …`, and `posthoc_holdout_metrics.py`.

`PROJECT_INDEX.md` maps every folder to the claim it supports. The hashes in `runs/final_selection*.txt` record the exact files used. Checkpoints (`*.pt`), features (`*.npy`/`*.npz`) and logits are not in the repository.

**Training checkpoints are not included in this repository; they will be released on Hugging Face.** The exception is the demo's ten small heads (R1 and R3, seeds 42–46, 30.6 MB in `app/assets/models/`), which the online demo needs. The CLIP backbone is downloaded from Hugging Face at a pinned revision.

## Demo app

`app/` is a bilingual (English / 繁體中文) Streamlit app. It shows the predicted emotion distribution, the measured attributes and ECSA attention maps for an uploaded image. The default model is R1, and R3 is available as a minority-sensitive option. Each model runs as a five-seed ensemble. Everything runs locally and uploaded images are not stored. The demo's ten heads (`app/assets/models/*/s*/best.pt`, 30.6 MB) are included; CLIP ViT-L/14@336 is downloaded on first use. Setup: `app/README.md`.

The online version on Streamlit Community Cloud (`streamlit_cloud/streamlit_app.py`, `APP_MODE=cloud_live`) shows the 12 examples with precomputed results and analyses uploads live on the CPU with CLIP's vision tower only (about 1.7 GB of memory). A torch-free **gallery mode** (`APP_MODE=gallery`) shows the examples only. See `app/README.md`.

> **Copyright note:** the example images in `app/assets/examples/` are WikiArt artworks. Their copyright status must be checked before this repository is ever made public.

## Study app (code only)

`study/` holds a Streamlit app for a human pilot study that compares viewers' feelings with the model's predictions. The participant information and consent texts are in English and Chinese. **No participant data, stimulus images or precomputed outputs are included.** `study/private/` and the stimulus images are excluded. `study/frozen.py` verifies the frozen stimulus set by SHA-256 and refuses to start on any difference, so the app runs only with the original stimulus files. Details: `study/README.md`.

## Repository layout

| Path | Contents |
|---|---|
| `src/artfae/` | model (AGF, ECSA), features, training, metrics, explanations |
| `scripts/`, `configs/`, `notebooks/` | entry points, experiment configs, training notebooks |
| `artfae_balance/`, `build_balanced_artemis.py` | dataset cleaning and balancing |
| `runs/` | selection files with hashes, evaluation reports, revision code (`revision_A/`, `revision_B/`) |
| `remaining_experiments/` | reviewer follow-up experiments (faithfulness, permutation, resources) |
| `app/`, `artfae_demo/` | demo app and an older installed copy |
| `study/` | human study app (code only) |
| `tests/` | unit tests |
| `docs/` | phase records and guides |

## Authors

- **Nawazish Hussain**, AI New Media Lab, Department of Digital Content and Technology, National Taichung University of Education (NTCU), Taiwan
- **Assoc. Prof. Shih-Yun Lu** (supervisor), AI New Media Lab, Department of Digital Content and Technology, NTCU

Private repository. Please do not redistribute the code or results without the authors' permission.
