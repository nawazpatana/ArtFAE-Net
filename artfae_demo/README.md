> **WARNING: Do NOT extract this bundle inside the ArtFAE-Net repository; it would overwrite tracked files
> with this snapshot. Extract it in a separate folder.**

Built from commit 9e11793 on 2026-11-14.

# ArtFAE-Net demo (English / 繁體中文)

A local Streamlit app that estimates how viewers might feel about an artwork, from CLIP ViT-L/14@336
features. The sidebar offers two models, both selected by validation soft cross-entropy:

- **Human-vote distribution (L/14 R1, recommended)**, the default and the paper's recommended model.
  An optional prior correction ("Match validation class frequencies") is off by default.
- **Minority-sensitive (L/14 R3)**: balanced sampler + presence head (λ = 0.3), raw outputs, no prior
  correction. Only this mode shows "Emotions viewers may also feel" (calibrated presence estimates).

All five trained seeds (42–46) of the chosen model run on every image and their probabilities are
averaged (a seed ensemble).

Everything runs on your computer. Uploaded images are not stored or sent anywhere. The only network
access is the one-time CLIP download and the web fonts.

## Setup

You need Python 3.12; 3.10 and 3.11 should also work. Unpack the bundle and work from its top folder.

**macOS / Linux**

```bash
tar -xzf app_bundle.tar.gz
cd artfae_demo
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Windows (PowerShell)**

```powershell
tar -xzf app_bundle.tar.gz
cd artfae_demo
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens at <http://localhost:8501>. Always start it from the `artfae_demo` folder, so that
Streamlit finds `.streamlit/config.toml`, which holds the theme.

**Linux laptop without an NVIDIA GPU:** `pip install torch==2.13.0` on Linux downloads the large CUDA
build. To get the much smaller CPU-only build, install torch first:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On macOS and Windows, the default torch wheel already works without a GPU.

## First run

- **CLIP download.** The first analysis downloads the image/text encoder
  `openai/clip-vit-large-patch14-336`, pinned to revision `ce19dc91`. It's about **1.7 GB**
  (1.6 GiB, the `pytorch_model.bin` plus tokenizer files) and is saved in the Hugging Face cache
  (`~/.cache/huggingface`). Later runs start offline.
- **Model loading** takes a few seconds and happens once per app session.
- **Time per image** is reported under each result. Measured with 8 CPU threads on a desktop
  Ryzen 9 9900X, the full pipeline (CLIP plus 5 seeds) takes about 0.6 s per image; with 4 threads
  it's about 1.1 s. On a typical laptop CPU, expect about **1–3 s per image**. On an NVIDIA GPU it
  takes under 0.05 s.
- The app uses the GPU automatically if PyTorch can see one; otherwise it uses the CPU.

## App modes (`APP_MODE`)

`APP_MODE` is read from the environment, then from Streamlit secrets; the default is `full`.

| mode | examples | uploads | needs |
|---|---|---|---|
| `full` (default) | live | live, full CLIP (GPU if available) | `requirements.txt`, weights |
| `cloud_live` (Streamlit Cloud) | precomputed | live on the CPU, CLIP **vision tower only** | `streamlit_cloud/requirements.txt` (CPU-only torch), weights |
| `gallery` | precomputed | disabled | streamlit, numpy, pandas, pillow, plotly (no torch) |

- **Precomputed examples** (`assets/precomputed/gallery.json`, built and verified against live inference by
  `app/precompute_gallery.py`): both models, the R1 prior correction, the 10-viewer counts, R3 presence chips,
  attention maps for all nine emotions, attributes, description and sentiment.
- **cloud_live** loads `CLIPVisionModelWithProjection` (the same image weights and projection, fp32, CPU) instead of the
  full CLIP, and looks the description's text embedding up in `assets/text_embeddings.npz` (all 27 possible
  sentences, `app/build_text_embeddings.py`). Its outputs equal the full pipeline on the CPU bit for bit
  (`app/verify_cloud_live.py`, 20 validation images). Peak memory is about 1.7 GB.
- Streamlit Community Cloud runs `streamlit_cloud/streamlit_app.py` (sets `APP_MODE=cloud_live`) with
  `streamlit_cloud/requirements.txt`; `APP_MODE=gallery` in the environment switches it to the torch-free version.

```bash
APP_MODE=cloud_live streamlit run app/streamlit_app.py
APP_MODE=gallery streamlit run app/streamlit_app.py
```

**Uploads** (full and cloud_live): JPG, PNG, WebP, and HEIC/HEIF from iPhones when `pillow-heif` is installed; at
most 10 MB and 50 megapixels. They are decoded in memory, downscaled by the same power-of-two rule the pipeline's
JPEG decoding uses (both sides stay at least 512 px), never written to disk, and their file names are not logged.

## Pages

- **Analyse:** click one of the 12 example artworks, or upload a JPG, PNG, WebP or HEIC, to see:
  - the most likely emotion and its sentiment;
  - the 9-emotion distribution;
  - "Emotions viewers may also feel" (presence head; minority-sensitive model only);
  - the sentiment totals;
  - the model's attention map;
  - the 28 image measurements, with training-set percentiles.

  The examples come from the ArtEmis *validation* split and were chosen by annotator agreement only,
  never by model accuracy. No test images are included.
- **Compare:** 2–4 images side by side.
- **About:** the method, the two models, the fresh-holdout and earlier test results, the known
  limitations and the changelog.

Switch models and languages in the sidebar. Switch between light and dark mode in Streamlit's menu
(⋮ → Settings → Theme).

## Contents

| path | what it is |
|---|---|
| `app/streamlit_app.py` | the UI |
| `app/inference.py` | the pipeline as plain functions (no Streamlit), usable from Python |
| `app/core.py` | torch-free paths, constants, upload decoding and display helpers |
| `app/gallery.py` | the precomputed example results (gallery and cloud_live) |
| `app/precompute_gallery.py`, `app/build_text_embeddings.py`, `app/verify_cloud_live.py` | build and verify `gallery.json`, the 27 text embeddings, and cloud_live against the full pipeline |
| `app/i18n.py` | all UI strings in English and zh-TW, plus the 27-sentence description table |
| `app/ui.py` | CSS, palette, Plotly charts, HTML components |
| `app/assets/models/r1/s42 … s46/`, `app/assets/models/r3/s42 … s46/` | the five checkpoints of each model with their `normalizer.json`, `dims.json`, `config.json` |
| `app/assets/priors.json` | validation and training class priors, and the optional R1 logit correction |
| `app/assets/presence.json` | the presence head's training pos_weight, undone before the sigmoid (R3) |
| `app/assets/attr_percentiles.json` | 1,001 training quantiles per attribute, used for the percentiles |
| `app/assets/examples/` | 12 validation artworks and their ArtEmis annotations; `rights_check.json` records the image-rights check (artist died before 1956, created before 1931, 2D), run with `app/check_gallery_rights.py` in the repository |
| `app/assets/precomputed/gallery.json`, `app/assets/text_embeddings.npz` | precomputed example results; CLIP text embeddings of the 27 descriptions |
| `app/assets/results.json` | fresh-holdout and earlier test results shown on the About page |
| `artfae_balance/`, `src/artfae/` | the unchanged research modules the pipeline imports |
| `tests/test_inference.py` | smoke and UI tests. The consistency check needs the research data, so it skips here. |

Run the tests with `pip install pytest`, then `pytest tests/test_inference.py`.

Using it from Python:

```python
import sys; sys.path.insert(0, "app")
import inference as inf
engine = inf.load_engine()
r = inf.predict_path(engine, "my_painting.jpg")
print(r["top"], dict(zip(inf.EMOTIONS, r["probs"].round(3))))
```

## Notes

- **Emotions are subjective.** The model learned the distribution of answers from ArtEmis annotators.
  It cannot know how a particular viewer will feel.
- **Anger is unreliable.** Test F1 for anger is about 0.12, because anger is rarely the majority
  answer.
- **The attention maps are not validated explanations.** They show model attention over the centre
  square that CLIP sees.
- **Reproducibility.** On the evaluation GPU, the app reproduces the saved validation logits of every
  seed to within 1e-4. On CPU the logits differ by up to about 3e-4 (probabilities by under 1e-5),
  with the same top classes. The cause is fp32-on-CPU vs TF32-on-GPU arithmetic in CLIP, made
  visible by the float16 feature cache.

Research context: NTCU TEEP project. Author: Nawazish Hussain.
