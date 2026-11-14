# App diagnosis: "Anger" on dark / near-uniform minimalist paintings

Date 2026-11-06. Read-only on every checkpoint, dataset, manifest and existing `runs/` entry; nothing retrained.
Images opened: validation images and synthetic stimuli only (no holdout, no test). Train manifest used for label
statistics only (no train image opened). Code: `code/`; raw outputs: `*.json`, `*.txt`, `*.npy`.

## 0. Things that differ from the brief

* **The suspicious images are not in the example gallery.** `app/`, `artfae_demo/app` and
  `artfae_app/artfae_demo/app` are byte-identical and contain the same 12 gallery images (one per emotion by
  annotator agreement + runner-ups for anger/disgust/fear). None is a colour-field / minimalist painting.
  Val itself contains only 2 Color_Field_Painting, 1 Minimalism and 8 Abstract_Expressionism artworks, and no
  purple monochrome or navy/black banded canvas (contact sheet `val_uniform_contact.png`). The paintings you saw
  were therefore almost certainly **uploads**. I reproduced the symptom with (i) synthetic stimuli built to the
  description (`stimuli/`, `stimuli_sheet.png`) and (ii) the closest val artworks, which have votes.
  If you send me the actual files I can run them through `code/s245_images.py` unchanged.
* `/workspace/_to_delete/MOVES.log` does not exist (nor `/workspace/_to_delete`); no path had moved.
* `/workspace/.venv` cannot import transformers (huggingface-hub 2.4.0 vs `<1.0` pin). Everything here ran in
  `artfae_demo/.venv` (torch 2.13.0, transformers 4.57.6, the app's pinned requirements); not modified.
* Val is the checkpoint-selection split for every model and the source of `val_prior`, so all val numbers are
  in-sample/optimistic, most of all for the macro-F1-selected R3_MF1 and the prior-corrected app.

## 1. Configuration (what the app actually loads) — `s1_config.json`

| item | value |
|---|---|
| checkpoints | `app/assets/models/s{42..46}/best.pt`, byte-identical to `runs/L14_R3_lam0.3_s{42..46}/best.pt` |
| SHA-256 vs `final_selection_v3.txt` | all 5 `best.pt` and `config.json` match the **R3_MF1** `checkpoint_sha256` entries |
| SHA-256 s42 / s43 / s44 / s45 / s46 | `841009042f2d…` / `d372f191553f…` / `f75b9cf97218…` / `926a307b3f48…` / `ef9d8378fd79…` (full hashes in s1_config.json) |
| registry label | "L/14 R3, existing macro-F1-selected runs (descriptive only; different protocol)" |
| selection rule | max val macro-F1 (strict >), saved epochs 3 / 2 / 2 / 1 / 3; min-val-softCE epochs would be 1 / 1 / 1 / 1 / 2 |
| note | s45: MF1 and softCE selection pick the same epoch → `L14_R3_lam0.3_s45` and `revision_B/step1/R3_s45` have identical weights and val logits |
| training manifest | `data/artemis_balanced/train.csv` (43,012), val `data/artemis_balanced/val.csv` (3,000) |
| architecture | ArtFAENetPlus, hidden 128, gated (AGF) on, spatial (ECSA, 7×7) on, numerical on, text on, presence head on; text_queries off, regional_gate off; dims 768/28/768/1024 |
| loss | soft CE (tau_train 0, no logit adjustment) + 0.3 × PresenceBCE(pos_weight=(1−rate)/rate, clamp 20; anger 14.53) |
| sampler | **on**: WeightedRandomSampler on `sampler_weight` (inverse dominant-class frequency), with replacement |
| class weights in the CE | none (only the presence BCE pos_weight) |
| features | CLIP ViT-L/14@336 rev ce19dc91, fp32, slow processor, 7×7 pooled patches stored fp16 |
| ensemble | per seed softmax(raw + δ), then **mean of probabilities** over 5 seeds; presence = mean of 5 sigmoids |
| prior correction δ = log(val_prior) − log(train_prior) | recomputed from the manifests: identical (max diff 0.0) |

Per-class prior correction (and what the sampler actually showed the model):

| | amus | awe | cont | exc | **anger** | disg | fear | sad | s_else |
|---|---|---|---|---|---|---|---|---|---|
| train prior | .1014 | .1675 | .2931 | .0849 | **.0127** | .0408 | .0825 | .0994 | .1177 |
| val prior | .0986 | .1749 | .2903 | .0772 | **.0189** | .0507 | .0862 | .1001 | .1031 |
| δ (app) | −.029 | +.043 | −.010 | −.095 | **+.394** | +.218 | +.044 | +.008 | −.133 |
| exp(δ) (odds ×) | 0.97 | 1.04 | 0.99 | 0.91 | **1.48** | 1.24 | 1.04 | 1.01 | 0.88 |
| sampler-effective train prior | .1038 | .1501 | .2225 | .0953 | **.0227** | .0579 | .1012 | .1142 | .1322 |
| δ if the sampler prior were used | −.052 | +.153 | +.266 | −.211 | **−.182** | −.133 | −.160 | −.132 | −.249 |

Anger has the largest shift (+0.394, odds ×1.48). The correction assumes the logits carry the *train* prior, but
R3 was trained through the balanced sampler, which already showed it anger at 2.27% (> val 1.89%). For this model
the correction pushes anger the wrong way. The holdout softCE claim checks out (`final_eval_v3/report.md`,
per-seed mean, raw logits: R3_MF1 1.8527 worst, R1 1.8357); I found no stored *holdout* prior-shift table, so the
"largest shift" statement is verified from priors.json (val vs train), not from holdout outputs.

Every one of the 20 checkpoints (R3_MF1, R3, R1, probe × 5) reproduces its saved `val_logits.npy` from
`features_l14/val` to ≤ 1.4e-6, and every `val_logits.npy` matches its registry hash.

## 2/5. Per-image decomposition — `s245_images.json`

Models: **a** app (R3_MF1 + δ) · **b** R3_MF1 raw · **c** R3 softCE-selected raw · **d** L/14 R1 raw · **e** linear probe raw.
All 5-seed probability ensembles. The app's own `predict()` equals column a to 0.0 on every image.

### Synthetic stimuli (matching the reported paintings; no human votes)

| stimulus | a | b | c | d | e | app top (a) |
|---|---|---|---|---|---|---|
| S1 tall purple monochrome | anger .099 / s_else .282 | .068 / .320 | .038 / .302 | .052 / .326 | .128 / .261 | something_else |
| S2 navy/black bands | **anger .196** / s_else .219 / sad .181 | .141 / .261 | .124 / .275 | .075 / .276 | .108 / .314 | something_else (per-seed tops: s_else, sadness, **anger**, s_else, fear) |
| S3 maroon on black | **anger .234** / fear .248 | .172 / .257 | .168 / .233 | .084 / .296 | .163 / .246 | fear |
| C1 uniform black | anger .190 / s_else .250 | .134 | .114 | .080 | .124 | something_else |
| C2 uniform mid grey | **anger .258** / s_else .337 | .186 | .149 | .139 | .116 | something_else |
| C3 uniform bright yellow | anger .232 / s_else .282 | .166 | .125 | .155 | .097 | something_else |

Anger mass on featureless canvases, step by step (d → c → b → a), e.g. S2: .075 → .124 → .141 → .196; C2:
.139 → .149 → .186 → .258. Base rate in val: 1.9%. Any texture, crop or colour variation in a real upload easily
flips the anger / something_else order, so the reported "Anger 24–30%, then something_else, then sadness"
reproduces in shape and magnitude.

### Val proxies (votes, utterances in `s245_images.json`)

| val row | style · painting | n | human shares | a anger / top | b | c | d | e | softCE a / b / c / d / e |
|---|---|---|---|---|---|---|---|---|---|
| 369 | Color_Field · paul-feeley_tenos-1956 | 5 | amus .2 cont .2 exc .2 disg .2 s_else .2 | .088 / disgust | .063 | .057 | .036 | .049 | 2.277 / 2.219 / 2.175 / 2.078 / **2.045** |
| 6 | AbEx · morris-graves_bird-moon-and-falling-stars-1940 | 5 | fear .6 disg .2 s_else .2 | .021 / fear | .015 | .020 | .014 | .013 | **1.429** / 1.452 / 1.485 / 1.523 / 1.499 |
| 1430 | Minimalism · jim-lambie_mental-oyster-installation-view-2005 | 6 | amus .33 exc .33 fear .17 s_else .17 | .036 / excitement | .024 | .035 | .024 | .007 | 1.672 / **1.645** / 1.768 / 1.742 / 1.683 |
| 1428 | Mannerism · titian_portrait-of-a-painter-with-a-palm-tree-1561 | 6 | cont .5 amus/awe/fear .17 (**anger 0**) | **.259 / ANGER** | .192 (anger) | .096 | .078 | .042 | 2.203 / 2.131 / 2.030 / 1.996 / **1.806** |
| 731 | Expressionism · pyotr-konchalovsky_portrait-of-baron-…-1911 | 10 | cont/disg/fear .2, anger .1, … | **.200 / ANGER** | .148 (fear) | .076 | .062 | .045 | 2.112 / 2.107 / **2.081** / 2.111 / 2.142 |

Raw votes for these rows match the official ArtEmis release counts (5, 5, 6, 6, 10). #1428 is the closest real
analogue of the symptom (dark, edge density 0.010, near-uniform background): app 25.9% anger, humans 0 anger votes.

## 3. Label order — `s3_labels.json`

`EMOTIONS` = amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something_else — identical in
artfae_balance, app inference, priors.json, all 20 checkpoints' saved `emotions`, i18n (en, zh), PALETTE (light,
dark), SENTIMENT, and the p_* column order of train/val.csv; training reads targets by name. One-hot test through
`ui.distribution_chart` and `ui.result_card` (9 classes × 2 themes × 2 languages) and a distinct-values test
(every bar carries its own value and colour; compare chart too; chips): **all pass**.

## 4. Preprocessing — val row 1428 (and all 5 proxies)

Titian #1428: file 2024×2456 → `load_image` draft → 1012×1228; CLIP resizes the short side to 336 and centre-crops,
i.e. original-pixel box (0, 108.4, 1012, 1120.4) — 82% of the height is seen. Attributes: luminance 0.223,
saturation 0.268, value 0.267, luminance_std 0.076, dark share 0.509, edge density 0.0096, warm 0.802, cool 0.000;
description "Low overall brightness, low luminance contrast, and predominantly warm colors."
(Paul Feeley #369: 300×500, box (0, 100, 300, 400), 60% of the height.)

| vs saved features_l14 val row | #369 | #1428 |
|---|---|---|
| global embedding | 4.5e-7 | 5.7e-7 |
| attribute-text embedding | 1.2e-7 | 6.3e-8 |
| 28 attributes | 0.0 | 0.0 |
| 7×7 patches (fp16) | 7.8e-3 abs (≈4.9% of values differ by fp16 rounding; ≤1.5e-4 near 0) | 2.0e-3 |
| **logits, all 20 runs** | **3.9e-5** | **6.1e-5** |

The patch differences are batch-64 vs batch-1 fp32 jitter crossing fp16 rounding boundaries; they do not
propagate (logits < 1e-4 on all 5 proxies; the repo test reports 5.6e-5 on its sample). No preprocessing bug.

## 6. Validation group analysis — `s6_groups.json`

val P25 thresholds: luminance 0.342, saturation 0.221, edge 0.069, luminance_std 0.159 (val quantiles).

| group | n | human anger share | human anger plurality (unique / incl. ties) | mean pred anger a / b / c / d / e | anger top-1 a / b / c / d / e | softCE a / b / c / d / e |
|---|---|---|---|---|---|---|
| G2 requested: CFP+Min+AbEx and (lum<P25 or sat<P25) | 3 | .000 | 0 / 0 | .038 / .026 / .031 / .023 / .018 | 0 / 0 / 0 / 0 / 0 | 1.656 / 1.655 / 1.697 / 1.684 / 1.614 |
| G1 CFP+Min+AbEx, all | 11 | .026 | 0 / .091 | .074 / .052 / .044 / .032 / .031 | 0 / 0 / 0 / 0 / 0 | 1.929 / 1.909 / 1.936 / 1.926 / 1.881 |
| G3 any style, dark + near-uniform | 178 | .022 | .006 / .028 | **.038** / .027 / .022 / .015 / .014 | .011 / .006 / 0 / 0 / 0 | 1.838 / 1.835 / 1.830 / **1.825** / 1.832 |
| G4 any style, dark + low saturation | 56 | .017 | 0 / .018 | .037 / .026 / .022 / .016 / .014 | 0 / 0 / 0 / 0 / 0 | 1.784 / 1.779 / 1.771 / 1.777 / **1.765** |
| G5 any style, near-uniform | 380 | .020 | .003 / .029 | .032 / .022 / .017 / .013 / .012 | .005 / .003 / 0 / 0 / 0 | 1.811 / 1.811 / 1.811 / **1.798** / 1.809 |
| **all val** | 3000 | .0189 | .005 / .023 | **.0287** / .0200 / .0157 / .0125 / .0112 | .006 / .002 / 0 / 0 / 0 | 1.757 / 1.756 / 1.756 / **1.750** / 1.756 |

* The requested group has n = 3 on val, too small to decide anything; G3–G5 are style-free attribute groups.
* The app inflates anger everywhere (×1.52 on all val, ×1.76 in G3) and is U-shaped in luminance (darkest decile:
  app .040 vs human .023). R1 and the probe *under*-predict anger (≈ .012 vs .019).
* The app's anger top-1 on val: 18 artworks; humans had anger as unique plurality on 1 of them. Without δ: 5;
  R3-softCE, R1, probe: 0. Humans: 14 unique-anger artworks.
* Prior-correction variants on val (`s8_correction_variants.txt`): for R3_MF1 the app's δ *worsens* softCE
  (1.7560 → 1.7571) and triples anger top-1 (5 → 18); the sampler-aware δ improves it (1.7524). For R1 the app's δ
  is neutral (1.7502 → 1.7502) and moves mean anger to .0182 (human .0189), as intended.
* Human signal is real but small: in **train labels** (no images opened) Color_Field_Painting (n 1,059) and
  Minimalism (n 737) have mean anger share 4.2% / 4.9% vs 1.3% overall, and something_else 27% / 35%; the 489 anger
  utterances are mostly "anyone could do this / this isn't art / just a black screen" (8 samples in
  `s8_correction_variants.txt`).

## 7. Presence head (chips)

* Chips = mean over the same 5 R3_MF1 seeds of sigmoid(presence logits); shown if > 0.5; the top emotion is
  deliberately excluded (`e != top` in streamlit_app.py). Same ensemble as the distribution: yes.
* Training target `present_c = (n_c > 0)` (any vote): verified on all 3,000 val rows. The label "probability that at
  least one viewer would name this emotion" matches the *target* — **but not the number shown**: the head was trained
  with pos_weight (anger 14.5), whose BCE optimum is logit + log(pos_weight). On val the anger chip averages 0.42 vs an
  empirical rate of 0.09; it fires on 39.5% of artworks and is right on 17% of those. Over all classes the mean
  calibration error is 0.169 (Brier 0.207), and the app shows 4.1 chips per artwork vs 3.3 emotions present.

## 8. Verdict

**Mixture; no label-order, preprocessing, ensembling or chart bug. One display bug (presence chips), fixed.**

1. *Base-model extrapolation from a genuine but small human signal.* Every model, including R1 and the linear probe,
   puts 8–16% on anger for featureless canvases (base rate 1.9%). Humans do say "anger" more for colour-field /
   minimalist works (≈4–5% share in train labels, "this isn't art" anger), and these styles are almost absent from val
   (3 artworks), so this behaviour was never visible in validation.
2. *Reweighting + selection amplify it.* Sampler + presence head (d → c) and macro-F1 checkpoint selection, i.e. later
   epochs past the softCE optimum (c → b), add roughly +0.02 to +0.09 of anger on these images (S2 .075 → .141,
   #1428 .078 → .192).
3. *The prior correction is mis-specified for this model and adds the last push.* δ = +0.394 (odds ×1.48) assumes
   train-prior logits, but the sampler already over-showed anger (2.27%). On val it makes R3_MF1's softCE worse and
   triples anger top-1 calls; on the reported images it adds +0.05 to +0.07 (S2 .141 → .196, C2 .186 → .258), which is what
   lifts anger over something_else.
4. *Presence chips were a separate display bug*: the anger chip appeared on these images (0.66–0.94) because pos_weight
   was never undone.

## 9. Fix applied (implementation bug only) — `s9_fix.diff`, `s9_after_fix.txt`

`app/inference.py` now computes chips as sigmoid(presence_logit − log(pos_weight)); `app/assets/presence.json` (new)
holds pos_weight rebuilt with the training code's own `PresenceBCE` (no fitting); `app/build_assets.py` writes it.
The emotion distribution, top emotion and model are untouched.

| | before | after |
|---|---|---|
| val mean calibration error (9 classes) | 0.169 | 0.019 |
| val presence Brier | 0.207 | 0.167 |
| val anger: mean prob / empirical rate | 0.422 / 0.090 | 0.072 / 0.090 |
| chips per val artwork (incl. top) vs emotions present | 4.10 vs 3.28 | 2.75 vs 3.28 |
| S2 navy/black anger chip | 0.885 | 0.351 (hidden) |
| C2 grey anger chip | 0.940 | 0.532 |
| #1428 Titian chips | amus, disg, fear, sad, s_else | fear .50, s_else .70 |
| tests/test_inference.py (7 tests, run without pytest) | — | all pass |

Not rebuilt: `app_bundle.tar.gz`, `artfae_demo/`, `artfae_app/` still contain the old app (re-run `app/package.sh`
when you want them updated). The About page still describes R3 + correction as "the pre-registered model" (v2
wording; under v3 R3_MF1 is "descriptive only"), left for the model decision below.

## Proposal (not implemented): model toggle

Sidebar radio, default first. Both options keep the same CLIP pipeline; assets would come from
`runs/revision_B/step1/R1_s{42..46}` and the existing R3 runs (which R3 selection to ship — MF1 or softCE — is your call).

| option | EN one-liner | zh-TW one-liner |
|---|---|---|
| **Human-vote distribution (L/14 R1, recommended)** — default | Predicts how a group of ArtEmis viewers would split their votes; the most accurate model for the full distribution (best softCE on the fresh holdout). | 預測一群 ArtEmis 觀者的投票分布；在新保留集上對整體分布最準確（softCE 最佳）的模型。 |
| **Minority-sensitive (L/14 R3)** | Trained to notice rarer emotions such as anger and fear; better at flagging them, but it overstates them, especially for plain or very dark images. | 以平衡取樣訓練，較能察覺憤怒、恐懼等少見情緒；但會高估這些情緒，在單色或極暗的畫面上尤其明顯。 |

Prior correction as an optional checkbox ("Match validation class frequencies" / 「校正為驗證集的類別頻率」), off by
default. Note for the UI help text: it is appropriate for R1 (neutral softCE on val, anger .0125 → .0182 vs human
.0189); for R3 the train→val δ is the wrong correction (it worsens val softCE); if kept for R3 it should use the
sampler-effective prior instead (δ_anger −0.18).
