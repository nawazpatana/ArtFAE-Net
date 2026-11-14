# Feature specification: `src/artfae/features/aesthetic.py` (version `rgb256-v1`)

Revision Phase A, item 7. This spec is transcribed from the source file (sha256 `55477737…`, the hash recorded in both selection files).

- **Cache check.** Recomputing `extract()` on 4 training images reproduces the cached `features/train/attributes.npy` rows exactly (max |Δ| = 0.0).
- **Empirical columns.** The ranges and sentence counts come from `code/t7_feature_stats.py`, outputs `t7_*.csv`.

## 1. Pre-processing (common to all 28 attributes)

1. **Load.** `artfae_balance.extract_features.load_image`:
   - `Image.open`;
   - `im.draft("RGB", (512, 512))` (JPEG DCT-domain downscale to the smallest scale ≥ 512 px; no effect for other formats);
   - EXIF orientation applied;
   - converted to RGB.
2. **Resize.** `image.convert('RGB').resize((256, 256), BILINEAR)`. The aspect ratio is **not** preserved: every image is squashed to 256 × 256.
3. **Arrays.** All three are 256 × 256, value range [0, 1], dimensionless.
   - `rgb = uint8 / 255`, the gamma-encoded sRGB values with no linearization.
   - `h, s, v` = PIL `convert('HSV')` / 255. PIL encodes hue as 0–255 over 0–360°, so **hue° = 360 · h** (pure red 0, yellow 42/255 = 59.3°, green 85/255 = 120°, blue 170/255 = 240°). Hue is quantized to 256 levels of 1.41°. **Achromatic pixels (s = 0) get h = 0, i.e. red.**
   - `y = 0.2126 R + 0.7152 G + 0.0722 B`. These are the Rec. 709 / sRGB luminance weights, applied to gamma-encoded values, so `y` is a *luma*, not CIE relative luminance.
4. **Edge map.** `dy, dx = np.gradient(y)` uses central differences in the interior, `(y[i+1] − y[i−1]) / 2`, and one-sided differences at the border. Units are luma per pixel at 256 px resolution. **`edge = sqrt(dx² + dy²) > 0.08`.**
5. **Chromatic mask.** `chromatic = s > 0.15`.

All means, variances and fractions below are over the 65,536 pixels of the 256 × 256 image. Variance and standard deviation use the population form (ddof = 0). Percentiles use NumPy linear interpolation.

## 2. The 28 attributes

Group = `GROUPS` in the same file, used for the numerical-group ablations. Empirical columns are over the 43K training set.

| # | name | formula | unit / range | threshold(s) | group | 43K train p5 / mean / p95 |
|---|---|---|---|---|---|---|
| 0 | red | mean(R) | [0, 1] | – | color | 0.237 / 0.517 / 0.819 |
| 1 | green | mean(G) | [0, 1] | – | color | 0.189 / 0.464 / 0.774 |
| 2 | blue | mean(B) | [0, 1] | – | color | 0.135 / 0.402 / 0.738 |
| 3 | hue_sin | mean(sin(2π h)) over **all** pixels (unweighted) | [−1, 1] | – | color | −0.364 / 0.276 / 0.734 |
| 4 | hue_cos | mean(cos(2π h)) over **all** pixels (unweighted; grey pixels add cos 0 = 1) | [−1, 1] | – | color | −0.392 / 0.473 / 0.950 |
| 5 | saturation | mean(s) (HSV saturation) | [0, 1] | – | color | 0.037 / 0.329 / 0.623 |
| 6 | saturation_var | var(s) | [0, 0.25] | – | color | 0.0004 / 0.030 / 0.074 |
| 7 | value | mean(v), v = max(R, G, B) | [0, 1] | – | color | 0.257 / 0.543 / 0.831 |
| 8 | warm | fraction of pixels with chromatic ∧ (h < 0.17 ∨ h > 0.92) | fraction of all pixels | s > 0.15; hue° < 61.2 or > 331.2 | color | 0.000 / 0.519 / 0.992 |
| 9 | cool | fraction of pixels with chromatic ∧ 0.33 ≤ h ≤ 0.75 | fraction of all pixels | s > 0.15; 118.8° ≤ hue° ≤ 270° | color | 0.000 / 0.121 / 0.528 |
| 10 | luminance | mean(y) | [0, 1] luma | – | brightness | 0.202 / 0.471 / 0.775 |
| 11 | luminance_std | std(y) | [0, 0.5] luma | – | contrast | 0.095 / 0.191 / 0.296 |
| 12 | dark | fraction of pixels with y < 0.2 | fraction | y < 0.2 | brightness | 0.000 / 0.196 / 0.667 |
| 13 | bright | fraction of pixels with y > 0.8 | fraction | y > 0.8 | brightness | 0.000 / 0.142 / 0.638 |
| 14 | contrast_p95_p05 | P95(y) − P5(y) | [0, 1] luma | – | contrast | 0.284 / 0.587 / 0.847 |
| 15 | edge_density | mean(edge) | fraction of pixels | ‖∇y‖ > 0.08 | complexity | 0.019 / 0.134 / 0.330 |
| 16 | edge_tl | mean(edge[0:128, 0:128]) (top-left quadrant) | fraction | ‖∇y‖ > 0.08 | complexity | 0.005 / 0.126 / 0.344 |
| 17 | edge_tr | mean(edge[0:128, 128:256]) (top-right) | fraction | ‖∇y‖ > 0.08 | complexity | 0.005 / 0.123 / 0.342 |
| 18 | edge_bl | mean(edge[128:256, 0:128]) (bottom-left) | fraction | ‖∇y‖ > 0.08 | complexity | 0.013 / 0.144 / 0.365 |
| 19 | edge_br | mean(edge[128:256, 128:256]) (bottom-right) | fraction | ‖∇y‖ > 0.08 | complexity | 0.012 / 0.141 / 0.362 |
| 20 | hue_bin_0 | saturation-weighted hue histogram, bin h ∈ [0, 1/8) | share, Σ bins = 1 | 0°–45° | color | 0.001 / 0.539 / 0.984 |
| 21 | hue_bin_1 | bin [1/8, 2/8) | share | 45°–90° | color | 0.000 / 0.159 / 0.612 |
| 22 | hue_bin_2 | bin [2/8, 3/8) | share | 90°–135° | color | 0.000 / 0.026 / 0.132 |
| 23 | hue_bin_3 | bin [3/8, 4/8) | share | 135°–180° | color | 0.000 / 0.030 / 0.160 |
| 24 | hue_bin_4 | bin [4/8, 5/8) | share | 180°–225° | color | 0.000 / 0.091 / 0.491 |
| 25 | hue_bin_5 | bin [5/8, 6/8) | share | 225°–270° | color | 0.000 / 0.048 / 0.258 |
| 26 | hue_bin_6 | bin [6/8, 7/8) | share | 270°–315° | color | 0.000 / 0.017 / 0.084 |
| 27 | hue_bin_7 | bin [7/8, 1] (right edge inclusive) | share | 315°–360° | color | 0.000 / 0.060 / 0.289 |

### Hue histogram (20–27)

`np.histogram(h, bins=8, range=(0, 1), weights=s)`, divided by `max(Σ weights, 1e-8)`.

- Each pixel contributes its saturation to the bin of its hue, so grey pixels contribute nothing.
- A fully achromatic image gives all-zero bins (Σ = 0, not 1).

### Ablation groups (`GROUPS`)

| group | attribute indices |
|---|---|
| color | 0–9 and 20–27, i.e. including `value`, `warm` and `cool` |
| brightness | 10, 12, 13 |
| contrast | 11, 14 |
| complexity | 15–19 |

### Normalization before the network

- **z-score:** `(a − mean_train) / max(std_train, 1e-6)` (`fit_normalizer` / `normalize`).
- **Fit on the training split of each run:** 43K for the R-ladder, paper 14K for R0n and the paper M1.
- **Stored** in each run's `normalizer.json`. The version, attribute names and `fit_split == 'train'` are checked at load.

### Notes for the manuscript

- **The attributes are deterministic image statistics, not aesthetic judgments.** This is the module docstring's own wording: "Versioned deterministic image measurements, not subjective aesthetic judgments."
- **hue_sin / hue_cos are biased toward red** by achromatic pixels (PIL assigns them hue 0) and are not saturation-weighted. The hue histogram and warm/cool are not affected, because they are saturation-weighted or masked.
- **The 0.08 edge threshold depends on resolution:** it is defined at 256 px after the non-uniform resize.

## 3. `describe()`: the 27 sentences

`describe(a)` uses **only 4 of the 28 attributes**: `luminance` (#10), `luminance_std` (#11), `warm` (#8) and `cool` (#9). Note that "luminance contrast" in the sentence is `luminance_std`, not `contrast_p95_p05`.

**Template:** `f"{B} overall brightness, {C} luminance contrast, and {K} colors."`

| slot | value | trigger rule (evaluated in this order) |
|---|---|---|
| B (brightness) | `Low` | luminance < 0.33 |
| | `High` | luminance > 0.67 |
| | `Moderate` | otherwise (0.33 ≤ luminance ≤ 0.67) |
| C (contrast) | `low` | luminance_std < 0.12 |
| | `high` | luminance_std > 0.25 |
| | `moderate` | otherwise (0.12 ≤ luminance_std ≤ 0.25) |
| K (colour) | `predominantly warm` | warm > cool + 0.1 |
| | `predominantly cool` | cool > warm + 0.1 |
| | `mixed or neutral` | otherwise (\|warm − cool\| ≤ 0.1) |

All 3 × 3 × 3 = 27 combinations are reachable, and all 27 occur in every split. Counts in each split:

| # | sentence | B rule | C rule | K rule | 43K train | paper 14K | val | test |
|---|---|---|---|---|---:|---:|---:|---:|
| 1 | Low overall brightness, low luminance contrast, and predominantly warm colors. | L < .33 | σ < .12 | w > c+.1 | 1,218 | 379 | 64 | 84 |
| 2 | Low overall brightness, low luminance contrast, and predominantly cool colors. | L < .33 | σ < .12 | c > w+.1 | 239 | 60 | 14 | 10 |
| 3 | Low overall brightness, low luminance contrast, and mixed or neutral colors. | L < .33 | σ < .12 | \|w−c\| ≤ .1 | 127 | 31 | 5 | 9 |
| 4 | Low overall brightness, moderate luminance contrast, and predominantly warm colors. | L < .33 | .12 ≤ σ ≤ .25 | w > c+.1 | 6,100 | 2,110 | 456 | 482 |
| 5 | Low overall brightness, moderate luminance contrast, and predominantly cool colors. | L < .33 | .12 ≤ σ ≤ .25 | c > w+.1 | 837 | 271 | 73 | 44 |
| 6 | Low overall brightness, moderate luminance contrast, and mixed or neutral colors. | L < .33 | .12 ≤ σ ≤ .25 | \|w−c\| ≤ .1 | 525 | 184 | 34 | 37 |
| 7 | Low overall brightness, high luminance contrast, and predominantly warm colors. | L < .33 | σ > .25 | w > c+.1 | 374 | 120 | 31 | 14 |
| 8 | Low overall brightness, high luminance contrast, and predominantly cool colors. | L < .33 | σ > .25 | c > w+.1 | 65 | 10 | 2 | 4 |
| 9 | Low overall brightness, high luminance contrast, and mixed or neutral colors. | L < .33 | σ > .25 | \|w−c\| ≤ .1 | 80 | 23 | 7 | 5 |
| 10 | Moderate overall brightness, low luminance contrast, and predominantly warm colors. | .33 ≤ L ≤ .67 | σ < .12 | w > c+.1 | 1,112 | 322 | 75 | 71 |
| 11 | Moderate overall brightness, low luminance contrast, and predominantly cool colors. | .33 ≤ L ≤ .67 | σ < .12 | c > w+.1 | 310 | 71 | 11 | 20 |
| 12 | Moderate overall brightness, low luminance contrast, and mixed or neutral colors. | .33 ≤ L ≤ .67 | σ < .12 | \|w−c\| ≤ .1 | 224 | 68 | 20 | 14 |
| 13 | Moderate overall brightness, moderate luminance contrast, and predominantly warm colors. | .33 ≤ L ≤ .67 | .12 ≤ σ ≤ .25 | w > c+.1 | 15,237 | 5,154 | 1,105 | 1,112 |
| 14 | Moderate overall brightness, moderate luminance contrast, and predominantly cool colors. | .33 ≤ L ≤ .67 | .12 ≤ σ ≤ .25 | c > w+.1 | 2,580 | 833 | 180 | 179 |
| 15 | Moderate overall brightness, moderate luminance contrast, and mixed or neutral colors. | .33 ≤ L ≤ .67 | .12 ≤ σ ≤ .25 | \|w−c\| ≤ .1 | 2,693 | 884 | 202 | 198 |
| 16 | Moderate overall brightness, high luminance contrast, and predominantly warm colors. | .33 ≤ L ≤ .67 | σ > .25 | w > c+.1 | 3,641 | 1,185 | 226 | 242 |
| 17 | Moderate overall brightness, high luminance contrast, and predominantly cool colors. | .33 ≤ L ≤ .67 | σ > .25 | c > w+.1 | 700 | 208 | 43 | 37 |
| 18 | Moderate overall brightness, high luminance contrast, and mixed or neutral colors. | .33 ≤ L ≤ .67 | σ > .25 | \|w−c\| ≤ .1 | 1,068 | 347 | 72 | 77 |
| 19 | High overall brightness, low luminance contrast, and predominantly warm colors. | L > .67 | σ < .12 | w > c+.1 | 736 | 223 | 43 | 44 |
| 20 | High overall brightness, low luminance contrast, and predominantly cool colors. | L > .67 | σ < .12 | c > w+.1 | 53 | 12 | 2 | 2 |
| 21 | High overall brightness, low luminance contrast, and mixed or neutral colors. | L > .67 | σ < .12 | \|w−c\| ≤ .1 | 716 | 189 | 44 | 34 |
| 22 | High overall brightness, moderate luminance contrast, and predominantly warm colors. | L > .67 | .12 ≤ σ ≤ .25 | w > c+.1 | 1,971 | 644 | 129 | 118 |
| 23 | High overall brightness, moderate luminance contrast, and predominantly cool colors. | L > .67 | .12 ≤ σ ≤ .25 | c > w+.1 | 244 | 66 | 15 | 12 |
| 24 | High overall brightness, moderate luminance contrast, and mixed or neutral colors. | L > .67 | .12 ≤ σ ≤ .25 | \|w−c\| ≤ .1 | 1,373 | 392 | 93 | 107 |
| 25 | High overall brightness, high luminance contrast, and predominantly warm colors. | L > .67 | σ > .25 | w > c+.1 | 349 | 97 | 28 | 15 |
| 26 | High overall brightness, high luminance contrast, and predominantly cool colors. | L > .67 | σ > .25 | c > w+.1 | 71 | 6 | 3 | 3 |
| 27 | High overall brightness, high luminance contrast, and mixed or neutral colors. | L > .67 | σ > .25 | \|w−c\| ≤ .1 | 369 | 111 | 23 | 25 |

L = luminance (#10), σ = luminance_std (#11), w = warm (#8), c = cool (#9).

The most frequent sentence (#13) covers 35.4% of the 43K train set and 37.1% of test.

### What the text branch actually receives

- **How `attr_text` is computed.** `attr_text.npy` is `CLIPModel.get_text_features` of the `describe()` sentence (`artfae_balance/extract_features*.py`). It is L2-normalized in both extractors, 512-d for B/32 and 768-d for L/14.
- **Exactly 27 distinct inputs.** Rows with the same sentence agree to within 2.4 × 10⁻⁷ (batch-padding float noise) in every split. The text branch therefore receives **one of exactly 27 fixed vectors**, a deterministic function of 4 thresholded attributes that are already inputs to the numerical branch (`t7_attr_text_check.csv`).
- **The 27 vectors are nearly collinear.** Pairwise cosine similarity between the B/32 sentence embeddings ranges from 0.936 to 0.9987. Any claim that the text branch adds semantic information beyond the numerical attributes should take this into account.
