# Subset construction: the paper's 20K subset and the 43K training set

Revision Phase A, item 6. Every number here was recomputed from files on disk (`code/t6_subset_stats.py`, outputs `t6_*.csv`) or is quoted from a named provenance file. Where the record is missing, this document says so.

## 1. Summary

- **There are two different "20K" subsets.** Both have 14,000 / 3,000 / 3,000 artworks, but they contain different artworks and assign them to splits differently.
  - **v1, the seed-42 selection.** Files: `data/{train,val,test}_artworks.csv`, `data/artworks_20k.csv`, `data/selection_report.json`, identical to `/home/dev/ArtFAE-Net/Data/ArtEmis-20k_manifests/`.
  - **v2, the "user-reviewed" export ArtEmis-20k-GPU.** Files: `/home/dev/ArtFAE-Net/Data/*` and its byte-identical copy `/workspace/datasets/ArtEmis-20k-v2/ArtEmis-20k-GPU/`.
- **Every test number reported so far uses v2 val/test.** This covers `runs/final_eval`, `runs/final_eval_v2` and every model in Phase A. `data/paper_split/` is a byte-identical copy of the v2 val/test manifests, and `data/artemis_balanced/{val,test}.csv` were rebuilt from it.
- **The paper's M1 checkpoint and R0n were trained on v2 train.** The paper checkpoint is `/home/dev/ArtFAE-Net/results/artfae_m1/20261027T012203Z_s42_c2c10f53`. R0n was trained on `data/paper14k/train.csv`, which is v2 train.
- **v1 and v2 share 18,617 of 20,000 artworks**, but only 9,137 of their 14,000 train artworks.
- **The v2 selection procedure is unknown** (revision note 2026-11-03). v2 is described empirically in §3a: per-class inclusion rates against v1 and the natural ArtEmis distribution.
- **No script that produced v1 or v2 survives.** We searched `/workspace`, `/home/dev/ArtFAE-Net`, `/tmp`, and the Codex and Claude session logs for the generator and for strings unique to its report (`eligible_consistent_style`, `TOP_VOTE_TIE`). Only the outputs and their reports exist. The selection is documented below from those artifacts. The selection *rule* cannot be verified beyond what the reports state.
- **The 43K training set is every ArtEmis artwork with a usable local image, minus v2 val/test.** It was built by `build_balanced_artemis.py`, with the exact command recorded below. With `--train-size 0`, no class enrichment was applied.

## 2. The v1 seed-42 selection

Source: `data/selection_report.json`. The selection script is not available. The report records:

| step | count |
|---|---:|
| official ArtEmis artworks matched to an image (audit) | 49,024 |
| paintings in the Kaggle WikiArt metadata CSV | 49,056 |
| identifiers common to both | 49,018 |
| excluded: Kaggle art-style disagrees with the ArtEmis art_style (`data/art_style_disagreements.csv`, 492 rows) | 492 |
| excluded: official matches absent from the Kaggle CSV (`data/artemis_matches_without_kaggle_metadata.csv`) | 6 |
| Kaggle metadata without an official match (`data/kaggle_metadata_without_artemis_match.csv`) | 38 |
| eligible (consistent style and image) | 48,526 |
| selected | 20,000 (14,000 / 3,000 / 3,000) |
| annotations of the selected artworks | 112,722 (78,795 / 16,959 / 16,968) |

- **Policy** (report text): "ArtEmis official emotion labels and art styles only; exclude Kaggle art-style disagreements for conservative pilot."
- **Seed:** 42.
- **Warning** (report text): "Custom pilot split, not official ArtEmis benchmark split."
- **Strata.** Each artwork has a `stratum` column: its unique-top emotion, or `TOP_VOTE_TIE`. Counts over the 20,000 are contentment 6,166, tie 5,003, awe 2,352, sadness 1,485, amusement 1,322, fear 1,280, something_else 1,255, excitement 728, disgust 350, anger 59.
- **Class proportions.** These are close to the natural proportions of the full ArtEmis release (80,031 artworks, recomputed from `artemis_official_data/official_data/artemis_dataset_release_v0.csv`): contentment 29.5%, tie 25.4%, awe 10.9%, sadness 7.9%, fear 7.0%, amusement 6.7%, something_else 6.4%, excitement 3.7%, disgust 2.1%, anger 0.31%.
- **What can be inferred.** The proportions are consistent with seed-42 stratified sampling by `stratum` at natural proportions. Without the script this is an inference, not a verified fact.

**Where v1 was used:** the workspace's Oct 22 baselines B1–B5 and the B2 linear head (14,000 v1 train artworks; `results/leakage_audit_20261102/leakage_audit.csv`, see §6) and the superseded `data/artemis_balanced_locked20k` and `data/artemis_balanced_30k` builds.

## 3. The v2 "user-reviewed" export (the evaluated 20K)

Source: `/home/dev/ArtFAE-Net/Data/README.md` and `selection_report.json`. Neither the generator nor the review procedure is recorded.

- **README text:** "This package contains the exact user-reviewed train, validation and test memberships: 14,000 / 3,000 / 3,000 paintings. The source split folders were not modified or reshuffled. Metadata was regenerated from the actual filenames and official ArtEmis human annotations."
- **Source folders** `Data/ArtEmis-20k-final/{train,val,test}` are named in the report but no longer exist on disk.
- **Integrity** (report): 20,000 distinct SHA-256, 0 exact-duplicate groups, 0 cross-split exact duplicates, 0 identifiers with modified contents.
- **Annotations:** 111,713 rows (train 78,277 / val 16,801 / test 16,635).
- **Filename changes.** `audit/reviewed_filename_changes.csv` lists 82 train, 30 val and 52 test artworks added and the same numbers removed, relative to "the earlier manifests". Those earlier manifests are **not v1**: v1 and v2 differ by about 1,383 artworks overall and by 4,863 in train membership. The intermediate manifest the audit refers to is not on disk.
- **Strata over the 20,000:** contentment 6,057, tie 5,040, awe 2,373, sadness 1,526, fear 1,316, amusement 1,284, something_else 946, disgust 690, excitement 668, anger 100.
- **v2 departs from natural proportions** (v1 does not): anger 0.50% (natural 0.31%), disgust 3.45% (2.1%), something_else 4.7% (6.4%). Whether this was intended by the review cannot be determined.
- **Same artworks between v1 and v2:** 18,617 of 20,000.

| | v1 train | v1 val | v1 test |
|---|---:|---:|---:|
| v2 train | 9,137 | 1,954 | 1,946 |
| v2 val | 1,965 | 437 | 396 |
| v2 test | 1,951 | 399 | 432 |

### 3a. The v2 selection procedure is unknown; empirical description (revision note, 2026-11-03)

**The procedure that selected v2 is unknown.** No generator, review protocol or log exists. The only recorded description is the README's "user-reviewed" wording. v2 is therefore described here only by what it contains, compared with v1 and with the natural ArtEmis distribution. Code: `runs/revision_B/code/v2_inclusion_rates.py`; table: `runs/revision_B/v2_inclusion_rates.csv`.

Definitions:

- **Stratum:** the artwork's unique-top emotion, or "tie" (isclose atol 1e-5 on vote counts).
- **Inclusion rate:** the number of subset artworks in a stratum divided by the number of artworks in that stratum in the 49,011-image pool. Both subsets were drawn from that pool, except 4 v2 artworks whose pool images are invalid.
- **Enrichment:** the stratum's share in the subset divided by its share in the full official release (80,031 artworks).
- **v1 kept in v2:** the fraction of v1's artworks in a stratum that are also in v2 (any split).
- **v2 from v1:** the fraction of v2's artworks in a stratum that are also in v1.

| stratum | ArtEmis artworks | pool artworks | v1 | v2 | v1 inclusion rate | v2 inclusion rate | v1 enrichment | v2 enrichment | v1 kept in v2 | v2 from v1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| amusement | 5,398 | 3,306 | 1,322 | 1,284 | 0.400 | 0.388 | 0.98 | 0.95 | 0.930 | 0.958 |
| awe | 8,749 | 5,566 | 2,352 | 2,373 | 0.423 | 0.426 | 1.08 | 1.09 | 0.972 | 0.963 |
| contentment | 23,592 | 15,158 | 6,166 | 6,057 | 0.407 | 0.400 | 1.05 | 1.03 | 0.965 | 0.983 |
| excitement | 2,986 | 1,874 | 728 | 668 | 0.388 | 0.356 | 0.98 | 0.90 | 0.882 | 0.961 |
| anger | 247 | 147 | 59 | 100 | 0.401 | **0.680** | 0.96 | **1.62** | 0.729 | **0.430** |
| disgust | 1,680 | 885 | 350 | 690 | 0.395 | **0.780** | 0.83 | **1.64** | 0.846 | **0.429** |
| fear | 5,625 | 3,172 | 1,280 | 1,316 | 0.404 | 0.415 | 0.91 | 0.94 | 0.945 | 0.919 |
| sadness | 6,299 | 3,649 | 1,485 | 1,526 | 0.407 | 0.418 | 0.94 | 0.97 | 0.976 | 0.950 |
| something_else | 5,093 | 3,012 | 1,255 | 946 | 0.417 | **0.314** | 0.99 | **0.74** | **0.716** | 0.950 |
| tie | 20,362 | 12,242 | 5,003 | 5,040 | 0.409 | 0.412 | 0.98 | 0.99 | 0.921 | 0.915 |
| all | 80,031 | 49,011 | 20,000 | 20,000 | 0.408 | 0.408 | 1.00 | 1.00 | 0.931 | 0.931 |

What the table shows (descriptive only; it does not identify the procedure):

- **v1 is consistent with uniform sampling.** Its inclusion rate is 0.39–0.42 in every stratum, around the overall 0.408. That is what stratified random sampling of 20,000 from the pool at natural proportions would give.
- **v2 matches v1 in seven of ten strata** (inclusion 0.39–0.43, enrichment 0.90–1.09).
- **v2 over-includes anger and disgust.** It contains 68% of the pool's anger artworks and 78% of its disgust artworks, about 1.6× their natural share. More than half of those (57% each) are not in v1: v2 kept 73% of v1's anger and 85% of v1's disgust artworks and added others from the pool.
- **v2 under-includes something_else** (inclusion 0.31, enrichment 0.74). It dropped 28% of v1's something_else artworks, and slightly under-includes excitement (0.36).
- **Overall, v2 = v1 with about 1,383 artworks swapped** (93.1% of each subset is shared). The swaps concentrate on adding anger and disgust and removing something_else. The split membership was also redrawn (§3 table).

**Consequence for the manuscript:** v2 is a curated, non-random subset with deliberate or incidental rare-class enrichment of unknown origin. Results on v2 val/test should not be presented as estimates for natural-distribution ArtEmis. The Phase C fresh holdout (`runs/final_selection_v3_DRAFT2.txt`) uses all eligible unused artworks, with no class-based selection.

## 4. Class distributions and annotation counts

Unique-top counts use the hard-metric protocol (`isclose(atol=1e-5)`). Mean vote share is the mean over artworks of the per-artwork vote proportion. Full table: `t6_class_distributions.csv`.

| split | artworks | annotations | ann./artwork | tied | unique-top | amu | awe | con | exc | ang | dis | fea | sad | som |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 train | 14,000 | 78,795 | 5.63 | 3,502 | 10,498 | 925 | 1,646 | 4,316 | 510 | 41 | 245 | 896 | 1,040 | 879 |
| v1 val | 3,000 | 16,959 | 5.65 | 751 | 2,249 | 199 | 353 | 925 | 109 | 9 | 52 | 192 | 222 | 188 |
| v1 test | 3,000 | 16,968 | 5.66 | 750 | 2,250 | 198 | 353 | 925 | 109 | 9 | 53 | 192 | 223 | 188 |
| **v2 train = paper 14K** | 14,000 | 78,277 | 5.59 | 3,530 | 10,470 | 902 | 1,660 | 4,235 | 469 | 72 | 476 | 918 | 1,067 | 671 |
| **v2 val = evaluation val** | 3,000 | 16,801 | 5.60 | 757 | 2,243 | 192 | 356 | 911 | 102 | 14 | 108 | 197 | 227 | 136 |
| v2 test | 3,000 | 16,635 | 5.55 | 753 | 2,247 | 190 | 357 | 911 | 97 | 14 | 106 | 201 | 232 | 139 |
| **evaluation test** (v2 test minus 1, §5) | 2,999 | 16,629 | 5.55 | 753 | 2,246 | 189 | 357 | 911 | 97 | 14 | 106 | 201 | 232 | 139 |
| **43K train** | 43,012 | 245,392 | 5.71 | 10,732 | 32,280 | 2,925 | 4,853 | 13,336 | 1,675 | 119 | 671 | 2,774 | 3,190 | 2,737 |

Mean vote share (amu, awe, con, exc, ang, dis, fea, sad, som):

- **v2 train:** .097 .170 .288 .081 .020 .053 .084 .102 .106
- **evaluation val:** .099 .175 .290 .077 .019 .051 .086 .100 .103
- **evaluation test:** .099 .172 .284 .078 .020 .053 .087 .103 .105
- **43K train:** .101 .168 .293 .085 .013 .041 .083 .099 .118
- **all 49,011 artworks with images** (`Data/artemis_full_v1_20261031/dataset_statistics.json`): .101 .168 .292 .084 .014 .042 .083 .100 .116

The 43K train is close to the distribution over all 49,011 artworks with images. v2 train/val/test have about 1.4× the anger mean share (.019–.020 vs .014) and 1.2–1.3× the disgust share (.051–.053 vs .042).

**Label agreement.** `data/paper14k/train.csv` and `data/artemis_balanced/{val,test}.csv` were re-aggregated from the official release by `build_balanced_artemis.py`. On every shared artwork they match the v2 manifests exactly: vote counts identical, art_style identical, max |Δp| = 5e-9.

## 5. Exclusions in the evaluated splits

- **Images.** The 43K build indexes `clean_artemis/images`: 49,011 decodable images, from the prepared dataset `Data/artemis_full_v1_20261031` (`dataset_statistics.json`: 49,011 valid, 13 invalid images removed in cleaning). Only 49,011 of the 80,031 official artworks matched an image ("matched 49,011 / 80,031 artworks (31,020 missing)", `logs/build_balanced.log`). The other 31,020 have no local image and cannot enter any split.
- **Four v2 artworks are among the 13 invalid images** (`Data/artemis_full_v1_20261031/excluded_records.csv`, reason `invalid_image`), so they are absent from every 43K-build manifest:
  - **v2 train (paper 14K) → not in 43K train:** franz-marc_sleeping-animals-1913, henri-rousseau_view-of-the-bridge-at-sevres-and-the-hills-at-clamart-st-cloud-and-bellevue-1908, juan-gris_glass-and-carafe-1917.
  - **v2 test → dropped from the evaluation test:** pablo-picasso_untitled-1920-2 (6 annotations). This is why the evaluation test has **2,999** artworks.
  - Their images do exist in `data/images/` and `/home/dev/ArtFAE-Net/Data/images/`. `paper14k/train.csv` uses those copies, so R0n trains on all 14,000.
- **Annotations.** The 43K build aggregates the raw official CSV (454,684 rows), keeping the 9 emotion labels with `something else` → `something_else`. It does not use `clean_artemis` `cleaned_annotations.csv` (75 excluded annotations there). Votes therefore equal the official release.
- **No other filtering.** There is no filtering by annotation count, agreement, style or duplicates. The quarantine manifest was passed, but `[quarantine] dropped 0`, because invalid images were already absent from the index.

## 6. Development history relevant to the splits (2 Nov 2026, UTC)

Reconstructed from file timestamps and the Claude session log of that day.

1. **19:10** `data/artemis_balanced` was built against **v1** val/test (now `data/artemis_balanced_locked20k`, train 43,011).
2. **19:23** `results/leakage_audit_20261102` compared every checkpoint's training artworks with those v1 val/test.
   - The paper M1, simple-fusion, component and pipeline checkpoints (trained on v2 train) overlap them by 1,954 / 1,946 artworks, and were marked "excluded from test comparison".
   - The workspace B1–B5 baselines (v1 train) overlap by 0.
3. **19:26** The v1-based manifests and features were moved aside. v2 val/test were copied to `data/paper_split/`, and `data/artemis_balanced` was rebuilt against them (command in §7). This made the paper's M1 comparable on a test set disjoint from its training data.
4. **20:12** `data/paper14k/train.csv` was built from v2 train (§8). R0n was then trained for 5 seeds.
5. **20:05 and 22:44** The final test evaluations (`runs/final_eval`, `runs/final_eval_v2`) used v2 val/test.

**Consequence.** The v1 val/test artworks (2,164 / 2,172) are in the 43K train. This matters only for results reported on v1 test (the earlier workspace baselines), not for any v2 result.

## 7. How the 43K set was built

Command (19:26:50 UTC, log `logs/build_balanced.log`):

```bash
/workspace/.venv/bin/python build_balanced_artemis.py \
  --annotations artemis_official_data/official_data/artemis_dataset_release_v0.csv \
  --images clean_artemis/images --out data/artemis_balanced \
  --locked-split data/paper_split \
  --quarantine clean_artemis/prepared/v1_20261031/quarantine_manifest.csv \
  --train-size 0 --min-share 0.4
```

What the script does with these arguments:

1. **Aggregate votes** per `art_style/painting`. Write `n_*`, soft targets `p_*`, multi-hot `present_*`, `n_votes`, `unique_top` (isclose atol 1e-5), `top_emotion` ("tie" if tied), and `dominant` (argmax with ties broken by class order; used only for sampler weights).
2. **Attach an image** by lower-cased filename stem, with style-folder match first and then a style-free match. Artworks without an image are dropped (31,020).
3. **Lock val/test** to the paintings in `data/paper_split/{val,test}_artworks.csv`. This gives 3,000 val and 2,999 test (one image missing, §5).
4. **Train = all remaining matched artworks.** `--train-size 0` makes `enrich_train` return the pool unchanged, so the rare-class priority rule (`--rare anger,fear,disgust`, `--min-share`) and the class-capped fill are **not applied**. 49,011 − 3,000 − 2,999 = 43,012.
5. **Sampler weights.** `sampler_weight = (1/freq(dominant))^0.5`, normalized to mean 1. These are used only by runs with `--sampler` (R2b, R3, R3g, R4*).
6. **Disjointness asserts** on `artwork_id` and `image_path` across train/val/test. These passed.

Overlap of the 43K train: 13,997 of the 14,000 paper-14K train artworks (3 missing images); 0 v2 val; 0 evaluation test.

Superseded builds that are not used by any reported result: `data/artemis_balanced_30k` (1 Nov; v1 val/test; 30,000-artwork enriched train) and `data/artemis_balanced_locked20k` (2 Nov 19:10; v1 val/test).

## 8. How the paper 14K train manifest was built

`data/paper14k/train.csv`, 2 Nov 20:12 UTC. This is a one-off Python snippet in the session log; no script file exists. It:

- reads `/home/dev/ArtFAE-Net/Data/train_artworks.csv` (v2 train);
- copies `count_*` → `n_*` and `p_*`;
- sets `present_* = count > 0`, `n_votes = Σ counts`, and `unique_top` and `top_emotion` with the same isclose rule;
- points `image_path` at `/home/dev/ArtFAE-Net/Data/images`, asserting that every image exists and that p sums to 1.

Result: 14,000 rows, 78,277 votes, 10,470 unique-top. These match the v2 report. Features: `artfae_balance.extract_features --manifest data/paper14k/train.csv --out features_paper14k/train` (CLIP B/32 revision 3d74acf9…).

## 9. Overlap with the official ArtEmis train/val/test splits

**Not determinable from local files.**

- The official release CSV (`artemis_dataset_release_v0.csv`; columns art_style, painting, emotion, utterance, repetition) has no split column.
- The official splits are produced by the ArtEmis preprocessing script, which is not in this workspace. No preprocessed official split file exists anywhere on disk (searched for `*artemis_preprocessed*` and `*artemis*split*`).
- `Data/splits/artemis_full_70_15_15_v1` is a new 70/15/15 project split, not the official one.
- To measure the overlap, run the official ArtEmis preprocessing (with its default split loads and seed) on the same release CSV, then intersect by `(art_style, painting)`.
- Expectation, not a measurement: if the official split assigns artworks at random with the 85/5/10 loads used by the ArtEmis code, then about 85% of our test artworks would be in the official train split. Our subsets must therefore not be described as compatible with the official benchmark split.
- Both subset reports already state "not official ArtEmis benchmark split".

## 10. Near-duplicate candidates across the evaluated splits

The cleaned-data QC lists 191 perceptual-hash near-duplicate candidate pairs (`Data/artemis_full_v1_20261031/duplicate_groups.csv`). Intersected with the current splits (`t6_near_duplicate_cross_split.csv`):

| train set | val | test |
|---|---:|---:|
| 43K train | 16 pairs | 23 pairs |
| paper 14K train | 0 | 6 pairs |

- There are no val–test candidates.
- **These are unverified pHash candidates.** Some are clearly the same work under two filenames (e.g. `nicholas-roerich_himalayas-64` / `nicholas-roerich_himalayas-1942-2` at Hamming 0; `peter-paul-rubens_our-lady-with-the-saints` / `…-1`). Others are different dark portraits by different artists (e.g. Reynolds / Kramskoy).
- At most 23 of 2,999 test artworks are involved.
- Exact (SHA-256) duplicates: 0 within v2 (v2 report). No exact-duplicate groups in the cleaned image set.
