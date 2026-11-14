# Revision Phase B: distribution prediction as the primary task

Generated 2026-11-03. Val only, except Step 0b, which re-reads the existing saved test logits for the requested
deduplication sensitivity analysis (no model run, nothing tuned). runs/final_eval* and runs/revision_A are
unchanged (revision_A/code/verify_untouched.py: 106 hashed artifacts match, no modified files). No
pre-registration written.

Code: code/ (contact_sheets.py, s0_dedup_sensitivity.py, extract_crop_attrs.py, train_abl.py, s1_eval.py,
s2_split.py, s2_eval.py). Runs: step1/<condition>_s<seed>/, step2/R1_s<seed>/. Logs: logs/.

## Step 0a. pHash pair classification (contact sheets in duplicates/pair_NN.jpg)

39 distinct pairs flagged in revision_A (23 test–43K train, 16 val–43K train; the 6 test–14K train pairs are a subset of the test–43K pairs). Classified by visual inspection of each contact sheet.

| classification | test pairs | val pairs | total |
|---|---|---|---|
| same artwork | 1 | 2 | 3 |
| same artwork, different reproduction | 15 | 14 | 29 |
| different | 7 | 0 | 7 |

Confirmed duplicates (not 'different'): 15 distinct test artworks and 16 val artworks, all against the 43K train; none of the 6 pairs against the paper 14K train is a duplicate.

| pair | train side | eval side | split | pHash | classification | note |
|---|---|---|---|---|---|---|
| pair_01 | albrecht-durer_a-quarry | albrecht-durer_quarry-1 | test | 2 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_02 | albrecht-durer_the-young-peasant-and-his-wife | albrecht-durer_rustic-couple-1497 | test | 4 | same artwork, different reproduction | engraving, different impressions/scans |
| pair_03 | albrecht-durer_mary-and-child | albrecht-durer_virgin-and-child-holding-a-half-eaten-pear-1512 | test | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_04 | antoine-watteau_view-through-the-trees-in-the-park-of-pierre-crozat | antoine-watteau_the-view | val | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_05 | camille-pissarro_avenue-de-l-opera-place-du-theatre-francais-misty | camille-pissarro_avenue-de-l-opera-place-du-thretre-francais-misty-weather-1898 | val | 2 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_06 | canaletto_an-island-in-the-lagoon-with-a-gateway-and-a-church | canaletto_capriccio-with-venetian-motifs | val | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_07 | canaletto_perspective-view-with-portico | canaletto_perspective-with-a-portico-1765 | val | 2 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_08 | childe-hassam_hackney-carriage-rue-bonaparte-also-known-as-fiacre-rue-bonaparte | childe-hassam_carriage-rue-bonaparte | val | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_09 | claude-monet_the-rocks-of-belle-lle | claude-monet_the-cote-sauvate | val | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_10 | edouard-manet_the-bock-drinkers-1878 | edouard-manet_at-the-cafe-concert | test | 4 | same artwork, different reproduction | different crop and colour |
| pair_11 | gustave-courbet_the-forest-edge | gustave-courbet_in-the-forest-1865 | val | 2 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_12 | henri-matisse_reader-on-a-black-background-1939 | henri-matisse_the-red-table-1939 | test | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_13 | grigoriy-myasoyedov_self-portrait-1878 | henry-raeburn_portrait-of-the-reverend-john-thomson-minister-of-duddingston | test | 4 | different | different sitters and artists (Myasoyedov vs Raeburn); dark-portrait pHash collision |
| pair_14 | ivan-aivazovsky_tempest-on-the-northern-sea-1865 | ivan-aivazovsky_the-shipwreck-on-northern-sea-1865 | test | 2 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_15 | ivan-aivazovsky_venice-1842-1 | ivan-aivazovsky_venice-1842 | val | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_16 | jacob-jordaens_holy-women-at-the-sepulchre | jacob-jordaens_the-disciples-at-the-tomb-1625 | val | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_17 | john-singer-sargent_lord-dalhousie-1900 | john-singer-sargent_the-earle-of-dalhousie-1900 | test | 2 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_18 | ivan-kramskoy_portrait-of-feodor-petrushevsky-1882 | joshua-reynolds_portrait-of-lord-john-townshend | test | 4 | different | different sitters and artists (Kramskoy vs Reynolds) |
| pair_19 | joshua-reynolds_self-portrait-4 | henry-raeburn_portrait-of-the-reverend-john-thomson-minister-of-duddingston | test | 4 | different | different sitters and artists (Reynolds vs Raeburn) |
| pair_20 | konstantin-korovin_port-in-marsele | konstantin-korovin_the-port-in-marseilles-1 | test | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_21 | mary-cassatt_emmie-and-her-child-1889 | mary-cassatt_mother-and-child-1 | val | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_22 | mary-cassatt_baby-in-his-mother-s-arms-sucking-his-finger-1889 | mary-cassatt_women-and-child-1889 | test | 4 | same artwork, different reproduction | pastel; different colour reproduction of the same work as pair 23's train copy |
| pair_23 | mary-cassatt_woman-and-child-in-front-of-a-shelf-which-are-placed-a-jug-and-basin-1889 | mary-cassatt_women-and-child-1889 | test | 0 | same artwork | identical image (pHash 0); the test artwork also has a second train copy in pair 22 |
| pair_24 | mikalojus-ciurlionis_sunrise-1904 | mikalojus-ciurlionis_deluge-vii-1904 | test | 4 | same artwork, different reproduction | same composition, two titles; colour differs strongly |
| pair_25 | nicholas-roerich_himalayas-64 | nicholas-roerich_himalayas-1942-2 | test | 0 | same artwork, different reproduction | pHash 0 but visibly different colour grading |
| pair_26 | nicholas-roerich_tangla-1943 | nicholas-roerich_thang-la | val | 2 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_27 | nicholas-roerich_hunting-1937 | nicholas-roerich_the-hunt-1937 | test | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_28 | nikolaos-lytras_landscape-of-sounio-1919 | nikolaos-lytras_landscape-sounion | val | 0 | same artwork | identical image |
| pair_29 | peter-paul-rubens_our-lady-with-the-saints | peter-paul-rubens_our-lady-with-the-saints-1 | val | 0 | same artwork | identical image; '-1' suffix copy |
| pair_30 | rembrandt_man-in-a-coat-and-fur-cap-leaning-against-a-bank-1630 | rembrandt_an-old-man-with-a-bushy-beard-1630 | test | 2 | same artwork, different reproduction | etching, different impressions |
| pair_31 | rembrandt_portrait-of-princess-amalia-van-solms-1632 | rembrandt_portrait-of-amalia-van-solms-1632 | val | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_32 | rembrandt_self-portrait-bareheaded | rembrandt_self-portrait-1633 | test | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_33 | rembrandt_st-peter-and-st-john-at-the-entrance-to-the-temple-1649 | rembrandt_peter-and-john-healing-the-cripple-at-the-gate-of-the-temple-1659 | val | 4 | same artwork, different reproduction | etching, different impressions |
| pair_34 | rembrandt_the-card-player-1641 | rembrandt_a-man-playing-cards-1641 | test | 4 | same artwork, different reproduction | etching, different impressions |
| pair_35 | leonardo-da-vinci_mona-lisa | robert-silvers_mona-lisa-remastered-2002 | test | 4 | different | derivative work: Silvers photomosaic after the Mona Lisa; different artwork, shared composition |
| pair_36 | thomas-cole_the-oxbow-the-connecticut-river-near-northampton-1836 | thomas-cole_view-from-mount-holyoke-1836 | val | 4 | same artwork, different reproduction | same painting; different photograph/scan, colour or crop |
| pair_37 | thomas-eakins_portrait-of-mary-adeline-williams-1899 | thomas-eakins_ernest-lee-parker-1910 | test | 4 | different | different sitters (Eakins, two portraits) |
| pair_38 | tintoretto_self-portrait-with-a-book | thomas-eakins_ernest-lee-parker-1910 | test | 4 | different | different artists (Tintoretto vs Eakins) |
| pair_39 | titian_portrait-of-antonio-anselmi | john-singer-sargent_peter-a-b-widener-1902 | test | 4 | different | different artists (Titian vs Sargent) |

## Step 0b. SENSITIVITY ANALYSIS: final_eval_v2 test summary with confirmed duplicates excluded

Not a new evaluation. The 15 test artworks visually confirmed as the same artwork as a 43K-train artwork (1 identical image, 14 different reproductions; duplicates/classification.csv) are removed, leaving 2984 test artworks (2238 unique-top). None of the pairs flagged against the paper 14K train is a duplicate, so R0n's training set has no confirmed duplicate in test; the same reduced test set is used for every model so comparisons stay paired. Existing test logits only; procedure identical to final_eval_v2 (full-test values reproduce test_per_seed.csv and bootstrap.json). The pHash candidate list (Hamming ≤ 4) is not exhaustive, so undetected duplicates may remain.

### Test, mean ± SD over seeds 42–46 (T fitted on val per seed); deduplicated set, Δ = dedup − full

| model | logits | macro-F1 | Δ | acc | balAcc | softCE(T) | Δ | Brier(T) | mean anger+fear AP | Δ | disgust F1 | neg-sentiment F1 | Δ | anger AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0n  B/32 paper 14K | raw | 0.3247 ± 0.0071 | -0.0003 | 0.4922 ± 0.0085 | 0.3208 ± 0.0089 | 1.8472 ± 0.0081 | -0.0000 | 0.1963 ± 0.0024 | 0.3813 ± 0.0085 | -0.0012 | 0.1949 ± 0.0139 | 0.5366 ± 0.0099 | -0.0029 | 0.6895 ± 0.0086 |
| R0n  B/32 paper 14K | corrected | 0.3232 ± 0.0028 | -0.0004 | 0.4929 ± 0.0070 | 0.3190 ± 0.0034 | 1.8457 ± 0.0078 | -0.0000 | 0.1960 ± 0.0024 | 0.3814 ± 0.0084 | -0.0012 | 0.1901 ± 0.0094 | 0.5326 ± 0.0082 | -0.0030 | 0.6894 ± 0.0086 |
| R1   B/32 | raw | 0.3244 ± 0.0046 | -0.0005 | 0.5113 ± 0.0091 | 0.3195 ± 0.0050 | 1.8222 ± 0.0150 | -0.0002 | 0.1884 ± 0.0049 | 0.4027 ± 0.0118 | -0.0004 | 0.1003 ± 0.0395 | 0.5479 ± 0.0180 | -0.0024 | 0.7016 ± 0.0171 |
| R1   B/32 | corrected | 0.3259 ± 0.0038 | -0.0005 | 0.5128 ± 0.0105 | 0.3185 ± 0.0033 | 1.8201 ± 0.0155 | -0.0002 | 0.1880 ± 0.0049 | 0.4022 ± 0.0121 | -0.0005 | 0.1553 ± 0.0351 | 0.5586 ± 0.0155 | -0.0025 | 0.7014 ± 0.0171 |
| R3   B/32 | raw | 0.3335 ± 0.0038 | -0.0003 | 0.5018 ± 0.0073 | 0.3375 ± 0.0049 | 1.8302 ± 0.0049 | -0.0001 | 0.1910 ± 0.0021 | 0.3940 ± 0.0081 | -0.0008 | 0.1544 ± 0.0414 | 0.5725 ± 0.0096 | -0.0025 | 0.6878 ± 0.0112 |
| R3   B/32 | corrected | 0.3319 ± 0.0059 | -0.0003 | 0.5001 ± 0.0070 | 0.3330 ± 0.0045 | 1.8298 ± 0.0059 | -0.0000 | 0.1913 ± 0.0023 | 0.3934 ± 0.0080 | -0.0008 | 0.1811 ± 0.0205 | 0.5801 ± 0.0086 | -0.0023 | 0.6874 ± 0.0114 |
| R4b  B/32 | raw | 0.3364 ± 0.0035 | -0.0002 | 0.5004 ± 0.0099 | 0.3409 ± 0.0035 | 1.8360 ± 0.0130 | -0.0000 | 0.1927 ± 0.0039 | 0.3985 ± 0.0120 | -0.0010 | 0.1650 ± 0.0423 | 0.5763 ± 0.0098 | -0.0025 | 0.6775 ± 0.0141 |
| R4b  B/32 | corrected | 0.3457 ± 0.0062 | -0.0001 | 0.5030 ± 0.0080 | 0.3474 ± 0.0069 | 1.8364 ± 0.0141 | +0.0000 | 0.1931 ± 0.0042 | 0.3981 ± 0.0120 | -0.0011 | 0.1955 ± 0.0400 | 0.5763 ± 0.0148 | -0.0022 | 0.6772 ± 0.0142 |
| R1   L/14@336 | raw | 0.3683 ± 0.0090 | -0.0004 | 0.5530 ± 0.0078 | 0.3579 ± 0.0079 | 1.7763 ± 0.0097 | +0.0002 | 0.1743 ± 0.0036 | 0.4485 ± 0.0047 | -0.0014 | 0.1762 ± 0.0262 | 0.6001 ± 0.0100 | -0.0023 | 0.7379 ± 0.0043 |
| R1   L/14@336 | corrected | 0.3813 ± 0.0110 | -0.0004 | 0.5566 ± 0.0112 | 0.3673 ± 0.0097 | 1.7735 ± 0.0105 | +0.0002 | 0.1738 ± 0.0037 | 0.4486 ± 0.0045 | -0.0014 | 0.2015 ± 0.0233 | 0.6139 ± 0.0111 | -0.0027 | 0.7381 ± 0.0044 |
| R3   L/14@336 | raw | 0.3829 ± 0.0099 | -0.0003 | 0.5414 ± 0.0040 | 0.3834 ± 0.0119 | 1.7851 ± 0.0030 | +0.0002 | 0.1774 ± 0.0007 | 0.4386 ± 0.0041 | -0.0015 | 0.1896 ± 0.0164 | 0.6310 ± 0.0130 | -0.0021 | 0.7239 ± 0.0069 |
| R3   L/14@336 | corrected | 0.3847 ± 0.0053 | -0.0004 | 0.5433 ± 0.0043 | 0.3867 ± 0.0092 | 1.7859 ± 0.0035 | +0.0003 | 0.1777 ± 0.0012 | 0.4383 ± 0.0046 | -0.0015 | 0.2069 ± 0.0208 | 0.6369 ± 0.0082 | -0.0024 | 0.7240 ± 0.0071 |

### Paired bootstrap, deduplicated test (2,000 resamples, rng seed 0); full-test values from final_eval_v2 for reference

| comparison | metric | Δ dedup | 95% CI dedup | p (Holm) dedup | Δ full | 95% CI full | p (Holm) full |
|---|---|---|---|---|---|---|---|
| (a) L/14 R3c − R0n | macro-F1 | +0.0480 | [+0.0220, +0.0754] | 0.0080 | +0.0482 | [+0.0224, +0.0749] | 0.0080 |
| (a) L/14 R3c − R0n | mean anger+fear AP | +0.0600 | [+0.0371, +0.0839] | 0.0040 | +0.0603 | [+0.0382, +0.0853] | 0.0040 |
| (a) L/14 R3c − R0n | neg-sentiment F1 | +0.1108 | [+0.0808, +0.1434] | 0.0040 | +0.1097 | [+0.0776, +0.1393] | 0.0040 |
| (a) L/14 R3c − R0n | softCE(T) | -0.0630 | [-0.0715, -0.0548] | 0.0040 | -0.0633 | [-0.0716, -0.0552] | 0.0040 |
| (b) L/14 R1 − B/32 R1 | macro-F1 | +0.0291 | [+0.0086, +0.0494] | 0.0360 | +0.0290 | [+0.0096, +0.0515] | 0.0090 |
| (b) L/14 R1 − B/32 R1 | mean anger+fear AP | +0.0471 | [+0.0233, +0.0695] | 0.0040 | +0.0481 | [+0.0253, +0.0708] | 0.0060 |
| (b) L/14 R1 − B/32 R1 | neg-sentiment F1 | +0.0463 | [+0.0125, +0.0808] | 0.0260 | +0.0460 | [+0.0102, +0.0801] | 0.0260 |
| (b) L/14 R1 − B/32 R1 | softCE(T) | -0.0414 | [-0.0482, -0.0346] | 0.0040 | -0.0418 | [-0.0489, -0.0350] | 0.0040 |
| (c) L/14 R3 − L/14 R1 | macro-F1 | +0.0318 | [+0.0046, +0.0605] | 0.0380 | +0.0317 | [+0.0034, +0.0613] | 0.0400 |
| (c) L/14 R3 − L/14 R1 | mean anger+fear AP | -0.0099 | [-0.0195, +0.0012] | 0.1619 | -0.0099 | [-0.0197, +0.0013] | 0.1659 |
| (c) L/14 R3 − L/14 R1 | neg-sentiment F1 | +0.0407 | [+0.0190, +0.0639] | 0.0040 | +0.0401 | [+0.0186, +0.0631] | 0.0040 |
| (c) L/14 R3 − L/14 R1 | softCE(T) | +0.0040 | [+0.0010, +0.0069] | 0.0260 | +0.0039 | [+0.0011, +0.0069] | 0.0220 |
| (d) L/14 R3c − L/14 R3 | macro-F1 | -0.0111 | [-0.0257, +0.0028] | 0.1299 | -0.0112 | [-0.0260, +0.0026] | 0.1299 |
| (d) L/14 R3c − L/14 R3 | mean anger+fear AP | +0.0003 | [-0.0011, +0.0024] | 0.8756 | +0.0003 | [-0.0010, +0.0022] | 0.8406 |
| (d) L/14 R3c − L/14 R3 | neg-sentiment F1 | +0.0088 | [-0.0032, +0.0220] | 0.1419 | +0.0086 | [-0.0038, +0.0215] | 0.1759 |
| (d) L/14 R3c − L/14 R3 | softCE(T) | +0.0002 | [-0.0014, +0.0017] | 0.7626 | +0.0001 | [-0.0014, +0.0016] | 0.8706 |

### Seed-averaged ensembles, deduplicated test

| model | logits | macro-F1 | mean anger+fear AP | neg-sentiment F1 | softCE(T) |
|---|---|---|---|---|---|
| R0n  B/32 paper 14K | raw | 0.3383 | 0.3899 | 0.5362 | 1.8335 |
| R0n  B/32 paper 14K | corrected | 0.3343 | 0.3901 | 0.5366 | 1.8323 |
| R1   B/32 | raw | 0.3365 | 0.4124 | 0.5512 | 1.8078 |
| R1   B/32 | corrected | 0.3321 | 0.4120 | 0.5712 | 1.8050 |
| R3   B/32 | raw | 0.3514 | 0.4086 | 0.5743 | 1.8113 |
| R3   B/32 | corrected | 0.3542 | 0.4085 | 0.5918 | 1.8101 |
| R4b  B/32 | raw | 0.3482 | 0.4208 | 0.5745 | 1.8126 |
| R4b  B/32 | corrected | 0.3459 | 0.4207 | 0.5821 | 1.8124 |
| R1   L/14@336 | raw | 0.3656 | 0.4595 | 0.5975 | 1.7663 |
| R1   L/14@336 | corrected | 0.3863 | 0.4593 | 0.6160 | 1.7632 |
| R3   L/14@336 | raw | 0.3974 | 0.4496 | 0.6383 | 1.7703 |
| R3   L/14@336 | corrected | 0.3863 | 0.4498 | 0.6471 | 1.7705 |

## Step 1. Controlled L/14 ablations (val, seeds 42–46, checkpoint = min val softCE)

Val n = 3,000 (2243 unique-top, 757 tied). Val is also the selection split, so all numbers are selection-split estimates. softCE(T): T refit on val per run (in-sample). 16 val artworks are confirmed reproductions of 43K-train artworks (Step 0); they are included here, as in every earlier val table.

### Val metrics, mean ± SD over 5 seeds

| condition | trainable params | sel. epoch | softCE | KL | Brier | softCE(T) | acc | macro-F1 | balAcc | mean anger+fear AP | neg-sentiment F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 base (L/14 R1, unweighted) | 762,249 | 3.6 | 1.7547 ± 0.0019 | 0.7081 ± 0.0019 | 0.1718 ± 0.0004 | 1.7537 ± 0.0014 | 0.5669 ± 0.0037 | 0.3576 ± 0.0050 | 0.3413 ± 0.0043 | 0.4420 ± 0.0059 | 0.5924 ± 0.0097 |
| CLIP-only linear probe (global) | 6,921 | 20.0 | 1.7772 ± 0.0001 | 0.7305 ± 0.0001 | 0.1775 ± 0.0000 | 1.7771 ± 0.0001 | 0.5439 ± 0.0009 | 0.3150 ± 0.0007 | 0.3032 ± 0.0007 | 0.4156 ± 0.0002 | 0.4803 ± 0.0015 |
| B5 concatenation baseline | 402,953 | 17.6 | 1.7536 ± 0.0004 | 0.7069 ± 0.0004 | 0.1713 ± 0.0001 | 1.7528 ± 0.0002 | 0.5605 ± 0.0017 | 0.3437 ± 0.0018 | 0.3300 ± 0.0018 | 0.4438 ± 0.0005 | 0.5927 ± 0.0065 |
| − numerical attributes | 709,257 | 3.4 | 1.7545 ± 0.0012 | 0.7079 ± 0.0012 | 0.1720 ± 0.0003 | 1.7538 ± 0.0011 | 0.5643 ± 0.0072 | 0.3574 ± 0.0120 | 0.3415 ± 0.0117 | 0.4492 ± 0.0064 | 0.5940 ± 0.0112 |
| − attribute text | 614,537 | 3.4 | 1.7548 ± 0.0020 | 0.7081 ± 0.0020 | 0.1717 ± 0.0006 | 1.7539 ± 0.0017 | 0.5694 ± 0.0026 | 0.3623 ± 0.0080 | 0.3472 ± 0.0079 | 0.4432 ± 0.0079 | 0.5954 ± 0.0090 |
| − both attribute branches | 561,545 | 3.4 | 1.7540 ± 0.0014 | 0.7073 ± 0.0014 | 0.1718 ± 0.0003 | 1.7530 ± 0.0012 | 0.5627 ± 0.0049 | 0.3564 ± 0.0103 | 0.3400 ± 0.0103 | 0.4491 ± 0.0071 | 0.5884 ± 0.0099 |
| − ECSA (no regional term) | 350,089 | 6.8 | 1.7530 ± 0.0007 | 0.7063 ± 0.0007 | 0.1715 ± 0.0005 | 1.7511 ± 0.0011 | 0.5591 ± 0.0008 | 0.3412 ± 0.0033 | 0.3276 ± 0.0023 | 0.4534 ± 0.0018 | 0.6060 ± 0.0078 |
| − AGF (concatenation, no gate) | 663,433 | 3.6 | 1.7574 ± 0.0024 | 0.7108 ± 0.0024 | 0.1725 ± 0.0006 | 1.7564 ± 0.0021 | 0.5648 ± 0.0061 | 0.3548 ± 0.0119 | 0.3396 ± 0.0110 | 0.4368 ± 0.0039 | 0.5861 ± 0.0107 |
| class-weighted soft CE (p 0.25, [0.5, 2]) | 762,249 | 3.8 | 1.7585 ± 0.0010 | 0.7118 ± 0.0010 | 0.1737 ± 0.0003 | 1.7580 ± 0.0011 | 0.5632 ± 0.0031 | 0.3740 ± 0.0020 | 0.3656 ± 0.0041 | 0.4418 ± 0.0096 | 0.6233 ± 0.0145 |
| attributes on 336-px center crop | 762,249 | 3.6 | 1.7547 ± 0.0018 | 0.7080 ± 0.0018 | 0.1721 ± 0.0008 | 1.7536 ± 0.0013 | 0.5687 ± 0.0059 | 0.3598 ± 0.0126 | 0.3456 ± 0.0118 | 0.4441 ± 0.0054 | 0.5992 ± 0.0168 |
| ref: existing L/14 R1, macro-F1 selection | 762,249 | 5.8 | 1.7652 ± 0.0160 | 0.7186 ± 0.0160 | 0.1751 ± 0.0046 | 1.7616 ± 0.0111 | 0.5682 ± 0.0105 | 0.3757 ± 0.0068 | 0.3621 ± 0.0058 | 0.4440 ± 0.0134 | 0.5928 ± 0.0159 |

### softCE and KL on unique-top vs tied val artworks (T = 1), mean ± SD

| condition | softCE unique-top | softCE tied | KL unique-top | KL tied |
|---|---|---|---|---|
| R1 base (L/14 R1, unweighted) | 1.6851 ± 0.0012 | 1.9610 ± 0.0056 | 0.6987 ± 0.0012 | 0.7359 ± 0.0056 |
| CLIP-only linear probe (global) | 1.7101 ± 0.0001 | 1.9760 ± 0.0001 | 0.7237 ± 0.0001 | 0.7509 ± 0.0001 |
| B5 concatenation baseline | 1.6830 ± 0.0002 | 1.9628 ± 0.0018 | 0.6965 ± 0.0002 | 0.7377 ± 0.0018 |
| − numerical attributes | 1.6854 ± 0.0014 | 1.9594 ± 0.0032 | 0.6989 ± 0.0014 | 0.7343 ± 0.0032 |
| − attribute text | 1.6848 ± 0.0019 | 1.9622 ± 0.0050 | 0.6984 ± 0.0019 | 0.7371 ± 0.0050 |
| − both attribute branches | 1.6843 ± 0.0015 | 1.9606 ± 0.0037 | 0.6978 ± 0.0015 | 0.7355 ± 0.0037 |
| − ECSA (no regional term) | 1.6809 ± 0.0015 | 1.9664 ± 0.0049 | 0.6945 ± 0.0015 | 0.7413 ± 0.0049 |
| − AGF (concatenation, no gate) | 1.6876 ± 0.0023 | 1.9645 ± 0.0053 | 0.7011 ± 0.0023 | 0.7394 ± 0.0053 |
| class-weighted soft CE (p 0.25, [0.5, 2]) | 1.6936 ± 0.0022 | 1.9509 ± 0.0035 | 0.7071 ± 0.0022 | 0.7258 ± 0.0035 |
| attributes on 336-px center crop | 1.6850 ± 0.0012 | 1.9609 ± 0.0046 | 0.6986 ± 0.0012 | 0.7358 ± 0.0046 |
| ref: existing L/14 R1, macro-F1 selection | 1.6948 ± 0.0147 | 1.9739 ± 0.0199 | 0.7083 ± 0.0147 | 0.7488 ± 0.0199 |

### Paired difference from R1 (condition − R1, same seed): mean [95% t-interval, df 4]

Lower is better for softCE/KL/Brier; higher for the rest. ✱ = 95% interval excludes 0.

| condition | softCE | KL | Brier | softCE(T) | acc | macro-F1 | balAcc | mean anger+fear AP | neg-sentiment F1 | softCE unique-top | softCE tied | KL unique-top | KL tied |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP-only linear probe (global) | +0.0225 [+0.0202, +0.0248]✱ | +0.0225 [+0.0202, +0.0248]✱ | +0.0057 [+0.0053, +0.0061]✱ | +0.0234 [+0.0217, +0.0251]✱ | -0.0230 [-0.0274, -0.0186]✱ | -0.0426 [-0.0493, -0.0360]✱ | -0.0381 [-0.0438, -0.0325]✱ | -0.0264 [-0.0337, -0.0191]✱ | -0.1121 [-0.1247, -0.0996]✱ | +0.0250 [+0.0235, +0.0265]✱ | +0.0150 [+0.0081, +0.0218]✱ | +0.0250 [+0.0235, +0.0265]✱ | +0.0150 [+0.0081, +0.0218]✱ |
| B5 concatenation baseline | -0.0012 [-0.0035, +0.0011] | -0.0012 [-0.0035, +0.0011] | -0.0005 [-0.0010, -0.0000]✱ | -0.0009 [-0.0026, +0.0008] | -0.0064 [-0.0124, -0.0004]✱ | -0.0140 [-0.0196, -0.0083]✱ | -0.0114 [-0.0165, -0.0062]✱ | +0.0018 [-0.0058, +0.0095] | +0.0003 [-0.0105, +0.0111] | -0.0022 [-0.0038, -0.0005]✱ | +0.0018 [-0.0050, +0.0086] | -0.0022 [-0.0038, -0.0005]✱ | +0.0018 [-0.0050, +0.0086] |
| − numerical attributes | -0.0002 [-0.0013, +0.0009] | -0.0002 [-0.0013, +0.0009] | +0.0002 [-0.0001, +0.0005] | +0.0000 [-0.0009, +0.0009] | -0.0026 [-0.0099, +0.0048] | -0.0003 [-0.0126, +0.0121] | +0.0002 [-0.0115, +0.0119] | +0.0072 [+0.0010, +0.0134]✱ | +0.0016 [-0.0171, +0.0203] | +0.0003 [-0.0012, +0.0018] | -0.0016 [-0.0060, +0.0029] | +0.0003 [-0.0012, +0.0018] | -0.0016 [-0.0060, +0.0029] |
| − attribute text | +0.0001 [-0.0006, +0.0008] | +0.0001 [-0.0006, +0.0008] | -0.0001 [-0.0004, +0.0003] | +0.0001 [-0.0005, +0.0008] | +0.0025 [-0.0034, +0.0084] | +0.0047 [-0.0039, +0.0133] | +0.0059 [-0.0041, +0.0159] | +0.0013 [-0.0072, +0.0097] | +0.0029 [-0.0088, +0.0147] | -0.0003 [-0.0016, +0.0011] | +0.0012 [-0.0018, +0.0041] | -0.0003 [-0.0016, +0.0011] | +0.0012 [-0.0018, +0.0041] |
| − both attribute branches | -0.0008 [-0.0018, +0.0003] | -0.0008 [-0.0018, +0.0003] | +0.0000 [-0.0004, +0.0004] | -0.0007 [-0.0017, +0.0003] | -0.0042 [-0.0098, +0.0014] | -0.0013 [-0.0110, +0.0084] | -0.0013 [-0.0110, +0.0084] | +0.0071 [-0.0001, +0.0144] | -0.0040 [-0.0169, +0.0089] | -0.0009 [-0.0018, +0.0001] | -0.0005 [-0.0033, +0.0024] | -0.0009 [-0.0018, +0.0001] | -0.0005 [-0.0033, +0.0024] |
| − ECSA (no regional term) | -0.0018 [-0.0036, +0.0000] | -0.0018 [-0.0036, +0.0000] | -0.0003 [-0.0008, +0.0001] | -0.0026 [-0.0052, -0.0001]✱ | -0.0078 [-0.0121, -0.0035]✱ | -0.0164 [-0.0233, -0.0096]✱ | -0.0137 [-0.0195, -0.0080]✱ | +0.0114 [+0.0020, +0.0209]✱ | +0.0136 [-0.0057, +0.0329] | -0.0042 [-0.0071, -0.0013]✱ | +0.0053 [-0.0000, +0.0107] | -0.0042 [-0.0071, -0.0013]✱ | +0.0053 [-0.0000, +0.0107] |
| − AGF (concatenation, no gate) | +0.0027 [+0.0015, +0.0039]✱ | +0.0027 [+0.0015, +0.0039]✱ | +0.0007 [+0.0001, +0.0012]✱ | +0.0027 [+0.0015, +0.0039]✱ | -0.0021 [-0.0106, +0.0063] | -0.0028 [-0.0131, +0.0074] | -0.0017 [-0.0114, +0.0079] | -0.0052 [-0.0084, -0.0020]✱ | -0.0064 [-0.0154, +0.0026] | +0.0024 [+0.0009, +0.0040]✱ | +0.0035 [+0.0022, +0.0048]✱ | +0.0024 [+0.0009, +0.0040]✱ | +0.0035 [+0.0022, +0.0048]✱ |
| class-weighted soft CE (p 0.25, [0.5, 2]) | +0.0038 [+0.0008, +0.0068]✱ | +0.0038 [+0.0008, +0.0068]✱ | +0.0019 [+0.0015, +0.0022]✱ | +0.0043 [+0.0022, +0.0065]✱ | -0.0037 [-0.0110, +0.0035] | +0.0163 [+0.0082, +0.0244]✱ | +0.0243 [+0.0166, +0.0320]✱ | -0.0002 [-0.0105, +0.0102] | +0.0309 [+0.0136, +0.0482]✱ | +0.0085 [+0.0049, +0.0120]✱ | -0.0101 [-0.0141, -0.0061]✱ | +0.0085 [+0.0049, +0.0120]✱ | -0.0101 [-0.0141, -0.0061]✱ |
| attributes on 336-px center crop | -0.0001 [-0.0002, +0.0001] | -0.0001 [-0.0002, +0.0001] | +0.0003 [-0.0004, +0.0010] | -0.0001 [-0.0003, +0.0001] | +0.0018 [-0.0077, +0.0112] | +0.0022 [-0.0142, +0.0185] | +0.0043 [-0.0110, +0.0196] | +0.0021 [-0.0073, +0.0114] | +0.0067 [-0.0093, +0.0228] | -0.0001 [-0.0006, +0.0004] | -0.0001 [-0.0020, +0.0018] | -0.0001 [-0.0006, +0.0004] | -0.0001 [-0.0020, +0.0018] |
| ref: existing L/14 R1, macro-F1 selection | +0.0105 [-0.0072, +0.0282] | +0.0105 [-0.0072, +0.0282] | +0.0033 [-0.0020, +0.0087] | +0.0079 [-0.0044, +0.0201] | +0.0012 [-0.0123, +0.0148] | +0.0181 [+0.0070, +0.0291]✱ | +0.0208 [+0.0097, +0.0319]✱ | +0.0021 [-0.0203, +0.0244] | +0.0004 [-0.0281, +0.0288] | +0.0097 [-0.0078, +0.0272] | +0.0129 [-0.0053, +0.0311] | +0.0097 [-0.0078, +0.0272] | +0.0129 [-0.0053, +0.0311] |

## Step 1 follow-up (val, L/14, seeds 42–46, checkpoint = min val softCE)

Probe lr chosen on val by mean selected softCE: 1e-3: 1.7555, 3e-4: 1.7579, 1e-4: 1.7670 → **1e-3**. Probe budget: up to 100 epochs, cosine schedule over 100 epochs, patience 10; all other settings as Step 1. MLP and AGF-only use the Step 1 recipe (20 epochs, patience 5, lr 3e-4).

| model | trainable params | sel. epoch (epochs run) | softCE | Δ softCE | Δ KL | Δ macro-F1 | Δ anger+fear AP |
|---|---|---|---|---|---|---|---|
| R1 base | 762,249 | 3.6 (8.6) | 1.7547 ± 0.0019 | – | – | – | – |
| linear probe, own budget (lr 1e-3, chosen on val) | 6,921 | 41.0 (51.0) | 1.7555 ± 0.0000 | +0.0007 [-0.0017, +0.0031] | +0.0007 [-0.0017, +0.0031] | -0.0184 [-0.0248, -0.0120]✱ | +0.0056 [-0.0017, +0.0130] |
| MLP on global (768→722→9) | 561,725 | 13.0 (18.0) | 1.7539 ± 0.0005 | -0.0009 [-0.0029, +0.0012] | -0.0009 [-0.0029, +0.0012] | -0.0097 [-0.0168, -0.0026]✱ | +0.0063 [-0.0010, +0.0135] |
| AGF-only (−both attribute branches, −ECSA) | 149,385 | 6.2 (11.2) | 1.7513 ± 0.0007 | -0.0035 [-0.0056, -0.0013]✱ | -0.0035 [-0.0056, -0.0013]✱ | -0.0097 [-0.0192, -0.0002]✱ | +0.0131 [+0.0063, +0.0198]✱ |
| ref: − both attribute branches (Step 1) | 561,545 | 3.4 (8.4) | 1.7540 ± 0.0014 | -0.0008 [-0.0018, +0.0003] | -0.0008 [-0.0018, +0.0003] | -0.0013 [-0.0110, +0.0084] | +0.0071 [-0.0001, +0.0144] |
| ref: linear probe, shared recipe (Step 1) | 6,921 | 20.0 (20.0) | 1.7772 ± 0.0001 | +0.0225 [+0.0202, +0.0248]✱ | +0.0225 [+0.0202, +0.0248]✱ | -0.0426 [-0.0493, -0.0360]✱ | -0.0264 [-0.0337, -0.0191]✱ |
| linear probe, own budget, lr 3e-4 (not chosen) | 6,921 | 96.4 (100.0) | 1.7579 ± 0.0000 | +0.0031 [+0.0007, +0.0055]✱ | +0.0031 [+0.0007, +0.0055]✱ | -0.0227 [-0.0288, -0.0166]✱ | +0.0015 [-0.0057, +0.0087] |
| linear probe, own budget, lr 1e-4 (not chosen) | 6,921 | 99.2 (100.0) | 1.7670 ± 0.0000 | +0.0122 [+0.0098, +0.0146]✱ | +0.0122 [+0.0098, +0.0146]✱ | -0.0333 [-0.0401, -0.0266]✱ | -0.0115 [-0.0188, -0.0042]✱ |

R1 absolute values: softCE 1.7547, KL 0.7081, macro-F1 0.3576, anger+fear AP 0.4420. Δ = model − R1, same seed; ✱ = 95% interval excludes 0. KL and softCE differences are identical (same targets).

## Step 2. Artist-held-out experiment (L/14 R1, seeds 42–44; no val or test used)

Split (runs/revision_B/step2/split.csv, code/s2_split.py): 10% of the 43K-train artists, sampled within artwork-count strata, form the unseen-artist holdout (all their artworks). A seen-artist control of the same size is drawn at random from the other artists' artworks (each artist keeps ≥ 1 training artwork). A 5% selection split of the remaining artworks replaces val for early stopping / checkpoint selection (min softCE). Training: 32,868 artworks, 961 artists. Recipe otherwise identical to Step 1 R1.

### Holdout composition

| holdout | artworks | artists | unique-top | mean vote entropy | mean vote share (amu awe con exc ang dis fea sad som) |
|---|---|---|---|---|---|
| holdout_unseen | 4,207 | 107 | 3,155 | 1.0831 | 0.121 0.145 0.232 0.089 0.016 0.056 0.108 0.109 0.124 |
| holdout_seen | 4,207 | 732 | 3,116 | 1.0509 | 0.102 0.172 0.297 0.085 0.013 0.039 0.078 0.098 0.116 |

### Metrics per holdout (mean ± SD over 3 seeds) and gap (unseen − seen) with artist-cluster bootstrap 95% CI

| metric | unseen artists | seen-artist control | gap (unseen − seen) | 95% CI |
|---|---|---|---|---|
| softCE | 1.8080 ± 0.0010 | 1.7383 ± 0.0012 | +0.0697 | [+0.0068, +0.1321] |
| KL | 0.7249 ± 0.0010 | 0.6874 ± 0.0012 | +0.0375 | [+0.0030, +0.0724] |
| macro-F1 | 0.3993 ± 0.0024 | 0.3482 ± 0.0077 | +0.0511 | [+0.0055, +0.0813] |
| mean anger+fear AP | 0.4704 ± 0.0040 | 0.4204 ± 0.0010 | +0.0501 | [-0.0125, +0.1022] |
| macro AP (9) | 0.6225 ± 0.0016 | 0.6034 ± 0.0015 | +0.0191 | [-0.0060, +0.0378] |

Per-seed gaps: s42: softCE +0.0709, KL +0.0387, macro-F1 +0.0599; s43: softCE +0.0688, KL +0.0366, macro-F1 +0.0487; s44: softCE +0.0694, KL +0.0372, macro-F1 +0.0448.
