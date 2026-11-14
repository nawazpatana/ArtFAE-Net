# Anger analysis (annotations + saved test logits only)

## 1. Anger in the votes

#### New 43K train (43,012 artworks)

- anger is the unique plurality: **119** (0.28%); anger tied for the top: 510
- artworks with ≥1 anger vote: 2,769 (6.4%); mean anger share among them: **0.198** (median 0.200, mean votes per artwork 7.5)
- artworks with ≥2 anger votes: 320

Artworks with ≥2 anger votes (n=320): which emotion wins, and the mean vote share of each emotion on those artworks

| emotion | wins (unique top) | share of wins | mean vote share |
|---|---|---|---|
| amusement | 9 | 2.8% | 0.074 |
| awe | 4 | 1.2% | 0.061 |
| contentment | 15 | 4.7% | 0.094 |
| excitement | 3 | 0.9% | 0.060 |
| anger | 119 | 37.2% | 0.323 |
| disgust | 8 | 2.5% | 0.069 |
| fear | 20 | 6.2% | 0.114 |
| sadness | 15 | 4.7% | 0.083 |
| something_else | 24 | 7.5% | 0.122 |
| tie | 103 | 32.2% | – |

#### Test (2,999 artworks)

- anger is the unique plurality: **14** (0.47%); anger tied for the top: 61
- artworks with ≥1 anger vote: 285 (9.5%); mean anger share among them: **0.209** (median 0.200, mean votes per artwork 5.9)
- artworks with ≥2 anger votes: 30

Artworks with ≥2 anger votes (n=30): which emotion wins, and the mean vote share of each emotion on those artworks

| emotion | wins (unique top) | share of wins | mean vote share |
|---|---|---|---|
| amusement | 2 | 6.7% | 0.066 |
| awe | 0 | 0.0% | 0.087 |
| contentment | 0 | 0.0% | 0.049 |
| excitement | 0 | 0.0% | 0.042 |
| anger | 14 | 46.7% | 0.360 |
| disgust | 0 | 0.0% | 0.033 |
| fear | 2 | 6.7% | 0.131 |
| sadness | 2 | 6.7% | 0.156 |
| something_else | 0 | 0.0% | 0.074 |
| tie | 10 | 33.3% | – |

## 2. The 14 unique-top anger test artworks

Rank of anger in the seed-averaged predicted distribution (R0: single checkpoint) and the predicted class instead.

| artwork | anger votes / total | anger share | 2nd emotion (votes) | R0 rank | R1 rank | R1 predicts | R4b rank | R4b predicts | R4b p(anger) |
|---|---|---|---|---|---|---|---|---|---|
| ivan-bilibin_king-of-the-seas-1911 | 2 / 6 | 0.33 | awe (1) | 7 | 9 | awe | 9 | awe | 0.011 |
| wilhelm-kotarbinski_good | 2 / 5 | 0.40 | awe (1) | 9 | 9 | awe | 9 | awe | 0.002 |
| esaias-van-de-velde_the-robbery-1616 | 2 / 5 | 0.40 | contentment (1) | 9 | 9 | awe | 9 | contentment | 0.014 |
| jamie-wyeth_the-rookery-1977 | 2 / 6 | 0.33 | sadness (1) | 9 | 9 | fear | 9 | fear | 0.010 |
| pyotr-konchalovsky_portrait-of-actress-vizarova-1917 | 4 / 10 | 0.40 | fear (2) | 5 | 9 | fear | 9 | fear | 0.007 |
| pyotr-konchalovsky_bullfight-study-1910 | 2 / 5 | 0.40 | excitement (1) | 9 | 9 | fear | 9 | fear | 0.037 |
| frederic-bazille_self-portrait-1868 | 2 / 5 | 0.40 | disgust (1) | 7 | 9 | contentment | 9 | something_else | 0.031 |
| max-liebermann_bethmann-hollweg | 2 / 5 | 0.40 | excitement (1) | 3 | 5 | fear | 2 | fear | 0.178 |
| tintoretto_battle-between-turks-and-christians-1589 | 2 / 5 | 0.40 | sadness (1) | 6 | 8 | fear | 9 | fear | 0.022 |
| giovanni-battista-tiepolo_maecenas-presenting-the-liberal-arts-to-emperor-augustus-1743 | 2 / 5 | 0.40 | awe (1) | 9 | 9 | awe | 9 | awe | 0.009 |
| nicholas-roerich_monhigan-study-1922-8 | 2 / 5 | 0.40 | amusement (1) | 7 | 8 | something_else | 2 | something_else | 0.146 |
| utagawa-kuniyoshi_rape-scene | 3 / 6 | 0.50 | fear (2) | 9 | 9 | amusement | 8 | amusement | 0.035 |
| utagawa-sadatora_oguri-hangan-sukeshige-and-yamasaki | 2 / 5 | 0.40 | awe (1) | 6 | 5 | fear | 2 | fear | 0.134 |
| utagawa-toyokuni_actor-bando-mitsugoro-iii-as-ko-no-moronao-1814 | 3 / 6 | 0.50 | amusement (2) | 8 | 9 | amusement | 9 | amusement | 0.030 |

| model | anger ranked 1st | 2nd | 3rd | lower | predicted instead (counts) |
|---|---|---|---|---|---|
| R0 | 0 | 0 | 1 | 13 | awe 4, fear 4, contentment 2, amusement 2, disgust 1, something_else 1 |
| R1 | 0 | 0 | 0 | 14 | fear 6, awe 4, amusement 2, contentment 1, something_else 1 |
| R4b | 0 | 3 | 0 | 11 | fear 6, awe 3, something_else 2, amusement 2, contentment 1 |

## 3. Anger presence detection on test (285 artworks with ≥1 anger vote; chance AP = 0.095)

| model | AUROC (seed-averaged) | AP (seed-averaged) | AUROC per seed, mean ± SD | AP per seed, mean ± SD |
|---|---|---|---|---|
| R0 | 0.6940 | 0.1895 | 0.6940 | 0.1895 |
| R1 | 0.7160 | 0.2168 | 0.7018 ± 0.0174 | 0.2075 ± 0.0142 |
| R4b | 0.6978 | 0.2191 | 0.6782 ± 0.0142 | 0.1926 ± 0.0182 |

## 4. Sentiment level (positive = amusement+awe+contentment+excitement; negative = anger+disgust+fear+sadness; something_else)

Targets are the summed vote shares; hard metrics on the 2,714 artworks with a unique top sentiment (of 2,999). Predictions sum the model's class probabilities per sentiment. Per-seed mean ± SD; R0 is a single checkpoint.

| model | macro-F1 | accuracy | F1 positive | F1 negative | F1 something_else | soft CE (3-way) |
|---|---|---|---|---|---|---|
| R0 | 0.4752 | 0.7948 | 0.8702 | 0.5554 | 0.0000 | 0.8127 |
| R1 | 0.4908 ± 0.0091 | 0.8169 ± 0.0032 | 0.8876 ± 0.0028 | 0.5504 ± 0.0176 | 0.0344 ± 0.0193 | 0.8031 ± 0.0065 |
| R4b | 0.5081 ± 0.0095 | 0.8099 ± 0.0040 | 0.8809 ± 0.0031 | 0.5787 ± 0.0096 | 0.0648 ± 0.0334 | 0.8052 ± 0.0068 |

Test sentiment support (unique-top): positive 2056, negative 614, something_else 44
