# Fresh holdout composition (/workspace/ArtFAE-Net/runs/revision_B/fresh_holdout)

Holdout: 30,280 artworks; 22,351 unique-top; tied-vote fraction 0.262 (natural ArtEmis: 0.254 of 80,031).

| class | holdout unique-top n | holdout unique-top % | natural unique-top % | holdout mean share | natural mean share |
|---|---:|---:|---:|---:|---:|
| amusement | 2,039 | 9.1 | 9.0 | 0.098 | 0.100 |
| awe | 3,099 | 13.9 | 14.7 | 0.150 | 0.161 |
| contentment | 8,245 | 36.9 | 39.5 | 0.253 | 0.277 |
| excitement | 1,077 | 4.8 | 5.0 | 0.079 | 0.082 |
| anger | 98 | 0.4 | 0.4 | 0.017 | 0.015 |
| disgust | 778 | 3.5 | 2.8 | 0.060 | 0.049 |
| fear | 2,372 | 10.6 | 9.4 | 0.105 | 0.092 |
| sadness | 2,606 | 11.7 | 10.6 | 0.121 | 0.108 |
| something_else | 2,037 | 9.1 | 8.5 | 0.117 | 0.117 |

Exclusions:
- official_artworks_aggregated: 80031
- ambiguous painting name: stems: 8
- ambiguous painting name: artworks: 528
- already used: 49024
- unused, no image in the WikiArt folder: 0
- undecodable: 0
- duplicate rule: 199
- duplicate rule by class: {"duplicate rule: pair classified 'same artwork, different reproduction'": 149, "duplicate rule: pair classified 'uncertain'": 33, "duplicate rule: pair classified 'same artwork'": 17}

Artists: 1,075 in the holdout; 39 absent from the 43K train, with 1,245 artworks (929 unique-top).
