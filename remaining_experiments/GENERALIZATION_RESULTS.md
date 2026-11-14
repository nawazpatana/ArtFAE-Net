# Saved-result generalization analysis (descriptive only)

No model was loaded. Existing holdout logits and existing style/artist membership columns were read; no new holdout predictions, metadata mapping or split was generated. Seed means match the locked main CE values. Groups are not new confirmatory hypotheses. Artwork styles and artist overlap differ in annotation entropy and composition, so metric gaps are not causal effects of style or artist novelty. Tiny groups are saved but not promoted as reliable estimates.

| Model | Group | N | Unique-top | CE | KL | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| R1 | population: all | 30280 | 22351 | 1.8357 | 0.7249 | 0.3607 |
| R1 | top: unique | 22351 | 22351 | 1.7813 | 0.7226 | 0.3607 |
| R1 | top: tied | 7929 | 0 | 1.9890 | 0.7315 | not applicable |
| R1 | artist: seen_in_train | 29035 | 21422 | 1.8378 | 0.7255 | 0.3616 |
| R1 | artist: absent_from_train | 1245 | 929 | 1.7863 | 0.7122 | 0.3285 |
| R1 | style: Abstract_Expressionism | 1222 | 844 | 2.0096 | 0.7876 | 0.2527 |
| R1 | style: Art_Nouveau_Modern | 1728 | 1286 | 1.8778 | 0.7340 | 0.3090 |
| R1 | style: Baroque | 1222 | 892 | 1.8623 | 0.7373 | 0.3291 |
| R1 | style: Color_Field_Painting | 537 | 389 | 1.9561 | 0.7452 | 0.2036 |
| R1 | style: Cubism | 797 | 563 | 1.9769 | 0.7765 | 0.2764 |
| R1 | style: Early_Renaissance | 595 | 439 | 1.9072 | 0.7506 | 0.2656 |
| R1 | style: Expressionism | 2751 | 1998 | 1.9090 | 0.7642 | 0.3443 |
| R1 | style: High_Renaissance | 579 | 401 | 1.9242 | 0.7758 | 0.2760 |
| R1 | style: Impressionism | 4602 | 3506 | 1.7319 | 0.6882 | 0.2399 |
| R1 | style: Mannerism_Late_Renaissance | 508 | 378 | 1.8898 | 0.7629 | 0.2917 |
| R1 | style: Minimalism | 512 | 389 | 1.9012 | 0.7478 | 0.2721 |
| R1 | style: Naive_Art_Primitivism | 1091 | 820 | 1.8232 | 0.7167 | 0.3052 |
| R1 | style: Northern_Renaissance | 1001 | 732 | 1.9054 | 0.7747 | 0.3162 |
| R1 | style: Pop_Art | 601 | 414 | 1.9313 | 0.7704 | 0.3205 |
| R1 | style: Post_Impressionism | 2319 | 1735 | 1.7666 | 0.6905 | 0.2574 |
| R1 | style: Realism | 3909 | 2946 | 1.7690 | 0.6899 | 0.3079 |
| R1 | style: Rococo | 627 | 460 | 1.9038 | 0.7524 | 0.2561 |
| R1 | style: Romanticism | 2501 | 1859 | 1.7776 | 0.6987 | 0.3154 |
| R1 | style: Symbolism | 1727 | 1275 | 1.8184 | 0.7250 | 0.3023 |
| PROBE | population: all | 30280 | 22351 | 1.8359 | 0.7252 | 0.3558 |
| PROBE | top: unique | 22351 | 22351 | 1.7813 | 0.7227 | 0.3558 |
| PROBE | top: tied | 7929 | 0 | 1.9898 | 0.7323 | not applicable |
| PROBE | artist: seen_in_train | 29035 | 21422 | 1.8381 | 0.7258 | 0.3563 |
| PROBE | artist: absent_from_train | 1245 | 929 | 1.7850 | 0.7110 | 0.3326 |

Complete per-style and per-seed tables are in the CSVs. The historical 32,868-training-artist experiment remains a separate protocol, not evidence newly obtained on the main training setup. No external collection or corruption inference was conducted.
