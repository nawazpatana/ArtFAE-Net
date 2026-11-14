## Task 5. Parameter counts

nn.Parameters only (buffers excluded). One row per distinct architecture within a model family; `n ckpt` = number of checkpoints with exactly these counts. Heads on cached features add the frozen extractor at inference (last rows). Per-checkpoint list: t5_params_all_checkpoints.csv.

| model family | n ckpt | total params | trainable params | note |
|---|---|---|---|---|
| runs/L14_R1 | 5 | 762,249 | 762,249 |  |
| runs/L14_R3_lam0.3 | 5 | 763,410 | 763,410 |  |
| runs/R0n_paper14k | 5 | 631,177 | 631,177 |  |
| runs/R1 | 5 | 631,177 | 631,177 |  |
| runs/R2_tau0.5 | 1 | 631,177 | 631,177 |  |
| runs/R2_tau1.0 | 1 | 631,177 | 631,177 |  |
| runs/R2a_tau1.0 | 1 | 631,177 | 631,177 |  |
| runs/R2b_sampler | 5 | 631,177 | 631,177 |  |
| runs/R3_lam0.1 | 1 | 632,338 | 632,338 |  |
| runs/R3_lam0.3 | 5 | 632,338 | 632,338 |  |
| runs/R3g_gate | 5 | 632,339 | 632,339 |  |
| runs/R4a_textq | 1 | 697,874 | 697,874 |  |
| runs/R4b_textq_gate | 5 | 697,875 | 697,875 |  |
| workspace B3 | 3 | 140,809 | 140,809 |  |
| workspace B4 | 3 | 264,713 | 264,713 |  |
| workspace B5 | 3 | 271,881 | 271,881 |  |
| workspace B2 (CLIP B/32 linear head) | 1 | 4,617 | 4,617 |  |
| workspace B1 (ResNet50, frozen backbone) | 1 | 23,526,473 | 18,441 |  |
| home artfae_m1 M1 | 1 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical,text (paper M1, runs/m1_paper) |
| home artfae_m1 M1 | 8 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical,text |
| home artfae_m1 M1 | 2 | 241,865 | 241,865 | hidden 64, branches: gated,spatial,numerical,text |
| home artfae_m1 M1 | 1 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,text |
| home artfae_m1 M1 | 1 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical |
| home artfae_m1 M1 | 1 | 284,553 | 284,553 | hidden 128, branches: gated,numerical,text |
| home artfae_m1 M1 | 1 | 532,361 | 532,361 | hidden 128, branches: spatial,numerical,text |
| home baseline_simple_fusion B5 | 2 | 271,881 | 271,881 | hidden 128, branches: gated,spatial,numerical,text |
| home matched_component_runs M1 | 2 | 532,361 | 532,361 | hidden 128, branches: spatial,numerical,text |
| home matched_component_runs M1 | 2 | 284,553 | 284,553 | hidden 128, branches: gated,numerical,text |
| home matched_component_runs M1 | 2 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,text |
| home matched_component_runs M1 | 2 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical |
| home research_baselines M1 | 2 | 631,177 | 631,177 | hidden 128, branches: gated,spatial,numerical,text |
| home B2 (CLIP B/32 linear head) | 1 | 4,617 | 4,617 |  |
| home pipeline 14K clip_l14 | 2 | 303,202,313 | 151,179,273 | clip_l14, frozen_blocks=12/24 |
| home pipeline 14K siglip2_so400m | 2 | 428,250,953 | 228,619,065 | siglip2_so400m, frozen_blocks=13/27 |
| home pipeline 14K bundle current | 1 | 1,462,906,532 | 759,596,676 | 4 members: siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24; siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24 |
| home pipeline 14K bundle ensemble4 | 1 | 1,462,906,532 | 759,596,676 | 4 members: siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24; siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24 |
| home pipeline 14K bundle siglip_clipL | 1 | 731,453,266 | 379,798,338 | 2 members: siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24 |
| home pipeline 43K clip_l14 | 2 | 303,202,313 | 151,179,273 | clip_l14, frozen_blocks=12/24 |
| home pipeline 43K siglip2_so400m | 2 | 428,250,953 | 228,619,065 | siglip2_so400m, frozen_blocks=13/27 |
| home pipeline 43K bundle ensemble4 | 1 | 1,462,906,532 | 759,596,676 | 4 members: siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24; siglip2_so400m, frozen_blocks=13/27; clip_l14, frozen_blocks=12/24 |
| openai/clip-vit-base-patch32 | 1 | 151,277,312 | 0 | vision+proj 87,849,216; text+proj 63,428,096 (text encodes the describe() sentence); logit_scale excluded |
| openai/clip-vit-large-patch14-336 | 1 | 427,944,192 | 0 | vision+proj 304,293,888; text+proj 123,650,304 (text encodes the describe() sentence); logit_scale excluded |

Inference-time totals for the Phase-A models (head + frozen extractor):

- R0n / B/32 R1: head 631,177 + clip-vit-base-patch32 151,277,312 = 151,908,489 (trainable 631,177 = 0.42%)
- L/14 R1: head 762,249 + clip-vit-large-patch14-336 427,944,192 = 428,706,441 (trainable 762,249 = 0.18%)
- L/14 R3 / R3c: head 763,410 + clip-vit-large-patch14-336 427,944,192 = 428,707,602 (trainable 763,410 = 0.18%)
