# Exploratory paired-attribute permutation

{
  "same_template_fraction_after_pair_permutation": 0.17811666666666667,
  "scope": "EXPLORATORY development validation; five fixed checkpoints, 20 paired derangements; no encoder, training or holdout inference",
  "baseline": {
    "softCE": 1.7547468057920006,
    "macroF1": 0.3576472639988014
  },
  "paired_permuted": {
    "softCE": 1.7605926237838616,
    "macroF1": 0.35204031272386255,
    "top1_flip_fraction": 0.06713333333333332,
    "mean_total_variation": 0.03762914253297253
  },
  "delta_softCE": 0.005845818145965501,
  "delta_softCE_paired_artwork_CI95": [
    0.004410098179547378,
    0.007290444907728912
  ],
  "delta_macroF1": -0.005606951274938876,
  "max_original_saved_logit_difference": 9.5367431640625e-07,
  "CI_scope": "Conditional on these five checkpoints and 20 fixed permutations; not training-seed or permutation-population uncertainty",
  "protocol_sha256": "82d041be51241d31cd8e05b7d3452956f8046803f102231a771c28de56c23a32"
}

Five fixed R1 checkpoints and 20 derangements of all 3,000 validation rows. Numerical/text inputs move together; global/patch inputs and all parameters remain fixed. CE intervals resample paired artworks conditional on these checkpoints/permutations; no new H1–H5 decision. Misaligned inputs can be off the natural joint distribution. Sensitivity is not evidence that retraining with the branch improves prediction, nor human causality. Some templates coincide after permutation because only 27 descriptions exist. Original logits reproduce the saved validation outputs. All predictions and assignments are saved.
