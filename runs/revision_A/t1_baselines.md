## Task 1. Reference baselines (constant predictors)

Hard metrics on unique-top artworks; softCE, KL, Brier on all artworks, eps = 1e-12 as in artfae_balance.evaluate. The one-hot majority predictor assigns probability 0 to eight classes, so its softCE and KL are dominated by log(1e-12) ≈ −27.6 and are not meaningful as calibration numbers; they are reported only for completeness.

| split | baseline | n | n unique-top | acc | macro-F1 | balAcc | softCE | KL | Brier |
|---|---|---|---|---|---|---|---|---|---|
| val | majority class (contentment, one-hot) | 3000 | 2243 | 0.4062 | 0.0642 | 0.1111 | 19.6108 | 18.5641 | 0.8248 |
| val | mean train vote distribution, 43K | 3000 | 2243 | 0.4062 | 0.0642 | 0.1111 | 1.9963 | 0.9497 | 0.2442 |
| val | mean train vote distribution, paper 14K | 3000 | 2243 | 0.4062 | 0.0642 | 0.1111 | 1.9926 | 0.9460 | 0.2438 |
| test | majority class (contentment, one-hot) | 2999 | 2246 | 0.4056 | 0.0641 | 0.1111 | 19.7797 | 18.7262 | 0.8332 |
| test | mean train vote distribution, 43K | 2999 | 2246 | 0.4056 | 0.0641 | 0.1111 | 2.0080 | 0.9545 | 0.2436 |
| test | mean train vote distribution, paper 14K | 2999 | 2246 | 0.4056 | 0.0641 | 0.1111 | 2.0035 | 0.9501 | 0.2431 |

Mean training vote distributions (class order amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something_else):

- mean train vote distribution, 43K: 0.1014, 0.1675, 0.2931, 0.0849, 0.0127, 0.0408, 0.0825, 0.0994, 0.1177
- mean train vote distribution, paper 14K: 0.0972, 0.1695, 0.2881, 0.0808, 0.0200, 0.0529, 0.0840, 0.1020, 0.1055
