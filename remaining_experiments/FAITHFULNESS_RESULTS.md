# Exploratory validation perturbation results

| fill | curve | ranking | mean_partial_AUC | CI95_low | CI95_high |
|---|---|---|---|---|---|
| mean | deletion | ECSA | 0.2199 | 0.2065 | 0.2336 |
| mean | deletion | RISE | 0.2301 | 0.2165 | 0.2440 |
| mean | deletion | Random | 0.2519 | 0.2395 | 0.2643 |
| mean | retention | ECSA | 0.2581 | 0.2458 | 0.2713 |
| mean | retention | RISE | 0.2480 | 0.2347 | 0.2619 |
| mean | retention | Random | 0.2009 | 0.1894 | 0.2126 |
| blur | deletion | ECSA | 0.2015 | 0.1891 | 0.2139 |
| blur | deletion | RISE | 0.2148 | 0.2017 | 0.2284 |
| blur | deletion | Random | 0.2386 | 0.2270 | 0.2510 |
| blur | retention | ECSA | 0.2547 | 0.2418 | 0.2682 |
| blur | retention | RISE | 0.2401 | 0.2271 | 0.2538 |
| blur | retention | Random | 0.1967 | 0.1855 | 0.2087 |

Paired advantages (positive favours stated ranking):

| fill | curve | contrast | mean | CI95_low | CI95_high |
|---|---|---|---|---|---|
| mean | deletion | ECSA_advantage_over_random | 0.0320 | 0.0240 | 0.0400 |
| mean | deletion | RISE_advantage_over_random | 0.0218 | 0.0149 | 0.0283 |
| mean | deletion | ECSA_advantage_over_RISE | 0.0102 | 0.0037 | 0.0171 |
| mean | retention | ECSA_advantage_over_random | 0.0572 | 0.0492 | 0.0657 |
| mean | retention | RISE_advantage_over_random | 0.0471 | 0.0397 | 0.0547 |
| mean | retention | ECSA_advantage_over_RISE | 0.0101 | 0.0024 | 0.0178 |
| blur | deletion | ECSA_advantage_over_random | 0.0372 | 0.0298 | 0.0449 |
| blur | deletion | RISE_advantage_over_random | 0.0238 | 0.0171 | 0.0302 |
| blur | deletion | ECSA_advantage_over_RISE | 0.0133 | 0.0068 | 0.0199 |
| blur | retention | ECSA_advantage_over_random | 0.0580 | 0.0498 | 0.0666 |
| blur | retention | RISE_advantage_over_random | 0.0434 | 0.0363 | 0.0502 |
| blur | retention | ECSA_advantage_over_RISE | 0.0146 | 0.0073 | 0.0219 |

No endpoints were measured or imputed. This is raw-probability partial AUC over [5/49,24/49], normalized by range width. Intervals are pointwise, paired-artwork and conditional on one fixed checkpoint, selected validation images and generated mask rankings. Whole-pipeline features are recomputed; a perturbation result does not isolate causal ECSA or human-semantic evidence. RISE-style uses 256 masks and is not claimed converged.
