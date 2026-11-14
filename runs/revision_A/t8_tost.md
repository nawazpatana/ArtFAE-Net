## Task 8. Equivalence test (TOST), comparison (d) L/14 R3c − L/14 R3, macro-F1, margin ±0.01

Primary: the existing final_eval_v2 bootstrap (seed-averaged probabilities, 2,000 paired artwork resamples, rng seed 0), regenerated and checked to reproduce bootstrap.json exactly. Equivalence at α = 0.05 requires the 90% CI to lie inside (−0.01, +0.01).

| analysis | Δ macro-F1 | 90% CI | 95% CI | p (H0: Δ ≤ −0.01) | p (H0: Δ ≥ +0.01) | p TOST | equivalent? |
|---|---|---|---|---|---|---|---|
| bootstrap, 5-seed ensemble (primary) | -0.0112 | [-0.0232, +0.0007] | [-0.0260, +0.0026] | 0.4958 | 0.0010 | 0.4958 | no |
| paired t over 5 seeds (secondary, df = 4) | +0.0020 | [-0.0054, +0.0094] | – | 0.0131 | 0.0419 | 0.0419 | yes |

Share of bootstrap differences inside ±0.01: 0.504; at or below −0.01: 0.495. Per-seed differences: s42 +0.0053, s43 -0.0087, s44 -0.0000, s45 +0.0126, s46 +0.0009.
