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
