"""Step 1 follow-up table: long-budget linear probe (lr chosen on val), parameter-matched MLP, AGF-only.

Probe lr choice: among {1e-3, 3e-4, 1e-4}, the lr with the lowest mean (over seeds 42-46) selected val softCE.
This is a val-based choice, as requested; the chosen row is therefore optimistically biased like every val number
here. All three lrs are shown. Paired difference vs Step 1 R1 (same seed), mean [95% t-interval, df 4].
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, F
from s1_eval import row

SEEDS = [42, 43, 44, 45, 46]
LRS = ["1e-3", "3e-4", "1e-4"]
METRICS = ["softCE", "KL", "macro-F1", "mean anger+fear AP"]


def load(c):
    out = []
    tv = load.tv
    for s in SEEDS:
        d = OUT / f"step1/{c}_s{s}"
        z = np.load(d / "val_logits.npy").astype(np.float64)
        cfg = json.loads((d / "config.json").read_text())
        h = json.loads((d / "history.json").read_text())
        sel = min(h, key=lambda x: x["sel_soft_ce"])
        out.append({"condition": c, "seed": s, "params": cfg["trainable_parameters"], "epoch": sel["epoch"],
                    "epochs run": len(h), **row(z, tv)})
    return pd.DataFrame(out)


def main():
    load.tv = F.Targets(F.VAL_CSV)
    probes = {lr: load(f"probe100_lr{lr}") for lr in LRS}
    best = min(LRS, key=lambda lr: probes[lr].softCE.mean())
    rows = {"R1": load("R1"), f"probe100_lr{best}": probes[best], "mlp": load("mlp"), "agf_only": load("agf_only"),
            "no_attr": load("no_attr"), "probe": load("probe")}
    for lr in LRS:
        rows.setdefault(f"probe100_lr{lr}", probes[lr])
    df = pd.concat(rows.values())
    df.to_csv(OUT / "s1b_val_per_seed.csv", index=False)
    lab = {"R1": "R1 base", f"probe100_lr{best}": f"linear probe, own budget (lr {best}, chosen on val)",
           "mlp": "MLP on global (768→722→9)", "agf_only": "AGF-only (−both attribute branches, −ECSA)",
           "no_attr": "ref: − both attribute branches (Step 1)", "probe": "ref: linear probe, shared recipe (Step 1)"}
    for lr in LRS:
        lab.setdefault(f"probe100_lr{lr}", f"linear probe, own budget, lr {lr} (not chosen)")
    base = rows["R1"].set_index("seed")
    md = ["## Step 1 follow-up (val, L/14, seeds 42–46, checkpoint = min val softCE)", "",
          f"Probe lr chosen on val by mean selected softCE: {', '.join(f'{lr}: {probes[lr].softCE.mean():.4f}' for lr in LRS)} "
          f"→ **{best}**. Probe budget: up to 100 epochs, cosine schedule over 100 epochs, patience 10; all other "
          "settings as Step 1. MLP and AGF-only use the Step 1 recipe (20 epochs, patience 5, lr 3e-4).", "",
          "| model | trainable params | sel. epoch (epochs run) | softCE | Δ softCE | Δ KL | Δ macro-F1 | Δ anger+fear AP |",
          "|---|---|---|---|---|---|---|---|"]
    for c, g in rows.items():
        g = g.set_index("seed")
        cells = []
        for m in METRICS:
            if c == "R1":
                cells.append("–"); continue
            d = (g[m] - base[m]).loc[SEEDS].to_numpy()
            h = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
            sig = "✱" if (d.mean() - h > 0 or d.mean() + h < 0) else ""
            cells.append(f"{d.mean():+.4f} [{d.mean() - h:+.4f}, {d.mean() + h:+.4f}]{sig}")
        md.append(f"| {lab[c]} | {int(g.params.iloc[0]):,} | {g.epoch.mean():.1f} ({g['epochs run'].mean():.1f}) | "
                  f"{g.softCE.mean():.4f} ± {g.softCE.std(ddof=1):.4f} | " + " | ".join(cells[0:1] + cells[1:]) + " |")
    md += ["", "R1 absolute values: softCE {:.4f}, KL {:.4f}, macro-F1 {:.4f}, anger+fear AP {:.4f}. Δ = model − R1, "
           "same seed; ✱ = 95% interval excludes 0. KL and softCE differences are identical (same targets)."
           .format(*(rows["R1"][m].mean() for m in METRICS))]
    (OUT / "s1b_followup.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
