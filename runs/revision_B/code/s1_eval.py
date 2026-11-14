"""Step 1 tables: val metrics per condition (mean ± SD over seeds 42-46) and paired differences from R1.

All metrics on the 3,000 val artworks (val is also the checkpoint-selection split, so these are selection-split
numbers, not held-out estimates). softCE / KL / Brier at T = 1 on all artworks; softCE(T) with T fitted on val
logits of the same run (in-sample, as in final_eval_v2's val table); acc / macro-F1 / balAcc on unique-top
artworks; mean anger+fear presence AP on all artworks; negative-sentiment F1 on artworks with a unique top
sentiment (final_eval_v2.metrics). softCE and KL also split into unique-top (2,243) and tied (757) artworks.
Paired difference: condition − R1 for the same seed; mean, SD and 95% t-interval (df = 4).
Reference row: the existing runs/L14_R1_s* (same recipe, checkpoint selected by val macro-F1).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import EPS, OUT, ROOT, F, fmt, softmax

COND = {"R1": "R1 base (L/14 R1, unweighted)", "probe": "CLIP-only linear probe (global)",
        "B5": "B5 concatenation baseline", "no_num": "− numerical attributes", "no_text": "− attribute text",
        "no_attr": "− both attribute branches", "no_ecsa": "− ECSA (no regional term)",
        "no_agf": "− AGF (concatenation, no gate)", "weighted": "class-weighted soft CE (p 0.25, [0.5, 2])",
        "crop": "attributes on 336-px center crop", "ref_mf1": "ref: existing L/14 R1, macro-F1 selection"}
SEEDS = [42, 43, 44, 45, 46]
METRICS = ["softCE", "KL", "Brier", "softCE(T)", "acc", "macro-F1", "balAcc", "mean anger+fear AP",
           "neg-sentiment F1", "softCE unique-top", "softCE tied", "KL unique-top", "KL tied"]
LOWER = {"softCE", "KL", "Brier", "softCE(T)", "softCE unique-top", "softCE tied", "KL unique-top", "KL tied"}


def row(z, tv):
    t = F.fit_t(z, tv.y)
    m = F.metrics(z, t, tv)
    p = softmax(z)
    ce = -(tv.y * np.log(p + EPS)).sum(1)
    kl = (tv.y * (np.log(tv.y + EPS) - np.log(p + EPS))).sum(1)
    br = ((p - tv.y) ** 2).sum(1)
    u = tv.uniq
    return {"softCE": ce.mean(), "KL": kl.mean(), "Brier": br.mean(), "softCE(T)": m["softCE(T)"], "T": t,
            "acc": m["acc"], "macro-F1": m["macro-F1"], "balAcc": m["balAcc"],
            "mean anger+fear AP": m["mean anger+fear AP"], "neg-sentiment F1": m["neg-sentiment F1"],
            "softCE unique-top": ce[u].mean(), "softCE tied": ce[~u].mean(),
            "KL unique-top": kl[u].mean(), "KL tied": kl[~u].mean()}


def main():
    tv = F.Targets(F.VAL_CSV)
    rows = []
    for c in COND:
        for s in SEEDS:
            if c == "ref_mf1":
                z = np.load(ROOT / f"runs/L14_R1_s{s}/val_logits.npy").astype(np.float64)
                params, ep = 762249, json.loads((ROOT / f"runs/L14_R1_s{s}/history.json").read_text())
                ep = max(ep, key=lambda h: h["val_macro_f1"])["epoch"]
            else:
                d = OUT / f"step1/{c}_s{s}"
                z = np.load(d / "val_logits.npy").astype(np.float64)
                cfg = json.loads((d / "config.json").read_text())
                params = cfg["trainable_parameters"]
                ep = min(json.loads((d / "history.json").read_text()), key=lambda h: h["sel_soft_ce"])["epoch"]
            rows.append({"condition": c, "seed": s, "params": params, "epoch": ep, **row(z, tv)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "s1_val_per_seed.csv", index=False)

    md = ["## Step 1. Controlled L/14 ablations (val, seeds 42–46, checkpoint = min val softCE)", "",
          f"Val n = 3,000 ({int(tv.uniq.sum())} unique-top, {int((~tv.uniq).sum())} tied). Val is also the "
          "selection split, so all numbers are selection-split estimates. softCE(T): T refit on val per run "
          "(in-sample). 16 val artworks are confirmed reproductions of 43K-train artworks (Step 0); they are "
          "included here, as in every earlier val table.", "",
          "### Val metrics, mean ± SD over 5 seeds", "",
          "| condition | trainable params | sel. epoch | " + " | ".join(METRICS[:9]) + " |",
          "|---|---|---|" + "---|" * 9]
    for c, lab in COND.items():
        g = df[df.condition == c]
        md.append(f"| {lab} | {int(g.params.iloc[0]):,} | {g.epoch.mean():.1f} | " +
                  " | ".join(fmt(g[m]) for m in METRICS[:9]) + " |")
    md += ["", "### softCE and KL on unique-top vs tied val artworks (T = 1), mean ± SD", "",
           "| condition | softCE unique-top | softCE tied | KL unique-top | KL tied |", "|---|---|---|---|---|"]
    for c, lab in COND.items():
        g = df[df.condition == c]
        md.append(f"| {lab} | " + " | ".join(fmt(g[m]) for m in METRICS[9:]) + " |")

    md += ["", "### Paired difference from R1 (condition − R1, same seed): mean [95% t-interval, df 4]", "",
           "Lower is better for softCE/KL/Brier; higher for the rest. ✱ = 95% interval excludes 0.", "",
           "| condition | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    base = df[df.condition == "R1"].set_index("seed")
    drows = []
    for c, lab in COND.items():
        if c == "R1":
            continue
        g = df[df.condition == c].set_index("seed")
        cells = []
        for m in METRICS:
            d = (g[m] - base[m]).loc[SEEDS].to_numpy()
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            sig = "✱" if (d.mean() - h > 0 or d.mean() + h < 0) else ""
            cells.append(f"{d.mean():+.4f} [{d.mean() - h:+.4f}, {d.mean() + h:+.4f}]{sig}")
            drows.append({"condition": c, "metric": m, "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
                          "ci_lo": d.mean() - h, "ci_hi": d.mean() + h, **{f"s{s}": v for s, v in zip(SEEDS, d)}})
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    pd.DataFrame(drows).to_csv(OUT / "s1_paired_diff.csv", index=False)
    (OUT / "s1_ablations.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
