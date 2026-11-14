"""NOT PRE-REGISTERED. Descriptive metrics requested after the Phase C holdout evaluation (2026-11-04).

Reads only the per-run holdout logits saved by eval_holdout.py in runs/final_eval_v3 (no model is loaded or rerun)
and the frozen holdout manifest. Uses existing definitions: artfae_balance.evaluate.evaluate (accuracy, balanced
accuracy and per-class F1 on unique-top artworks; anger presence AUROC on all artworks) and the softCE/KL formulas of
eval_holdout.stat_vec (EPS 1e-12), split by unique-top vs tied artworks. Per seed; reported as mean ± seed SD.
No bootstrap, no tests. Writes runs/final_eval_v3_posthoc/ (create-once).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import EMOTIONS, EPS, ROOT, F, softmax

from artfae_balance.evaluate import evaluate  # noqa: E402

SEL = ROOT / "runs/final_selection_v3.txt"
IN = ROOT / "runs/final_eval_v3"
OUT = ROOT / "runs/final_eval_v3_posthoc"


def main():
    sel = json.loads(SEL.read_text())
    OUT.mkdir(parents=True, exist_ok=False)
    man = ROOT / sel["HOLDOUT"]["manifest"]
    tg = F.Targets(man)
    u, t = tg.uniq, ~tg.uniq
    rows = []
    for m, spec in sel["models"].items():
        for run in spec["runs"]:
            p = softmax(np.load(IN / f"{m}_{Path(run).name}_holdout_logits.npy").astype(np.float64))
            r = evaluate(p, tg.y, tg.present, tg.uniq)
            lp = np.log(p + EPS)
            ce = -(tg.y * lp).sum(1)
            kl = (tg.y * (np.log(tg.y + EPS) - lp)).sum(1)
            row = {"model": m, "run": run, "acc": r["accuracy"], "balAcc": r["balanced_accuracy"],
                   "anger AUROC": r["per_class"]["anger"]["present_auroc"],
                   "softCE unique-top": ce[u].mean(), "softCE tied": ce[t].mean(),
                   "KL unique-top": kl[u].mean(), "KL tied": kl[t].mean()}
            row.update({f"F1 {e}": r["per_class"][e]["f1"] for e in EMOTIONS})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "posthoc_per_seed.csv", index=False)
    agg = df.drop(columns="run").groupby("model", sort=False).agg(["mean", "std"])
    agg.to_csv(OUT / "posthoc_mean_sd.csv")
    print(f"n = {len(tg.y)}, unique-top {int(u.sum())}, tied {int(t.sum())}")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    sys.exit(main())
