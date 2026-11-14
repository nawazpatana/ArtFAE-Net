"""Stage table from runs/<run>/val_report.json (artfae_balance.evaluate output on val.csv).

Calibrated columns: one temperature T per run, fit on the val logits to minimise val soft CE
(all artworks), then softCE/Brier recomputed on softmax(logits / T). T is fit and scored on the
same val split, so the calibrated columns are in-sample; argmax metrics do not depend on T.

usage: python scripts/val_table.py runs/m1_paper runs/R1_s42 ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artfae_balance import EMOTIONS  # noqa: E402
from artfae_balance.evaluate import softmax  # noqa: E402

VAL = pd.read_csv(Path(__file__).resolve().parents[1] / "data/artemis_balanced/val.csv")
Y = VAL[[f"p_{e}" for e in EMOTIONS]].to_numpy(float)


def soft_ce(p):
    return float(-(Y * np.log(p + 1e-12)).sum(1).mean())


def temperature(logits):
    res = minimize_scalar(lambda lt: soft_ce(softmax(logits / np.exp(lt))), bounds=(-3, 3), method="bounded")
    t = float(np.exp(res.x))
    p = softmax(logits / t)
    return t, soft_ce(p), float(((p - Y) ** 2).sum(1).mean())


def row(run):
    r = json.loads((Path(run) / "val_report.json").read_text())
    t, ce_t, brier_t = temperature(np.load(Path(run) / "val_logits.npy").astype(np.float64))
    c = r["per_class"]
    ap_mean = (c["anger"]["present_ap"] + c["fear"]["present_ap"]) / 2
    return (f"| {Path(run).name} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | "
            f"{r['soft_ce']:.4f} | {r['brier']:.4f} | {t:.3f} | {ce_t:.4f} | {brier_t:.4f} | "
            f"{c['anger']['f1']:.3f} / {c['anger']['n_predicted']} / {c['anger']['present_ap']:.3f} | "
            f"{c['fear']['f1']:.3f} / {c['fear']['n_predicted']} / {c['fear']['present_ap']:.3f} | "
            f"{c['disgust']['f1']:.3f} | {ap_mean:.3f} |")


print("| run | macro-F1 | acc | balAcc | softCE | Brier | T | softCE (T) | Brier (T) "
      "| anger F1 / #pred / AP | fear F1 / #pred / AP | disgust F1 | mean anger+fear AP |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for run in sys.argv[1:]:
    print(row(run))
