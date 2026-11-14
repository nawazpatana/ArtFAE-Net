"""Val table, raw and prior-shift corrected. Val only; never reads test.

Correction (no tuning): logits + log(val_prior) - log(train_prior), where val_prior is the mean val p_*
and train_prior the mean p_* of the model's own training manifest (from its config.json).
softCE(T)/Brier(T): a temperature is refit on val for each row (raw and corrected separately), so those
two columns are in-sample. Sentiment: positive = amusement+awe+contentment+excitement, negative =
anger+disgust+fear+sadness, something_else; hard metrics on artworks with a unique top sentiment.

usage: python scripts/val_prior_table.py   -> runs/val_prior_table.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artfae_balance import EMOTIONS  # noqa: E402
from artfae_balance.evaluate import evaluate, load_targets, softmax  # noqa: E402

GROUPS = [  # label, run prefix, seeds
    ("R0n  B/32 paper 14K (check)", "R0n_paper14k", [42, 43, 44, 45, 46]),
    ("R1   B/32", "R1", [42, 43, 44, 45, 46]),
    ("R3   B/32", "R3_lam0.3", [42, 43, 44, 45, 46]),
    ("R4b  B/32", "R4b_textq_gate", [42, 43, 44, 45, 46]),
    ("R1   L/14@336", "L14_R1", [42, 43, 44]),
    ("R3   L/14@336", "L14_R3_lam0.3", [42, 43, 44]),
    ("R1   B/32, seeds 42-44", "R1", [42, 43, 44]),
    ("R3   B/32, seeds 42-44", "R3_lam0.3", [42, 43, 44]),
]
SENT = [["amusement", "awe", "contentment", "excitement"], ["anger", "disgust", "fear", "sadness"], ["something_else"]]
COLS = ["macro-F1", "acc", "balAcc", "softCE(T)", "Brier(T)", "anger AP", "fear AP", "mean anger+fear AP",
        "disgust F1", "neg-sentiment F1", "sentiment macro-F1"]

_, Y, PRESENT, UNIQ = load_targets(ROOT / "data/artemis_balanced/val.csv")
LOG_VAL_PRIOR = np.log(Y.mean(0) + 1e-8)
SIDX = [[EMOTIONS.index(e) for e in g] for g in SENT]
YS = np.stack([Y[:, i].sum(1) for i in SIDX], 1)
SU = np.isclose(YS, YS.max(1, keepdims=True), atol=1e-5).sum(1) == 1


def fit_t(logits):
    ce = lambda lt: -(Y * np.log(softmax(logits / np.exp(lt)) + 1e-12)).sum(1).mean()
    return float(np.exp(minimize_scalar(ce, bounds=(-3, 3), method="bounded").x))


def metrics(logits):
    p = softmax(logits)
    r = evaluate(p, Y, PRESENT, UNIQ)
    pt = softmax(logits / fit_t(logits))
    ps = np.stack([p[:, i].sum(1) for i in SIDX], 1)
    sf = f1_score(YS[SU].argmax(1), ps[SU].argmax(1), labels=[0, 1, 2], average=None, zero_division=0)
    c = r["per_class"]
    return [r["macro_f1"], r["accuracy"], r["balanced_accuracy"],
            float(-(Y * np.log(pt + 1e-12)).sum(1).mean()), float(((pt - Y) ** 2).sum(1).mean()),
            c["anger"]["present_ap"], c["fear"]["present_ap"], (c["anger"]["present_ap"] + c["fear"]["present_ap"]) / 2,
            c["disgust"]["f1"], sf[1], sf.mean()]


def main():
    lines = ["# Val table, raw vs prior-corrected (val only)", "",
             "Correction: logits + log(val prior) − log(train prior); no tuning. T refit on val per row.", "",
             "| model | n | logits | shift ‖Δ‖∞ | " + " | ".join(COLS) + " |",
             "|---|---|---|---|" + "---|" * len(COLS)]
    for label, prefix, seeds in GROUPS:
        res = {"raw": [], "prior-corrected": []}
        shifts = []
        for s in seeds:
            d = ROOT / "runs" / f"{prefix}_s{s}"
            logits = np.load(d / "val_logits.npy").astype(np.float64)
            tr = pd.read_csv(ROOT / json.loads((d / "config.json").read_text())["train_manifest"])
            delta = LOG_VAL_PRIOR - np.log(tr[[f"p_{e}" for e in EMOTIONS]].to_numpy().mean(0) + 1e-8)
            shifts.append(np.abs(delta).max())
            res["raw"].append(metrics(logits))
            res["prior-corrected"].append(metrics(logits + delta))
        for kind, rows in res.items():
            a = np.array(rows)
            cells = [f"{a[:, j].mean():.4f}" + (f" ± {a[:, j].std(ddof=1):.4f}" if len(a) > 1 else "")
                     for j in range(len(COLS))]
            sh = f"{np.mean(shifts):.3f}" if kind != "raw" else "–"
            lines.append(f"| {label} | {len(seeds)} | {kind} | {sh} | " + " | ".join(cells) + " |")
    out = "\n".join(lines) + "\n"
    (ROOT / "runs/val_prior_table.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
