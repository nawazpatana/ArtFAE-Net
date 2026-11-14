"""Shared loaders for revision Phase A. Read-only on runs/final_eval* and every run directory.

Reuses the definitions of scripts/final_eval_v2.py (prior correction, T fit, Targets) by importing
that file as a module; nothing in it executes on import.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/revision_A"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("final_eval_v2", ROOT / "scripts/final_eval_v2.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

from artfae_balance import EMOTIONS  # noqa: E402
from artfae_balance.evaluate import evaluate, softmax  # noqa: E402

SEEDS = F.SEEDS
VAL_CSV, TEST_CSV = F.VAL_CSV, F.TEST_CSV
TRAIN_43K = ROOT / "data/artemis_balanced/train.csv"
TRAIN_14K = ROOT / "data/paper14k/train.csv"

# label -> (final_eval_v2 model key, kind)
MODELS = {
    "R0n": ("R0n", "raw"),
    "B/32 R1": ("B32_R1", "raw"),
    "L/14 R1": ("L14_R1", "raw"),
    "L/14 R3": ("L14_R3", "raw"),
    "L/14 R3c": ("L14_R3", "corrected"),
}
EPS = 1e-12


def test_logits_raw(m, s):
    pat = F.MODELS[m][2]
    p = Path(str(pat).format(s=s)) if pat else ROOT / f"runs/final_eval_v2/test/{m}_s{s}_test_logits.npy"
    return np.load(p).astype(np.float64)


class Data:
    """Targets plus per-seed val/test logits (raw or prior-corrected) for every model in MODELS."""

    def __init__(self):
        self.tv, self.tt = F.Targets(VAL_CSV), F.Targets(TEST_CSV)
        self.val_df, self.test_df = pd.read_csv(VAL_CSV), pd.read_csv(TEST_CSV)
        self.vl = F.val_logits_all(self.tv)
        self._cache = {}

    def logits(self, label, s, split="test"):
        m, k = MODELS[label]
        key = (label, s, split)
        if key not in self._cache:
            v = self.vl[(m, s)]
            if split == "val":
                z = v[k]
            else:
                z = test_logits_raw(m, s) + (v["delta"] if k == "corrected" else 0)
            self._cache[key] = z
        return self._cache[key]

    def temperature(self, label, s):
        m, k = MODELS[label]
        return F.fit_t(self.vl[(m, s)][k], self.tv.y)


def dist_metrics(p, y):
    """softCE, KL, Brier per artwork-mean, exactly as artfae_balance.evaluate (eps 1e-12)."""
    ce = -(y * np.log(p + EPS)).sum(1)
    kl = (y * (np.log(y + EPS) - np.log(p + EPS))).sum(1)
    br = ((p - y) ** 2).sum(1)
    return {"softCE": float(ce.mean()), "KL": float(kl.mean()), "Brier": float(br.mean())}


def fmt(v, nd=4):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f}" + (f" ± {v.std(ddof=1):.{nd}f}" if len(v) > 1 else "")


def md_table(df, floatfmt=4):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{x:.{floatfmt}f}" if isinstance(x, (float, np.floating)) else str(x) for x in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
