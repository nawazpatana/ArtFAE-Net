"""Shared helpers for the app diagnosis. Read-only on every checkpoint, manifest and feature file.

Uses VALIDATION features / images only. Never opens holdout or test files.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/app_diagnosis"
for p in (ROOT, ROOT / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from artfae.features.aesthetic import NAMES, normalize  # noqa: E402
from artfae.models.network import ArtFAENet  # noqa: E402
from artfae_balance import EMOTIONS  # noqa: E402
from artfae_balance.model import build_model  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
EPS = 1e-12
VAL_CSV = ROOT / "data/artemis_balanced/val.csv"
TRAIN_CSV = ROOT / "data/artemis_balanced/train.csv"
VAL_FEAT = ROOT / "features_l14/val"
SEL = json.loads((ROOT / "runs/final_selection_v3.txt").read_text())
PRI = json.loads((ROOT / "app/assets/priors.json").read_text())
DELTA = np.asarray(PRI["delta"], np.float64)

# model sets a-e (a and b share checkpoints; a adds the prior correction)
MODELS = {
    "a_app": {"label": "App as deployed: R3_MF1 ensemble + prior correction", "key": "R3_MF1", "correct": True},
    "b_mf1_raw": {"label": "R3_MF1 ensemble, no correction", "key": "R3_MF1", "correct": False},
    "c_r3_ce": {"label": "R3 softCE-selected, no correction", "key": "R3", "correct": False},
    "d_r1": {"label": "L/14 R1 softCE-selected, no correction", "key": "R1", "correct": False},
    "e_probe": {"label": "Linear probe (lr 1e-3), no correction", "key": "PROBE", "correct": False},
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Probe(nn.Module):  # == runs/revision_B/code/common.Probe
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d["visual_dim"], 9)

    def forward(self, g, a, t, patches):
        return self.head(g)


def load_run(run, dev):
    """Same branches as runs/revision_B/code/eval_holdout.load_run (R1 / R3 / probe / existing runs)."""
    ck = torch.load(ROOT / run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if "condition" in cfg:
        c, d, arch = cfg["condition"], cfg["dims"], cfg["architecture"]
        if c == "R3":
            model = build_model(d, arch, presence=True)
        elif c == "R1":
            model = ArtFAENet(**d, **arch)
        elif c.startswith("probe100"):
            model = Probe(d)
        else:
            raise ValueError(c)
    else:
        d = json.loads((ROOT / run / "dims.json").read_text())
        model = build_model(d, cfg["architecture"], presence=cfg["lambda_present"] > 0)
    model.load_state_dict(ck["model"], strict=True)
    assert ck["emotions"] == EMOTIONS, run
    return model.to(dev).eval(), ck["normalizer"], ck


def load_set(key, dev):
    return [(run, *load_run(run, dev)[:2]) for run in SEL["models"][key]["runs"]]


@torch.no_grad()
def forward_set(mset, g, a_raw, t, p, dev, bs=256):
    """raw logits (seeds, N, 9) and presence sigmoid (seeds, N, 9) or None."""
    L, P = [], []
    for run, model, norm in mset:
        a = normalize(a_raw, norm)
        ls, ps = [], []
        for i in range(0, len(g), bs):
            sl = slice(i, i + bs)
            args = [torch.as_tensor(x[sl]).to(dev).float() for x in (g, a, t, p)]
            if hasattr(model, "presence_head") and model.presence_head is not None:
                lo, pr = model(*args, return_presence=True)
                ps.append(torch.sigmoid(pr).double().cpu().numpy())
            else:
                lo = model(*args)
            ls.append(lo.double().cpu().numpy())
        L.append(np.concatenate(ls))
        P.append(np.concatenate(ps) if ps else None)
    return np.stack(L), (np.stack(P) if P[0] is not None else None)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ensemble(raw, correct):
    """App rule: per seed (logits [+ delta]) -> softmax, then average probabilities over seeds."""
    return softmax(raw + (DELTA if correct else 0.0)).mean(0)


def soft_ce(y, p):
    return -(y * np.log(p + EPS)).sum(-1)


def val_df():
    return pd.read_csv(VAL_CSV)


def val_features(rows=None):
    f = {k: np.load(VAL_FEAT / f"{k}.npy", mmap_mode="r") for k in ("global", "attributes", "attr_text", "patches")}
    if rows is None:
        rows = np.arange(len(f["global"]))
    rows = np.asarray(rows)
    return (np.asarray(f["global"][rows], np.float32), np.asarray(f["attributes"][rows], np.float32),
            np.asarray(f["attr_text"][rows], np.float32), np.asarray(f["patches"][rows]).astype(np.float32))
