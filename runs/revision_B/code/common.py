"""Shared helpers for revision Phase B. Imports scripts/final_eval_v2.py as a module (nothing runs on import)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/revision_B"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("final_eval_v2", ROOT / "scripts/final_eval_v2.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

from artfae_balance import EMOTIONS  # noqa: E402
from artfae_balance.evaluate import evaluate, load_targets, softmax  # noqa: E402

EPS = 1e-12
SIDX = F.SIDX


class Sub:
    """A row subset of final_eval_v2.Targets with the attributes metrics()/boot_stats() use."""

    def __init__(self, tg, mask):
        for k in ("y", "present", "uniq", "su", "ys"):
            setattr(self, k, getattr(tg, k)[mask])


def fmt(v, nd=4):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f}" + (f" ± {v.std(ddof=1):.{nd}f}" if len(v) > 1 else "")
