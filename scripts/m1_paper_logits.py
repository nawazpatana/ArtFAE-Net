"""Raw logits of the paper's M1 checkpoint on an artfae_balance feature folder, in manifest row order.

Attributes are normalised with the statistics saved in the checkpoint (fit on the paper's
14K train split), never refit.

usage: python scripts/m1_paper_logits.py --features features/test --out runs/m1_paper/test_logits.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artfae import EMOTIONS  # noqa: E402
from artfae.features.aesthetic import normalize  # noqa: E402
from artfae.models.network import ArtFAENet  # noqa: E402

CHECKPOINT = "/home/dev/ArtFAE-Net/results/artfae_m1/20261027T012203Z_s42_c2c10f53/best_model.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if state["emotions"] != list(EMOTIONS):
        raise ValueError("Class order mismatch")
    cfg = state["config"]
    model = ArtFAENet(**cfg["dimensions"], **cfg["architecture"])
    model.load_state_dict(state["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    f = {k: np.load(a.features / f"{k}.npy", mmap_mode="r") for k in ("global", "attributes", "attr_text", "patches")}
    attrs = normalize(np.asarray(f["attributes"]), state["normalizer"])
    n = len(attrs)
    logits = np.empty((n, len(EMOTIONS)), np.float32)
    with torch.no_grad():
        for s in range(0, n, a.batch):
            e = min(s + a.batch, n)
            g, t = (torch.from_numpy(np.array(f[k][s:e])).to(dev) for k in ("global", "attr_text"))
            p = torch.from_numpy(np.asarray(f["patches"][s:e], dtype=np.float32)).to(dev)
            logits[s:e] = model(g, torch.from_numpy(attrs[s:e]).to(dev), t, p).float().cpu().numpy()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, logits)
    print(f"{n:,} rows -> {a.out}  (normalizer fit_split={state['normalizer']['fit_split']}, "
          f"version={state['normalizer']['version']}, checkpoint epoch={state['epoch']})")


if __name__ == "__main__":
    main()
