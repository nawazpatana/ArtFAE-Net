"""artfae_balance.train for any backbone: identical training, input dims read from the feature files.

train.py hard-codes the B/32 dims (512 / 28 / 512 / 768) and its hash is recorded in
runs/final_selection.txt, so it is not edited. This wrapper swaps only the dims passed to build_model,
then runs train.main unchanged, and records the dims in <out>/dims.json.

usage: python -m artfae_balance.train_backbone --out runs/L14_R1_s42 --seed 42 \
           --train-features features_l14/train --val-features features_l14/val [train.py options]
"""
import json
import sys
from pathlib import Path

import numpy as np

from . import train as T
from .model import build_model


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    feats = Path(arg("--train-features", "features/train"))
    shape = lambda k: np.load(feats / f"{k}.npy", mmap_mode="r").shape
    dims = {"visual_dim": shape("global")[1], "attribute_dim": shape("attributes")[1],
            "text_dim": shape("attr_text")[1], "patch_dim": shape("patches")[2]}
    val = Path(arg("--val-features", "features/val"))
    for k, d in (("global", 1), ("attr_text", 1), ("patches", 2)):
        if np.load(val / f"{k}.npy", mmap_mode="r").shape[d] != np.load(feats / f"{k}.npy", mmap_mode="r").shape[d]:
            raise ValueError(f"train/val feature dims differ for {k}")
    T.build_model = lambda _b32_dims, architecture, **kw: build_model(dims, architecture, **kw)
    T.main()
    (Path(arg("--out", None)) / "dims.json").write_text(json.dumps(dims))
    print(f"dims {dims}")


if __name__ == "__main__":
    main()
