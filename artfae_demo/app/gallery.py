"""Gallery mode: the 12 example artworks with results precomputed by the full pipeline (app/precompute_gallery.py).

No torch, transformers or CLIP. result() returns the same dict as inference.predict(), so the app renders
both modes with the same code:
  probs / probs_per_seed  recomputed from the stored float64 seed logits exactly as predict() does (softmax per
                          seed, optional R1 prior correction, mean over the 5 seeds)
  maps, presence, attrs   stored as computed (float32 values written with their shortest exact repr)
  description             stored text
  palette, aesthetic      dominant colours and descriptive aesthetic properties (app/precompute_visual.py,
                          torch-free; identical to visual.palette / visual.aesthetic_measures on the image)
Attention overlays are drawn at view time by core.attention_overlay from the stored 7x7 maps, which is the
same function full mode uses, so the opacity slider keeps working.
"""
import hashlib
from functools import lru_cache

import numpy as np

from core import ASSETS, EMOTIONS, MODES, POOL_GRID, SEEDS, load_json, softmax_mean

FILE = "precomputed/gallery.json"
FORMAT = "artfae-gallery/2"  # /2: + palette and aesthetic (app/precompute_visual.py)


class NotInGallery(KeyError):
    """The image is not one of the precomputed examples (gallery mode cannot analyse new images)."""


@lru_cache(maxsize=1)
def _data(assets=str(ASSETS)):
    d = load_json(FILE, assets)
    assert d["format"] == FORMAT and d["emotions"] == EMOTIONS and d["seeds"] == SEEDS, "unexpected gallery file"
    by_sha = {ex["sha256"]: ex for ex in d["examples"].values()}
    return d, by_sha


@lru_cache(maxsize=1)
def _delta(assets=str(ASSETS)):
    pri = load_json("priors.json", assets)
    assert pri["emotions"] == EMOTIONS
    return np.asarray(pri["delta"], np.float64)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def has(data: bytes, assets=ASSETS) -> bool:
    """True if data is one of the precomputed examples (byte for byte)."""
    return sha256(data) in _data(str(assets))[1]


def available(assets=ASSETS) -> bool:
    return (assets / FILE).exists()


def result(data: bytes, mode: str = "r1", prior_correction: bool = False, assets=ASSETS) -> dict:
    """Precomputed equivalent of inference.predict(engine, load_image(data), mode, prior_correction)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if prior_correction and mode != "r1":
        raise ValueError("the prior correction is only available for the R1 model")
    d, by_sha = _data(str(assets))
    ex = by_sha.get(sha256(data))
    if ex is None:
        raise NotInGallery("only the example gallery is available in this version")
    m = ex[mode]
    raw = np.asarray(m["raw_logits"], np.float64)
    p, per_seed = softmax_mean(raw, _delta(str(assets)) if prior_correction else None)
    return {"probs": p, "probs_per_seed": per_seed, "raw_logits": raw,
            "presence": None if m["presence"] is None else np.asarray(m["presence"], np.float32),
            "maps": np.asarray(m["maps"], np.float32).reshape(9, POOL_GRID, POOL_GRID),
            "attrs": np.asarray(ex["attrs"], np.float32), "description": ex["description"],
            "palette": ex["palette"], "aesthetic": ex["aesthetic"],
            "top": EMOTIONS[int(p.argmax())], "mode": mode, "prior_correction": bool(prior_correction),
            "seconds": float(m["seconds"]), "device": d["device"], "precomputed": True}
