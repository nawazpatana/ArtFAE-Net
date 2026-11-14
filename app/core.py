"""Torch-free part of the demo: paths, constants, image loading, display helpers and the app mode.

Imported by inference.py (full mode, with torch/CLIP) and by gallery.py (gallery mode, precomputed results only).
Nothing here may import torch, transformers or CLIP.

Layout: the repository (app/, src/artfae/, artfae_balance/) and the artfae_demo bundle (artfae_demo/app/,
artfae_demo/src/artfae/, artfae_demo/artfae_balance/) have the same shape, so the research modules are found
relative to this file in both.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for _p in (APP_DIR, ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from artfae.features.aesthetic import NAMES  # noqa: E402  (numpy + PIL only)
from artfae_balance import EMOTIONS  # noqa: E402  (a constant; the package __init__ imports nothing)

Image.MAX_IMAGE_PIXELS = None  # as artfae_balance.extract_features: one WikiArt image is ~100 MP

ASSETS = APP_DIR / "assets"
POOL_GRID = 7
SEEDS = [42, 43, 44, 45, 46]
MODES = ("r1", "r3")  # r1 = default
APP_MODES = ("full", "gallery")
SENTIMENT = {"amusement": "positive", "awe": "positive", "contentment": "positive", "excitement": "positive",
             "anger": "negative", "disgust": "negative", "fear": "negative", "sadness": "negative",
             "something_else": "other"}


def app_mode(secrets=None) -> str:
    """APP_MODE from the environment, else from Streamlit secrets (pass st.secrets), else "full"."""
    mode = os.environ.get("APP_MODE")
    if not mode and secrets is not None:
        try:
            mode = secrets.get("APP_MODE")
        except Exception:  # no secrets.toml: Streamlit raises instead of returning None
            mode = None
    mode = (mode or "full").strip().lower()
    if mode not in APP_MODES:
        raise ValueError(f"APP_MODE must be one of {APP_MODES}, not {mode!r}")
    return mode


def load_image(path) -> Image.Image:
    """Identical to artfae_balance.extract_features.load_image (which imports torch, so it is copied here)."""
    with Image.open(path) as im:
        im.draft("RGB", (512, 512))
        return ImageOps.exif_transpose(im).convert("RGB")


def softmax_mean(raw_logits, delta=None):
    """Per-seed softmax of float64 logits (+ optional prior shift), then the seed mean: exactly as predict()."""
    probs = []
    for z in np.asarray(raw_logits, np.float64):
        c = z + delta if delta is not None else z
        e = np.exp(c - c.max()); probs.append(e / e.sum())
    return np.mean(probs, 0), np.array(probs)


def sentiment_totals(probs) -> dict:
    out = {"positive": 0.0, "negative": 0.0, "other": 0.0}
    for e, p in zip(EMOTIONS, probs):
        out[SENTIMENT[e]] += float(p)
    return out


def percentiles(attrs, table: dict) -> np.ndarray:
    """Mid-rank percentile (0-100) of each raw attribute within the training distribution."""
    out = []
    for x, q in zip(attrs, table["quantiles"]):
        q = np.asarray(q)  # 1,001 evenly spaced training quantiles
        lo, hi = np.searchsorted(q, x, "left"), np.searchsorted(q, x, "right")
        out.append(100 * (lo + hi) / (2 * len(q)))  # ties (e.g. many zeros) get their mid-rank
    return np.asarray(out)


def load_json(name, assets=ASSETS):
    return json.loads((Path(assets) / name).read_text(encoding="utf-8"))


def load_examples(assets=ASSETS):
    ex = load_json("examples/examples.json", assets)
    for e in ex:
        e["path"] = str(Path(assets) / "examples" / e["file"])
    return ex


def center_square(img: Image.Image) -> Image.Image:
    """The region CLIP sees: shortest side resized to 336, then a centre crop."""
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def colormap(x):
    """x in [0, 1] -> RGB, a perceptually ordered dark-violet -> amber ramp."""
    stops = np.array([[20, 12, 40], [87, 31, 104], [173, 55, 92], [236, 120, 58], [252, 220, 140]], float)
    pos = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(x, pos, stops[:, i]) for i in range(3)], -1)


def attention_overlay(img: Image.Image, map7, alpha: float, size: int = 640) -> Image.Image:
    """Centre square of img with the 7x7 map upsampled (bicubic) and blended at opacity alpha."""
    base = center_square(img).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    m = np.asarray(map7, np.float64)
    m = (m - m.min()) / max(m.max() - m.min(), 1e-12)
    up = Image.fromarray((m * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC)
    heat = colormap(np.asarray(up, np.float64) / 255)
    out = (1 - alpha) * np.asarray(base, np.float64) + alpha * heat
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
