"""Torch-free part of the demo: paths, constants, image loading, display helpers and the app mode.

Imported by inference.py (full mode, with torch/CLIP) and by gallery.py (gallery mode, precomputed results only).
Nothing here may import torch, transformers or CLIP.

Layout: the repository (app/, src/artfae/, artfae_balance/) and the artfae_demo bundle (artfae_demo/app/,
artfae_demo/src/artfae/, artfae_demo/artfae_balance/) have the same shape, so the research modules are found
relative to this file in both.
"""
import io
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
APP_MODES = ("full", "gallery", "cloud_live")
# full        live inference for everything (local demo, GPU if available)
# gallery     precomputed examples only, no torch, no uploads
# cloud_live  precomputed examples + live CPU inference for uploads (CLIP vision tower only; Streamlit Cloud)
UPLOAD_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
UPLOAD_MAX_PIXELS = 50_000_000        # decoded size guard (a 10 MB PNG can expand to gigabytes)
DRAFT = 512                           # as artfae_balance.extract_features.load_image: im.draft("RGB", (512, 512))

APP_VERSION = "1.3.0"                 # shown beside every result and written into the JSON record
# The result card reports a dispersed distribution instead of a single top emotion when the two highest predicted
# proportions are within this many percentage points (0.05 = 5 points), compared on the unrounded values.
DISPERSED_MAX_GAP = 0.05
MODEL_INFO = {  # what each demo model is, for the JSON record
    "r1": {"name": "Emotion distribution (L/14 R1)", "checkpoints": "runs/revision_B/step1/R1_s{42..46} (softCE-selected)"},
    "r3": {"name": "Rare-emotion sensitivity (L/14 R3)", "checkpoints": "runs/revision_B/step1/R3_s{42..46} (softCE-selected)"},
}
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


class UploadTooLarge(ValueError):
    """More than UPLOAD_MAX_BYTES, or more than UPLOAD_MAX_PIXELS once decoded."""


def heic_supported() -> bool:
    """Register pillow-heif (iPhone HEIC/HEIF) with Pillow if it is installed."""
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return False
    register_heif_opener()
    return True


def upload_types() -> list:
    return ["jpg", "jpeg", "png", "webp"] + (["heic", "heif"] if heic_supported() else [])


def load_upload(data: bytes) -> Image.Image:
    """Decode an uploaded image in memory (nothing is written to disk) and downscale it before inference.

    JPEG goes through load_image unchanged (its draft mode decodes at the largest 1/2, 1/4 or 1/8 scale that
    keeps both sides >= 512 px, exactly as in the training pipeline). Other formats (PNG, WebP, HEIC) are
    reduced by the same power-of-two rule after decoding, so a large upload is analysed at the size a JPEG of
    the same dimensions would be.
    """
    if len(data) > UPLOAD_MAX_BYTES:
        raise UploadTooLarge(f"{len(data)} bytes")
    heic_supported()
    with Image.open(io.BytesIO(data)) as probe:  # header only: check the decoded size before decoding
        if probe.size[0] * probe.size[1] > UPLOAD_MAX_PIXELS:
            raise UploadTooLarge(f"{probe.size[0]}x{probe.size[1]} px")
        is_jpeg = probe.format == "JPEG"
    img = load_image(io.BytesIO(data))
    if not is_jpeg:
        w, h = img.size
        f = next((f for f in (8, 4, 2) if w // f >= DRAFT and h // f >= DRAFT), 1)
        if f > 1:
            img = img.reduce(f)
    return img


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


# ---------------------------------------------------------------- result wording and reproducibility
def top_two(probs):
    """-> ((emotion, p), (emotion, p)) for the two highest predicted proportions."""
    p = np.asarray(probs, np.float64)
    a, b = np.argsort(-p, kind="stable")[:2]
    return (EMOTIONS[a], float(p[a])), (EMOTIONS[b], float(p[b]))


def is_dispersed(probs, max_gap: float = DISPERSED_MAX_GAP) -> bool:
    """True when the top two predicted proportions are within max_gap (5 percentage points by default)."""
    (_, pa), (_, pb) = top_two(probs)
    return pa - pb <= max_gap + 1e-12


def result_record(r: dict, *, app_mode: str, image_sha256: str | None = None, artwork_id: str | None = None,
                  timestamp: str | None = None, palette: list | None = None) -> dict:
    """The downloadable JSON record of one result. Identifies the input by its SHA-256 (uploads) or its ArtEmis
    artwork ID (gallery examples); the image itself is never included. palette: visual.palette() of the image."""
    from datetime import datetime, timezone
    if (image_sha256 is None) == (artwork_id is None):
        raise ValueError("give exactly one of image_sha256 or artwork_id")
    probs = np.asarray(r["probs"], np.float64)
    (a, pa), (b, pb) = top_two(probs)
    rec = {
        "app": "ArtFAE-Net demo", "app_version": APP_VERSION, "app_mode": app_mode,
        "timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": ({"type": "gallery_example", "artwork_id": artwork_id} if artwork_id else
                  {"type": "upload", "sha256": image_sha256}),
        "model": {"key": r["mode"], **MODEL_INFO[r["mode"]], "seeds": SEEDS,
                  "ensemble": "mean of the five seeds' softmax probabilities"},
        "calibration": {"prior_correction": bool(r["prior_correction"]),
                        "description": "logits + log(validation prior) - log(training prior)" if r["prior_correction"]
                        else "none (raw model outputs)"},
        "source": "precomputed" if r.get("precomputed") else "live inference",
        "emotion_distribution": {e: round(float(p), 6) for e, p in zip(EMOTIONS, probs)},
        "highest": {"emotion": a, "proportion": round(pa, 6)},
        "dispersed": {"is_dispersed": is_dispersed(probs), "threshold_points": DISPERSED_MAX_GAP * 100,
                      "top_two": [{"emotion": a, "proportion": round(pa, 6)}, {"emotion": b, "proportion": round(pb, 6)}]},
        "sentiment_totals": {k: round(v, 6) for k, v in sentiment_totals(probs).items()},
    }
    if palette is not None:
        rec["palette"] = {"method": "k-means (k = 6) in CIELAB on a copy downscaled to 128 px, k-means++ init, seed 0",
                          "colours": [{"hex": c["hex"], "share": round(float(c["share"]), 6), "percent": c["percent"]}
                                      for c in palette]}
    if r.get("presence") is not None:
        rec["presence_r3"] = {"meaning": "estimated chance that at least one annotator names the emotion; a separate "
                                         "output, not part of the distribution",
                              "values": {e: round(float(p), 6) for e, p in zip(EMOTIONS, r["presence"])}}
    return rec
