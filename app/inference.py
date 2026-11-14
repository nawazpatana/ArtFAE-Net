"""ArtFAE-Net inference for the demo: two L/14 seed ensembles, both selected by min val soft CE.

  mode "r1"  Human-vote distribution (default): L/14 R1, runs/revision_B/step1/R1_s42-46 (paper's recommended
             model). Optional prior correction logits + log(val_prior) - log(train_prior), off by default.
  mode "r3"  Minority-sensitive: L/14 R3 (balanced sampler + presence head, lambda 0.3),
             runs/revision_B/step1/R3_s42-46. Raw logits only; the prior correction is not available here.

Pure functions, no Streamlit. Per image, exactly as features_l14:
  1. artfae_balance.extract_features.load_image  (JPEG draft 512 px, exif_transpose, MAX_IMAGE_PIXELS=None)
  2. its HOOK: artfae.features.aesthetic extract() -> 28 attributes, describe() -> description
  3. CLIP ViT-L/14@336 (pinned revision, fp32): L2-normalised projected global embedding, patch tokens
     average-pooled 24x24 -> 7x7 and stored as float16 (as in the feature cache), L2-normalised CLIP
     text embedding of the description
  4. each seed: attributes normalised with that run's normalizer.json, forward -> class logits,
     ECSA attention maps (+ presence logits for r3)
  5. [r1 with prior correction only: + delta], softmax; probabilities averaged over the 5 seeds
  6. r3 presence: sigmoid(presence_logits - log(pos_weight)), averaged over the 5 seeds. The presence head was
     trained with BCE pos_weight (presence.json), which shifts its optimal logit by +log(pos_weight); undoing
     it makes the chips the probability that at least one viewer names the emotion, as the UI states.
"""
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from artfae.features.aesthetic import NAMES, normalize  # noqa: E402
from artfae_balance import EMOTIONS  # noqa: E402
from artfae_balance.extract_features import compute_attributes_and_text, load_image  # noqa: E402
from artfae_balance.model import build_model  # noqa: E402
from artfae.models.network import ArtFAENet  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"
CLIP_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
POOL_GRID = 7
SEEDS = [42, 43, 44, 45, 46]
MODES = ("r1", "r3")  # r1 = default
SENTIMENT = {"amusement": "positive", "awe": "positive", "contentment": "positive", "excitement": "positive",
             "anger": "negative", "disgust": "negative", "fear": "negative", "sadness": "negative",
             "something_else": "other"}

__all__ = ["EMOTIONS", "NAMES", "SENTIMENT", "MODES", "Engine", "load_engine", "load_image", "predict", "predict_path",
           "sentiment_totals", "percentiles", "load_examples", "attention_overlay", "center_square"]


@dataclass
class Engine:
    clip: object
    processor: object
    models: dict            # mode -> [(seed, model, normalizer dict)]
    delta: np.ndarray       # log(val_prior) - log(train_prior), (9,)
    device: torch.device
    presence_offset: torch.Tensor = None  # log(pos_weight) of the training PresenceBCE, (9,)


def pick_device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_engine(assets=ASSETS, device=None) -> Engine:
    from transformers import CLIPModel, CLIPProcessor
    dev = pick_device(device)
    # The pinned revision ships only pytorch_model.bin. Load exactly that file: use_safetensors=False, and
    # stop transformers' background "auto_conversion" thread, which would otherwise download a 1.7 GB
    # safetensors copy from a different (unpinned) revision on every online load of a .bin checkpoint.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")
    clip = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION, use_safetensors=False).to(dev).eval()  # fp32
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION, use_fast=False)
    models = {mode: [(s, *load_model(Path(assets) / f"models/{mode}/s{s}", mode, dev)) for s in SEEDS]
              for mode in MODES}
    pri = json.loads((Path(assets) / "priors.json").read_text())
    assert pri["emotions"] == EMOTIONS
    pres = json.loads((Path(assets) / "presence.json").read_text())
    assert pres["emotions"] == EMOTIONS
    offset = torch.tensor(pres["offset"], dtype=torch.float32, device=dev)
    return Engine(clip, proc, models, np.asarray(pri["delta"], np.float64), dev, offset)


def load_model(d: Path, mode: str, dev):
    """One seed, built as runs/revision_B/code/common.build: R1 = paper ArtFAENet, R3 = + PresenceHead."""
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    cfg, dims = ck["config"], json.loads((d / "dims.json").read_text())
    assert ck["emotions"] == EMOTIONS and cfg["condition"] == {"r1": "R1", "r3": "R3"}[mode], d
    model = ArtFAENet(**dims, **cfg["architecture"]) if mode == "r1" else \
        build_model(dims, cfg["architecture"], presence=True)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(dev).eval(), json.loads((d / "normalizer.json").read_text())


@torch.no_grad()
def encode(engine: Engine, img: Image.Image):
    """CLIP + attribute features, matching artfae_balance.extract_features_backbone --pool-grid 7."""
    attrs, text = compute_attributes_and_text(img)
    m, proc, dev = engine.clip, engine.processor, engine.device
    px = proc(images=[img], return_tensors="pt")["pixel_values"].to(dev)
    vo = m.vision_model(pixel_values=px)
    g = m.visual_projection(vo.pooler_output); g = g / g.norm(dim=-1, keepdim=True)
    tk = proc(text=[text], padding=True, truncation=True, return_tensors="pt").to(dev)
    t = m.get_text_features(**tk); t = t / t.norm(dim=-1, keepdim=True)
    patches = vo.last_hidden_state[:, 1:, :]
    b, n, h = patches.shape
    side = int(round(n ** .5))
    grid = patches.transpose(1, 2).reshape(b, h, side, side)
    patches = F.adaptive_avg_pool2d(grid, POOL_GRID).flatten(2).transpose(1, 2)
    return {"g": g.float(), "t": t.float(), "p": patches.half().float(), "attrs": attrs, "description": text}


@torch.no_grad()
def predict(engine: Engine, img: Image.Image, mode: str = "r1", prior_correction: bool = False) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if prior_correction and mode != "r1":
        raise ValueError("the prior correction is only available for the R1 model")
    t0 = time.perf_counter()
    f = encode(engine, img)
    raw, probs, presence, maps = [], [], [], []
    for _, model, norm in engine.models[mode]:
        a = torch.from_numpy(normalize(f["attrs"][None], norm)).to(engine.device)
        if mode == "r3":
            logits, att, pres = model(f["g"], a, f["t"], f["p"], return_attention=True, return_presence=True)
            presence.append(torch.sigmoid(pres[0] - engine.presence_offset).cpu().numpy())
        else:
            logits, att = model(f["g"], a, f["t"], f["p"], return_attention=True)
        z = logits[0].double().cpu().numpy()
        raw.append(z)
        c = z + engine.delta if prior_correction else z
        e = np.exp(c - c.max()); probs.append(e / e.sum())
        maps.append(att[0].cpu().numpy())
    if engine.device.type == "cuda":
        torch.cuda.synchronize()
    p = np.mean(probs, 0)
    return {"probs": p, "probs_per_seed": np.array(probs), "raw_logits": np.array(raw),
            "presence": np.mean(presence, 0) if presence else None,
            "maps": np.mean(maps, 0).reshape(9, POOL_GRID, POOL_GRID),
            "attrs": f["attrs"], "description": f["description"], "top": EMOTIONS[int(p.argmax())],
            "mode": mode, "prior_correction": bool(prior_correction),
            "seconds": time.perf_counter() - t0, "device": engine.device.type}


def predict_path(engine: Engine, path_or_file, **kw) -> dict:
    return predict(engine, load_image(path_or_file), **kw)


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
