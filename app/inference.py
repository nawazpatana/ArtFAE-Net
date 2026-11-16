"""ArtFAE-Net inference for the demo: two L/14 seed ensembles, both selected by min val soft CE.

  mode "r1"  Emotion distribution (default): L/14 R1, runs/revision_B/step1/R1_s42-46 (paper's recommended
             model). Optional prior correction logits + log(val_prior) - log(train_prior), off by default.
  mode "r3"  Rare-emotion sensitivity: L/14 R3 (balanced sampler + presence head, lambda 0.3),
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

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
# core puts the repository / bundle root and its src/ on sys.path, relative to this file
from core import (ASSETS, EMOTIONS, MODES, NAMES, POOL_GRID, ROOT, SEEDS, SENTIMENT,  # noqa: E402,F401
                  attention_overlay, center_square, colormap, load_examples, load_json, percentiles,
                  sentiment_totals)
from artfae.features.aesthetic import normalize  # noqa: E402
from artfae_balance.extract_features import compute_attributes_and_text, load_image  # noqa: E402
from artfae_balance.model import build_model  # noqa: E402
from artfae.models.network import ArtFAENet  # noqa: E402

CLIP_MODEL = "openai/clip-vit-large-patch14-336"
CLIP_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

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
    text_table: dict = None  # vision-only engine: describe() sentence -> L2-normalised CLIP text embedding (768,)


def pick_device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_text_table(assets=ASSETS) -> dict:
    """app/assets/text_embeddings.npz (app/build_text_embeddings.py): all 27 describe() sentences."""
    z = np.load(Path(assets) / "text_embeddings.npz")
    assert str(z["clip"]) == f"{CLIP_MODEL}@{CLIP_REVISION}", "text embeddings from another CLIP"
    return {str(s): e for s, e in zip(z["sentences"], z["embeddings"])}


def load_engine(assets=ASSETS, device=None, vision_only=False) -> Engine:
    """Full engine (CLIP image + text towers), or vision_only=True for the memory-limited cloud_live mode:
    CLIPVisionModelWithProjection (the same image weights and projection, without the text encoder) and the
    precomputed text embeddings of the 27 possible descriptions."""
    from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPVisionModelWithProjection
    dev = pick_device(device)
    # The pinned revision ships only pytorch_model.bin. Load exactly that file: use_safetensors=False, and
    # stop transformers' background "auto_conversion" thread, which would otherwise download a 1.7 GB
    # safetensors copy from a different (unpinned) revision on every online load of a .bin checkpoint.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")
    if vision_only:
        clip = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION,
                                                             use_safetensors=False).to(dev).eval()  # fp32
        proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION, use_fast=False)
        table = load_text_table(assets)
    else:
        clip = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION, use_safetensors=False).to(dev).eval()  # fp32
        proc = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION, use_fast=False)
        table = None
    models = {mode: [(s, *load_model(Path(assets) / f"models/{mode}/s{s}", mode, dev)) for s in SEEDS]
              for mode in MODES}
    pri = json.loads((Path(assets) / "priors.json").read_text())
    assert pri["emotions"] == EMOTIONS
    pres = json.loads((Path(assets) / "presence.json").read_text())
    assert pres["emotions"] == EMOTIONS
    offset = torch.tensor(pres["offset"], dtype=torch.float32, device=dev)
    return Engine(clip, proc, models, np.asarray(pri["delta"], np.float64), dev, offset, table)


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
    if engine.text_table is None:  # full CLIP
        vo = m.vision_model(pixel_values=px)
        g = m.visual_projection(vo.pooler_output)
        tk = proc(text=[text], padding=True, truncation=True, return_tensors="pt").to(dev)
        t = m.get_text_features(**tk); t = t / t.norm(dim=-1, keepdim=True)
    else:  # vision tower only: image_embeds = visual_projection(pooler_output), the same two steps
        vo = m(pixel_values=px)
        g = vo.image_embeds
        t = torch.from_numpy(engine.text_table[text][None]).to(dev)  # precomputed, already normalised
    g = g / g.norm(dim=-1, keepdim=True)
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
