"""Cache frozen CLIP features for a manifest (Step 2).

Writes, in manifest row order:
  global.npy      (N, 512)      L2-normalised projected CLIP image embedding
  patches.npy     (N, 49, 768)  float16 patch tokens (CLS excluded)
  attributes.npy  (N, 28)       raw numerical attributes (normalise with TRAIN stats later)
  attr_text.npy   (N, 512)      L2-normalised CLIP embedding of the deterministic description

Usage:
  python -m artfae_balance.extract_features --manifest data/artemis_balanced/train.csv \
      --out features/train --batch 128

Disk: ~7.4 GB of patches per 100K artworks (float16).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

Image.MAX_IMAGE_PIXELS = None  # one WikiArt image is ~100 MP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artfae.features.aesthetic import extract, describe  # noqa: E402  (rgb256-v1, paper code)

# CLIP snapshot used for the paper's clean_dataset_v2 cache
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def load_image(path) -> Image.Image:
    """Same as artfae.datasets.manifests.open_image, minus its 40-MP guard."""
    with Image.open(path) as im:
        im.draft("RGB", (512, 512))
        return ImageOps.exif_transpose(im).convert("RGB")


# --------------------------------------------------------------------------------------
# HOOK: ArtFAE-Net attribute code, so features match the paper.
# Returns (28 raw floats in artfae.features.aesthetic.NAMES order, description string).
# --------------------------------------------------------------------------------------
def compute_attributes_and_text(img: Image.Image):
    vec = extract(img)
    return vec.astype(np.float32), describe(vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--revision", default=CLIP_REVISION)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.manifest)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # fp32 compute and the slow image processor, as in the paper's cache
    model = CLIPModel.from_pretrained(a.model, revision=a.revision).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(a.model, revision=a.revision, use_fast=False)

    n = len(df)
    G = np.lib.format.open_memmap(out / "global.npy", "w+", np.float32, (n, 512))
    P = np.lib.format.open_memmap(out / "patches.npy", "w+", np.float16, (n, 49, 768))
    A = np.lib.format.open_memmap(out / "attributes.npy", "w+", np.float32, (n, 28))
    T = np.lib.format.open_memmap(out / "attr_text.npy", "w+", np.float32, (n, 512))

    paths = df["image_path"].tolist()
    with torch.no_grad():
        for s in range(0, n, a.batch):
            imgs = [load_image(p) for p in paths[s:s + a.batch]]
            attrs, texts = zip(*(compute_attributes_and_text(im) for im in imgs))
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev, model.dtype)
            vo = model.vision_model(pixel_values=px)
            g = model.visual_projection(vo.pooler_output)
            g = g / g.norm(dim=-1, keepdim=True)
            tk = proc(text=list(texts), padding=True, truncation=True, return_tensors="pt").to(dev)
            t = model.get_text_features(**tk)
            t = t / t.norm(dim=-1, keepdim=True)
            e = s + len(imgs)
            G[s:e] = g.float().cpu().numpy()
            P[s:e] = vo.last_hidden_state[:, 1:, :].half().cpu().numpy()
            A[s:e] = np.stack(attrs)
            T[s:e] = t.float().cpu().numpy()
            if (s // a.batch) % 20 == 0:
                print(f"{e:,}/{n:,}", flush=True)
    for m in (G, P, A, T):
        m.flush()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
