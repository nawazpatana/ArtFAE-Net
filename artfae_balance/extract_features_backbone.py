"""extract_features.py for any CLIP backbone: same load_image and attribute HOOK, dims from the model config.

Writes global (N, proj) float32, patches (N, tokens, hidden) float16, attributes (N, 28), attr_text (N, proj).
--pool-grid k average-pools the patch-token grid to k x k (fp32, before the float16 cast).
extract_features.py itself is left unchanged (its hash is recorded in runs/final_selection.txt).

usage: python -m artfae_balance.extract_features_backbone --manifest ... --out ... \
           --model openai/clip-vit-large-patch14-336 --revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from .extract_features import compute_attributes_and_text, load_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (speed test)")
    ap.add_argument("--pool-grid", type=int, default=None, help="average-pool patch tokens to k x k")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.manifest)
    if a.limit:
        df = df.head(a.limit)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = CLIPModel.from_pretrained(a.model, revision=a.revision).to(dev).eval()  # fp32
    proc = CLIPProcessor.from_pretrained(a.model, revision=a.revision, use_fast=False)
    vc = model.config.vision_config
    side = vc.image_size // vc.patch_size
    tokens = (a.pool_grid or side) ** 2
    proj = model.config.projection_dim
    print(f"model={a.model}@{a.revision[:8]} image={vc.image_size} patch={vc.patch_size} "
          f"grid={side}x{side} -> {a.pool_grid or side}x{a.pool_grid or side} "
          f"global=({proj},) patches=({tokens},{vc.hidden_size}) text=({proj},)", flush=True)

    n = len(df)
    G = np.lib.format.open_memmap(out / "global.npy", "w+", np.float32, (n, proj))
    P = np.lib.format.open_memmap(out / "patches.npy", "w+", np.float16, (n, tokens, vc.hidden_size))
    A = np.lib.format.open_memmap(out / "attributes.npy", "w+", np.float32, (n, 28))
    T = np.lib.format.open_memmap(out / "attr_text.npy", "w+", np.float32, (n, proj))
    paths = df["image_path"].tolist()
    t0 = time.monotonic()
    with torch.no_grad():
        for s in range(0, n, a.batch):
            imgs = [load_image(p) for p in paths[s:s + a.batch]]
            attrs, texts = zip(*(compute_attributes_and_text(im) for im in imgs))
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
            vo = model.vision_model(pixel_values=px)
            g = model.visual_projection(vo.pooler_output); g = g / g.norm(dim=-1, keepdim=True)
            tk = proc(text=list(texts), padding=True, truncation=True, return_tensors="pt").to(dev)
            t = model.get_text_features(**tk); t = t / t.norm(dim=-1, keepdim=True)
            e = s + len(imgs)
            patches = vo.last_hidden_state[:, 1:, :]
            if a.pool_grid:
                b, _, h = patches.shape
                grid = patches.transpose(1, 2).reshape(b, h, side, side)          # row-major token grid
                patches = F.adaptive_avg_pool2d(grid, a.pool_grid).flatten(2).transpose(1, 2)
            G[s:e] = g.cpu().numpy(); P[s:e] = patches.half().cpu().numpy()
            A[s:e] = np.stack(attrs); T[s:e] = t.cpu().numpy()
            if (s // a.batch) % 20 == 0:
                print(f"{e:,}/{n:,}  {e / (time.monotonic() - t0):.1f} img/s", flush=True)
    for m in (G, P, A, T):
        m.flush()
    print(f"done -> {out}  {n / (time.monotonic() - t0):.1f} img/s", flush=True)


if __name__ == "__main__":
    main()
