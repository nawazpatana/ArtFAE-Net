"""Step 1 (crop ablation): recompute the 28 attributes and the describe() text embedding on CLIP L/14@336's own
view of each artwork (shortest side resized to 336 with the processor's bicubic resize, then 336x336 center crop),
instead of the whole image. Train and val only (never test).

Same loader as the original cache (artfae_balance.extract_features.load_image) and the same extract()/describe().
The crop is produced by the exact CLIPImageProcessor of the L/14 snapshot used for features_l14 (rescale and
normalize disabled), rounded back to uint8 RGB. Text: CLIP L/14 get_text_features of the describe() sentence,
L2-normalised; computed once per distinct sentence (the original cache batched sentences, which differs only by
float noise <= 2.4e-7, see revision_A t7_attr_text_check.csv).
global.npy / patches.npy are symlinked from features_l14 (the CLIP image features already use this crop).

usage: python extract_crop_attrs.py --split train|val
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import OUT, ROOT

from artfae.features.aesthetic import describe, extract  # noqa: E402
from artfae_balance.extract_features import load_image  # noqa: E402

MODEL, REV = "openai/clip-vit-large-patch14-336", "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
_proc = None


def crop_view(img):
    global _proc
    if _proc is None:
        from transformers import CLIPImageProcessor
        _proc = CLIPImageProcessor.from_pretrained(MODEL, revision=REV)
    px = _proc(images=img, do_rescale=False, do_normalize=False, return_tensors="np")["pixel_values"][0]
    return Image.fromarray(np.clip(np.rint(px.transpose(1, 2, 0)), 0, 255).astype(np.uint8))


class Rows(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        c = crop_view(load_image(self.paths[i]))
        assert c.size == (336, 336)
        a = extract(c)
        return a, describe(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    a = ap.parse_args()
    df = pd.read_csv(ROOT / f"data/artemis_balanced/{a.split}.csv")
    paths = [p if os.path.isabs(p) else str(ROOT / p) for p in df.image_path]
    out = OUT / "features_l14_crop336" / a.split
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    A, S = [], []
    dl = DataLoader(Rows(paths), batch_size=64, num_workers=14, collate_fn=lambda b: b)
    for i, b in enumerate(dl):
        A += [x[0] for x in b]; S += [x[1] for x in b]
        if i % 50 == 0:
            print(f"{len(A):,}/{len(paths):,} {len(A) / (time.monotonic() - t0):.1f} img/s", flush=True)
    A = np.stack(A).astype(np.float32)

    from transformers import CLIPModel, CLIPTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL, revision=REV).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained(MODEL, revision=REV)
    uniq = sorted(set(S))
    with torch.no_grad():
        emb = {}
        for s in uniq:
            t = model.get_text_features(**tok([s], return_tensors="pt").to(dev))
            emb[s] = (t / t.norm(dim=-1, keepdim=True))[0].float().cpu().numpy()
    T = np.stack([emb[s] for s in S]).astype(np.float32)
    np.save(out / "attributes.npy", A); np.save(out / "attr_text.npy", T)
    for k in ("global", "patches"):
        link = out / f"{k}.npy"
        if not link.exists():
            link.symlink_to(ROOT / f"features_l14/{a.split}/{k}.npy")
    pd.Series(S).to_csv(out / "sentences.csv", index=False, header=["sentence"])
    orig = np.load(ROOT / f"features_l14/{a.split}/attr_text.npy")
    print(f"done {a.split}: {len(A)} rows, {len(uniq)} distinct sentences, "
          f"{np.mean([describe(x) for x in np.load(ROOT / f'features_l14/{a.split}/attributes.npy')] != np.array(S)):.3f} "
          f"of artworks change sentence vs whole image; {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
