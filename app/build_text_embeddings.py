"""Precompute the CLIP text embeddings of every sentence describe() can produce -> app/assets/text_embeddings.npz.

describe() (src/artfae/features/aesthetic.py) has 3 x 3 x 3 = 27 possible outputs (brightness, contrast, colour
temperature). The cloud_live engine loads only CLIP's vision tower and looks the attribute-text embedding up
here instead of running the text encoder. Each embedding is computed exactly as inference.encode() does in full
mode: CLIPProcessor tokenisation (padding, truncation), get_text_features, L2 normalisation, float32; on CPU.

run (full environment, CLIP in the HF cache):  HF_HUB_OFFLINE=1 python app/build_text_embeddings.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

OUT = core.ASSETS / "text_embeddings.npz"
BRIGHTNESS, CONTRAST = ("Low", "Moderate", "High"), ("low", "moderate", "high")
COLOURS = ("predominantly warm", "predominantly cool", "mixed or neutral")


def sentences():
    """Every describe() output; the template is checked against describe() itself."""
    from artfae.features.aesthetic import NAMES, describe
    out = [f"{b} overall brightness, {c} luminance contrast, and {w} colors."
           for b, c, w in itertools.product(BRIGHTNESS, CONTRAST, COLOURS)]
    i = {n: k for k, n in enumerate(NAMES)}
    seen = set()
    for lum, std, (warm, cool) in itertools.product([.1, .5, .9], [.05, .2, .4], [(.6, .1), (.1, .6), (.3, .3)]):
        a = np.zeros(len(NAMES), np.float32)
        a[i["luminance"]], a[i["luminance_std"]], a[i["warm"]], a[i["cool"]] = lum, std, warm, cool
        seen.add(describe(a))
    assert seen == set(out) and len(out) == 27, "describe() changed: update the template here"
    return out


def main():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    import inference as inf
    sents = sentences()
    m = CLIPModel.from_pretrained(inf.CLIP_MODEL, revision=inf.CLIP_REVISION, use_safetensors=False).eval()
    proc = CLIPProcessor.from_pretrained(inf.CLIP_MODEL, revision=inf.CLIP_REVISION, use_fast=False)
    emb = []
    with torch.no_grad():
        for s in sents:  # one at a time, exactly as encode() tokenises a single description
            tk = proc(text=[s], padding=True, truncation=True, return_tensors="pt")
            t = m.get_text_features(**tk); t = t / t.norm(dim=-1, keepdim=True)
            emb.append(t.float()[0].numpy())
    np.savez(OUT, sentences=np.array(sents), embeddings=np.stack(emb).astype(np.float32),
             clip=np.array(f"{inf.CLIP_MODEL}@{inf.CLIP_REVISION}"), device=np.array("cpu"))
    print(f"{OUT.relative_to(core.ROOT)}: {len(sents)} sentences x {emb[0].shape[0]}, {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
