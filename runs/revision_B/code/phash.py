"""Pinned 64-bit DCT perceptual hash and Hamming search, for the fresh-holdout duplicate check.

The cleaning QC (Data/artemis_full_v1_20261031/provenance.json: "64-bit DCT perceptual hash, Hamming distance <=4")
did not keep its code (prepare.py, sha256 d62034c9…, is not on disk). This module pins one standard
implementation (imagehash.phash convention) and is validated against the 191 recorded candidate pairs
(validate_against_qc). Decode: full decode, EXIF orientation, RGB → L (PIL), resize 32x32 LANCZOS,
2-D DCT-II (scipy.fft.dct, norm=None, axis 0 then 1), low-frequency 8x8 block incl. DC, bit = coeff > median(block).

usage: python phash.py reference          # hash the reference images (43K train + v2 val/test + all of v2 + 49,011)
       python phash.py validate           # compare within-reference candidates to the recorded QC pairs
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dct
from torch.utils.data import DataLoader, Dataset

from common import OUT, ROOT

PREP = OUT / "holdout_prep"
THRESH = 4


def phash(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB").convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float64)
    low = dct(dct(a, axis=0), axis=1)[:8, :8]
    bits = (low > np.median(low)).ravel()
    return int(np.packbits(bits).view(">u8")[0])


class _Paths(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return i, phash(self.paths[i]), ""
        except Exception as e:  # recorded, never silently skipped
            return i, 0, f"{type(e).__name__}: {e}"


def hash_paths(paths, workers=14):
    h, err = np.zeros(len(paths), dtype=np.uint64), [""] * len(paths)
    for b in DataLoader(_Paths(list(paths)), batch_size=256, num_workers=workers, collate_fn=lambda x: x):
        for i, v, e in b:
            h[i], err[i] = np.uint64(v), e
    return h, err


def pairs_within(h, thresh=THRESH, chunk=2048):
    out = []
    for s in range(0, len(h), chunk):
        d = np.bitwise_count(h[s:s + chunk, None] ^ h[None, :])
        i, j = np.nonzero(d <= thresh)
        k = (s + i) < j
        out += list(zip(s + i[k], j[k], d[i[k], j[k]]))
    return out


def pairs_between(ha, hb, thresh=THRESH, chunk=2048):
    out = []
    for s in range(0, len(ha), chunk):
        d = np.bitwise_count(ha[s:s + chunk, None] ^ hb[None, :])
        i, j = np.nonzero(d <= thresh)
        out += list(zip(s + i, j, d[i, j]))
    return out


def reference_table():
    """Every image already used: 43K train, eval val/test (v2), v2 20K (incl. the 4 images missing from 43K)."""
    rows = []
    for split, f in (("train43K", "data/artemis_balanced/train.csv"), ("val", "data/artemis_balanced/val.csv"),
                     ("test", "data/artemis_balanced/test.csv")):
        df = pd.read_csv(ROOT / f)
        rows += [{"key": Path(p).stem.lower(), "split": split, "path": str(ROOT / p) if not Path(p).is_absolute() else p}
                 for p in df.image_path]
    v2 = Path("/home/dev/ArtFAE-Net/Data")
    for split in ("train", "val", "test"):
        df = pd.read_csv(v2 / f"{split}_artworks.csv")
        rows += [{"key": Path(p).stem.lower(), "split": f"v2_{split}", "path": str(v2 / p)} for p in df.image_path]
    return pd.DataFrame(rows).drop_duplicates("key", keep="first").reset_index(drop=True)


def main():
    PREP.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "reference":
        ref = reference_table()
        h, err = hash_paths(ref.path)
        ref["phash"] = [f"{int(x):016x}" for x in h]
        ref["error"] = err
        ref.to_csv(PREP / "reference_phash.csv", index=False)
        print(f"{len(ref)} reference images hashed, {sum(bool(e) for e in err)} errors")
    elif sys.argv[1] == "validate":
        ref = pd.read_csv(PREP / "reference_phash.csv", keep_default_na=False)
        clean = ref[ref.path.str.contains("clean_artemis/images")].reset_index(drop=True)
        h = np.array([int(x, 16) for x in clean.phash], dtype=np.uint64)
        found = {frozenset((clean.key[i], clean.key[j])): d for i, j, d in pairs_within(h)}
        qc = pd.read_csv(ROOT / "Data/artemis_full_v1_20261031/duplicate_groups.csv")
        qc = {frozenset((Path(a).stem.lower(), Path(b).stem.lower())): d
              for a, b, d in zip(qc.filename, qc.reference_filename, qc.hamming_distance)}
        both = set(found) & set(qc)
        same_d = sum(found[k] == qc[k] for k in both)
        res = {"clean images": len(clean), "QC pairs": len(qc), "pinned pairs": len(found), "in both": len(both),
               "QC recall": len(both) / len(qc), "same Hamming distance": same_d}
        pd.Series(res).to_csv(PREP / "phash_validation.csv", header=["value"])
        print(res)


if __name__ == "__main__":
    main()
