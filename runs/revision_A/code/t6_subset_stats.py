"""Task 6 support: counts, class distributions and overlaps for subset_construction.md."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import EMOTIONS, OUT, ROOT

V2 = Path("/workspace/datasets/ArtEmis-20k-v2/ArtEmis-20k-GPU")
MANIFESTS = {  # name -> (csv, vote-count column prefix, n-votes column)
    "v1 train (seed-42 selection)": (ROOT / "data/train_artworks.csv", "count_", "n_annotations"),
    "v1 val": (ROOT / "data/val_artworks.csv", "count_", "n_annotations"),
    "v1 test": (ROOT / "data/test_artworks.csv", "count_", "n_annotations"),
    "v2 train = paper 14K": (V2 / "train_artworks.csv", "count_", "n_annotations"),
    "v2 val = eval val": (V2 / "val_artworks.csv", "count_", "n_annotations"),
    "v2 test": (V2 / "test_artworks.csv", "count_", "n_annotations"),
    "eval test (43K build)": (ROOT / "data/artemis_balanced/test.csv", "n_", "n_votes"),
    "43K train": (ROOT / "data/artemis_balanced/train.csv", "n_", "n_votes"),
}


def key(s):
    return s.map(lambda p: Path(str(p)).stem.lower())


def dist_row(name, df, pre, ncol):
    c = df[[f"{pre}{e}" for e in EMOTIONS]].to_numpy(float)
    u = np.isclose(c, c.max(1, keepdims=True), atol=1e-5).sum(1) == 1
    top = np.bincount(c[u].argmax(1), minlength=9)
    r = {"split": name, "artworks": len(df), "annotations": int(df[ncol].sum()),
         "ann/artwork": round(float(df[ncol].mean()), 3), "tied": int((~u).sum()), "unique-top": int(u.sum())}
    r.update({f"top {e}": int(top[i]) for i, e in enumerate(EMOTIONS)})
    r.update({f"share {e}": round(float((c / c.sum(1, keepdims=True))[:, i].mean()), 4) for i, e in enumerate(EMOTIONS)})
    return r


def main():
    dfs = {n: pd.read_csv(p) for n, (p, _, _) in MANIFESTS.items()}
    dist = pd.DataFrame([dist_row(n, dfs[n], pre, nc) for n, (_, pre, nc) in MANIFESTS.items()])
    dist.to_csv(OUT / "t6_class_distributions.csv", index=False)

    keys = {n: set(key(d.painting)) for n, d in dfs.items()}
    names = list(keys)
    ov = pd.DataFrame([[len(keys[a] & keys[b]) for b in names] for a in names], index=names, columns=names)
    ov.to_csv(OUT / "t6_overlaps.csv")

    # cross-split near-duplicate candidates (pHash, from the cleaned-data QC) for the evaluated splits
    dup = pd.read_csv(ROOT / "Data/artemis_full_v1_20261031/duplicate_groups.csv")
    split_of = {}
    for n, s in (("43K train", "train43K"), ("v2 train = paper 14K", "train14K")):
        for k in keys[n]:
            split_of.setdefault(k, set()).add(s)
    for n, s in (("v2 val = eval val", "val"), ("eval test (43K build)", "test")):
        for k in keys[n]:
            split_of.setdefault(k, set()).add(s)
    rows = []
    for _, r in dup.iterrows():
        a, b = Path(r.filename).stem.lower(), Path(r.reference_filename).stem.lower()
        sa, sb = split_of.get(a, set()), split_of.get(b, set())
        for tr in ("train43K", "train14K"):
            for ev in ("val", "test"):
                if (tr in sa and ev in sb) or (ev in sa and tr in sb):
                    rows.append({"train set": tr, "eval split": ev, "a": a, "b": b,
                                 "hamming": r.hamming_distance, "kind": r.kind})
    nd = pd.DataFrame(rows)
    nd.to_csv(OUT / "t6_near_duplicate_cross_split.csv", index=False)
    print(dist.iloc[:, :15].to_string()); print(ov.to_string())
    print(nd.groupby(["train set", "eval split"]).size() if len(nd) else "no cross-split candidates")
    print("val-test candidates:", sum(1 for _, r in dup.iterrows()
                                      if {*split_of.get(Path(r.filename).stem.lower(), set()),
                                          *split_of.get(Path(r.reference_filename).stem.lower(), set())} >= {"val", "test"}))


if __name__ == "__main__":
    main()
