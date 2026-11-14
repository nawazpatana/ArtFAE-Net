#!/usr/bin/env python3
"""
Build a rebalanced ArtEmis subset that improves anger / fear recognition.

What it does
  1. Reads the full ArtEmis annotation CSV (~455K rows) and aggregates votes per artwork.
  2. Matches each artwork to an image in the WikiArt image folder (extracted archive.zip).
  3. Keeps validation / test at the NATURAL class distribution
     (either your locked 20k val/test manifests, or a new stratified split).
  4. Builds an ENRICHED training split: every artwork where anger / fear (and other rare
     classes) received a meaningful share of votes, plus a class-capped stratified fill.
  5. Writes manifests with soft targets, plurality label, tie flag, "emotion present"
     multi-hot columns (for an auxiliary head) and a per-artwork sampler weight.

Usage (from VS Code terminal, after unzipping the archives):
  python build_balanced_artemis.py \
      --annotations path/to/artemis_dataset_release_v0.csv \
      --images path/to/wikiart \
      --out data/artemis_balanced \
      --locked-split path/to/ArtEmis-20k   # optional: reuse old val/test manifests

  Use --train-size 0 to put ALL remaining artworks in train (recommended if compute allows).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EMOTIONS = ["amusement", "awe", "contentment", "excitement",
            "anger", "disgust", "fear", "sadness", "something_else"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def norm_key(s: str) -> str:
    """Normalise a painting / file name so CSV names and file names match.

    Punctuation is kept: "untitled-1971-1" and "untitled-1971(1)" are different paintings.
    """
    return Path(str(s)).stem.lower()


def load_annotations(path: Path) -> pd.DataFrame:
    paths = [path] if path.is_file() else sorted(path.rglob("*.csv"))
    for p in paths:
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        if "emotion" in cols and "painting" in cols:
            df = df.rename(columns={cols["emotion"]: "emotion", cols["painting"]: "painting"})
            if "art_style" in cols:
                df = df.rename(columns={cols["art_style"]: "art_style"})
            else:
                df["art_style"] = ""
            print(f"[annotations] {p} -> {len(df):,} rows")
            return df
    sys.exit("No CSV with 'emotion' and 'painting' columns found under --annotations")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["emotion"] = (df["emotion"].astype(str).str.strip().str.lower()
                     .str.replace(" ", "_"))
    df = df[df["emotion"].isin(EMOTIONS)]
    df["artwork_id"] = df["art_style"].astype(str) + "/" + df["painting"].astype(str)
    counts = (df.groupby(["artwork_id", "art_style", "painting", "emotion"])
                .size().unstack("emotion", fill_value=0)
                .reindex(columns=EMOTIONS, fill_value=0).reset_index())
    c = counts[EMOTIONS].to_numpy(float)
    n = c.sum(1, keepdims=True)
    probs = c / n
    for i, e in enumerate(EMOTIONS):
        counts[f"n_{e}"] = c[:, i].astype(int)
        counts[f"p_{e}"] = probs[:, i]
        counts[f"present_{e}"] = (c[:, i] > 0).astype(int)   # auxiliary multi-label target
    counts["n_votes"] = n[:, 0].astype(int)
    top = c.max(1, keepdims=True)
    is_top = np.isclose(c, top, atol=1e-5)
    counts["unique_top"] = is_top.sum(1) == 1
    counts["top_emotion"] = np.where(counts["unique_top"],
                                     np.array(EMOTIONS)[c.argmax(1)], "tie")
    counts["dominant"] = np.array(EMOTIONS)[c.argmax(1)]  # tie-broken, for stratification only
    return counts.drop(columns=EMOTIONS)


def index_images(root: Path) -> dict:
    idx = {}
    for dirpath, _, files in os.walk(root):
        style = Path(dirpath).name
        for f in files:
            if Path(f).suffix.lower() in IMG_EXT:
                full = str(Path(dirpath) / f)
                idx.setdefault((norm_key(style), norm_key(f)), full)
                idx.setdefault(("", norm_key(f)), full)
    print(f"[images] indexed {len(idx)//2:,} files under {root}")
    return idx


def attach_images(art: pd.DataFrame, idx: dict) -> pd.DataFrame:
    def find(r):
        return (idx.get((norm_key(r.art_style), norm_key(r.painting)))
                or idx.get(("", norm_key(r.painting))))
    art = art.copy()
    art["image_path"] = [find(r) for r in art.itertuples()]
    missing = art["image_path"].isna().sum()
    print(f"[images] matched {len(art)-missing:,} / {len(art):,} artworks ({missing:,} missing)")
    return art.dropna(subset=["image_path"])


def load_locked_ids(root: Path):
    """Find val/test manifests from the old 20k study; return sets of painting keys."""
    out = {}
    for split in ("val", "validation", "test"):
        for p in root.rglob(f"*{split}*.csv"):
            df = pd.read_csv(p)
            col = next((c for c in df.columns if c.lower() in
                        ("painting", "image_id", "artwork_id", "image", "filename", "image_path")), None)
            if col:
                key = "val" if split.startswith("val") else "test"
                out.setdefault(key, set()).update(df[col].map(norm_key))
                print(f"[locked] {p.name}: {len(df):,} rows via column '{col}' -> {key}")
                break
    return out


def stratified_split(df, frac, seed):
    rng = np.random.default_rng(seed)
    take = []
    for _, g in df.groupby("dominant"):
        k = int(round(len(g) * frac))
        take.extend(rng.choice(g.index, size=k, replace=False))
    return df.loc[take], df.drop(index=take)


def enrich_train(pool, size, rare, min_share, seed):
    rng = np.random.default_rng(seed)
    if size <= 0 or size >= len(pool):
        return pool.copy()
    # 1) priority: every artwork where a rare class got a real share of votes
    mask = np.zeros(len(pool), bool)
    for e in rare:
        mask |= pool[f"p_{e}"].to_numpy() >= min_share
    priority = pool[mask]
    if len(priority) >= size:
        return priority.sample(size, random_state=seed)
    rest = pool[~mask]
    budget = size - len(priority)
    # 2) class-capped stratified fill by dominant emotion (water-filling)
    groups = {e: list(rng.permutation(g.index)) for e, g in rest.groupby("dominant")}
    chosen = []
    while budget > 0 and any(groups.values()):
        active = [e for e in groups if groups[e]]
        cap = max(1, budget // len(active))
        for e in active:
            k = min(cap, len(groups[e]), budget)
            chosen.extend(groups[e][:k]); groups[e] = groups[e][k:]; budget -= k
            if budget == 0:
                break
    return pd.concat([priority, rest.loc[chosen]])


def add_sampler_weights(train, power):
    freq = train["dominant"].value_counts(normalize=True)
    w = train["dominant"].map(lambda e: (1.0 / freq[e]) ** power)
    train = train.copy()
    train["sampler_weight"] = w / w.mean()
    return train


def summary(name, df):
    s = pd.DataFrame({
        "unique_top": df[df.unique_top]["top_emotion"].value_counts().reindex(EMOTIONS, fill_value=0),
        "mean_share": pd.Series({e: df[f"p_{e}"].mean() for e in EMOTIONS}).round(3),
        "any_vote": pd.Series({e: int(df[f"present_{e}"].sum()) for e in EMOTIONS}),
    })
    print(f"\n=== {name}: {len(df):,} artworks, {int(df.n_votes.sum()):,} votes, "
          f"{(~df.unique_top).sum():,} tied-top ===")
    print(s.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--locked-split", type=Path, help="folder with old val/test manifests to keep fixed")
    ap.add_argument("--quarantine", type=Path, help="quarantine_manifest.csv; integrity_status=invalid images are dropped")
    ap.add_argument("--train-size", type=int, default=30000, help="0 = all remaining artworks")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--rare", default="anger,fear,disgust", help="classes to enrich in train")
    ap.add_argument("--min-share", type=float, default=0.2, help="vote share counting as 'has this emotion'")
    ap.add_argument("--sampler-power", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    art = aggregate(load_annotations(a.annotations))
    art = attach_images(art, index_images(a.images))
    art["key"] = art["painting"].map(norm_key)
    if a.quarantine:
        q = pd.read_csv(a.quarantine)
        invalid = art.key.isin(q.loc[q.integrity_status == "invalid", "painting"].map(norm_key))
        print(f"[quarantine] dropped {invalid.sum():,} artworks with undecodable images")
        art = art[~invalid]

    if a.locked_split:
        locked = load_locked_ids(a.locked_split)
        val = art[art.key.isin(locked.get("val", set()))]
        test = art[art.key.isin(locked.get("test", set()))]
        pool = art[~art.key.isin(locked.get("val", set()) | locked.get("test", set()))]
    else:
        test, rest = stratified_split(art, a.test_frac, a.seed)
        val, pool = stratified_split(rest, a.val_frac / (1 - a.test_frac), a.seed + 1)

    rare = [e.strip() for e in a.rare.split(",") if e.strip()]
    train = add_sampler_weights(enrich_train(pool, a.train_size, rare, a.min_share, a.seed),
                                a.sampler_power)
    assert not (set(train.artwork_id) & set(val.artwork_id) | set(train.artwork_id) & set(test.artwork_id))
    for x, y in ((train, val), (train, test), (val, test)):
        assert not set(x.image_path) & set(y.image_path)

    a.out.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", train), ("val", val), ("test", test)):
        df.drop(columns=["key"]).to_csv(a.out / f"{name}.csv", index=False)
        summary(name, df)
    print(f"\nWrote manifests to {a.out}")


if __name__ == "__main__":
    main()
