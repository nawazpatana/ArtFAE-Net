"""Step 2: artist-held-out split of the 43K train set (no val or test involved).

artist = painting slug before the first underscore.
1. Artists are stratified by number of 43K-train artworks into bins 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127,
   128-255, >=256. In each bin round(10%) of the artists (at least 1 if the bin has >= 5 artists) are drawn at
   random (rng seed 0). All artworks of these artists form holdout_unseen.
2. holdout_seen: the same number of artworks drawn at random from the other artists' artworks, constrained so that
   every drawn artwork's artist keeps at least one artwork in training (so the artist is seen).
3. select: 5% of the remaining artworks drawn at random (each artist keeps >= 1 training artwork), used only for early stopping / checkpoint selection
   (replaces val, which this experiment must not touch).
4. train: everything else.
Writes runs/revision_B/step2/split.csv in 43K manifest row order with a `role` column.
"""
import numpy as np
import pandas as pd

from common import OUT, ROOT

BINS = [1, 2, 4, 8, 16, 32, 64, 128, 256, np.inf]


def main():
    rng = np.random.default_rng(0)
    df = pd.read_csv(ROOT / "data/artemis_balanced/train.csv")
    df["artist"] = df.painting.str.split("_").str[0].str.lower()
    n_art = df.artist.value_counts()
    strata = pd.cut(n_art, BINS, right=False, labels=[f"{BINS[i]}-{BINS[i+1]-1}" for i in range(len(BINS) - 1)])
    held = []
    rows = []
    for b, g in n_art.groupby(strata, observed=True):
        k = int(round(0.10 * len(g))) or (1 if len(g) >= 5 else 0)
        pick = list(rng.choice(g.index.values, size=k, replace=False)) if k else []
        held += pick
        rows.append({"stratum (artworks/artist)": b, "artists": len(g), "held-out artists": k,
                     "held-out artworks": int(g[pick].sum()) if k else 0, "artworks": int(g.sum())})
    role = np.full(len(df), "train", dtype=object)
    unseen = df.artist.isin(held).to_numpy()
    role[unseen] = "holdout_unseen"
    n_hold = int(unseen.sum())

    # seen control: random artworks of other artists, each artist keeps >= 1 artwork in training
    cand = np.flatnonzero(~unseen)
    perm = rng.permutation(cand)
    left = df.artist[~unseen].value_counts().to_dict()
    chosen = []
    for i in perm:
        a = df.artist.iat[i]
        if left[a] > 1:
            chosen.append(i); left[a] -= 1
            if len(chosen) == n_hold:
                break
    role[chosen] = "holdout_seen"
    rest = np.flatnonzero(role == "train")
    n_sel = int(round(0.05 * len(rest)))
    left = pd.Series(df.artist.values[rest]).value_counts().to_dict()
    sel = []
    for i in rng.permutation(rest):  # every artist keeps >= 1 training artwork
        a = df.artist.iat[i]
        if left[a] > 1:
            sel.append(i); left[a] -= 1
            if len(sel) == n_sel:
                break
    role[sel] = "select"
    df["role"] = role
    (OUT / "step2").mkdir(parents=True, exist_ok=True)
    df[["artwork_id", "painting", "artist", "art_style", "role"]].to_csv(OUT / "step2/split.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "step2/strata.csv", index=False)

    tr_art = set(df.artist[df.role == "train"])
    summ = df.groupby("role").agg(artworks=("artist", "size"), artists=("artist", "nunique"),
                                  tied=("unique_top", lambda s: int((~s).sum())))
    summ["artists also in train"] = [len(set(df.artist[df.role == r]) & tr_art) for r in summ.index]
    print(pd.DataFrame(rows).to_string(index=False)); print(summ.to_string())
    assert not set(df.artist[df.role == "holdout_unseen"]) & tr_art
    assert set(df.artist[df.role == "holdout_seen"]) <= tr_art


if __name__ == "__main__":
    main()
