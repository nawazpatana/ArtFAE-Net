"""Choose the 60 study stimuli (+1 practice, +2 attention-check images) from the FRESH HOLDOUT. Deterministic.

Strata (60):
  45 unique-top artworks: for each of the 9 ArtEmis emotions, 3 high-agreement (top vote share >= 0.6) and
     2 low-agreement (unique top, share <= 0.4); anger, fear and disgust are therefore always included
   9 tied artworks (two or more emotions share the top vote)
   3 dark artworks (lowest 5% mean luminance in the holdout feature cache)
   3 minimalist artworks (lowest 5% edge density and below-median saturation)
Within a stratum, candidates are ranked: year (from the WikiArt slug) before 1950 > year unknown > 1950 or later,
then a seeded random order, and one artist is used at most once overall. Copyright: "before 1950" is only a proxy;
check the artist's death year before publishing images (see README).

Outputs
  study/stimuli.csv            stimulus_id, role, stratum, artwork_id, art_style, artist, title, year, year_known,
                               image (study-local path) -- no vote data
  study/private/votes.csv      stimulus_id, artwork_id, n_votes, n_<emotion>, p_<emotion>, top_emotions -- never shown
  study/stimuli/images/*.jpg   resized copies (longest side 1200 px) of the WikiArt images

run:  python study/select_stimuli.py     (any Python with numpy, pandas, pillow)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STUDY = Path(__file__).resolve().parent
HOLDOUT = ROOT / "runs/revision_B/fresh_holdout/holdout.csv"
ATTRS = ROOT / "features_l14_fresh_holdout/attributes.npy"  # row-aligned with holdout.csv
sys.path.insert(0, str(STUDY))
from common import EMOTIONS  # noqa: E402

SEED = 20261112
N_HIGH, N_LOW, N_TIED, N_DARK, N_MIN = 3, 2, 9, 3, 3
HIGH, LOW = 0.6, 0.4
ATTR_NAMES = ["red", "green", "blue", "hue_sin", "hue_cos", "saturation", "saturation_var", "value", "warm", "cool",
              "luminance", "luminance_std", "dark", "bright", "contrast_p95_p05", "edge_density", "edge_tl", "edge_tr",
              "edge_bl", "edge_br", *(f"hue_bin_{i}" for i in range(8))]


def slug_year(painting: str):
    m = re.search(r"-(1[0-9]{3}|20[0-2][0-9])(?:-\d{1,2})?$", painting)
    return int(m.group(1)) if m else None


def title_from_slug(painting: str) -> str:
    t = painting.split("_", 1)[1] if "_" in painting else painting
    t = re.sub(r"-(1[0-9]{3}|20[0-2][0-9])(?:-\d{1,2})?$", "", t)
    t = re.sub(r"-\d{1,2}$", "", t)
    return t.replace("-", " ").strip().capitalize() or "Untitled"


def artist_name(slug: str) -> str:
    return " ".join(w.capitalize() for w in slug.split("-"))


def year_rank(y):
    return 0 if (y is not None and y < 1950) else (1 if y is None else 2)


def load_pool():
    """The holdout with vote, agreement, brightness, edge and year columns, in the deterministic random order."""
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(HOLDOUT)
    a = np.load(ATTRS)
    assert len(a) == len(d), "feature cache is not row-aligned with holdout.csv"
    from_attr = lambda k: a[:, ATTR_NAMES.index(k)]  # noqa: E731
    d["luminance"], d["edges"], d["sat"] = from_attr("luminance"), from_attr("edge_density"), from_attr("saturation")
    n = d[[f"n_{e}" for e in EMOTIONS]].to_numpy()
    d["n_votes"] = n.sum(1)
    d["top_share"] = n.max(1) / d["n_votes"]
    d["top_emotions"] = ["|".join(e for e, c in zip(EMOTIONS, row) if c == row.max()) for row in n]
    d["year"] = d.painting.map(slug_year)
    d["rand"] = rng.random(len(d))
    d["yrank"] = d.year.map(year_rank)
    return d[d.image_path.map(lambda p: Path(p).exists())]


def stratum_pool(d, stratum: str):
    """All holdout works eligible for a stratum name as written in stimuli.csv, in selection rank order."""
    uniq = d[d.unique_top]
    if stratum.endswith("/high") or stratum.endswith("/low"):
        e, level = stratum.split("/")
        c = uniq[(uniq.top_emotion == e) & ((uniq.top_share >= HIGH) if level == "high" else (uniq.top_share <= LOW))]
    elif stratum == "tied":
        c = d[~d.unique_top]
    elif stratum == "dark":
        c = d[d.luminance <= d.luminance.quantile(.05)]
    elif stratum == "minimalist":
        c = d[(d.edges <= d.edges.quantile(.05)) & (d.sat <= d.sat.median())]
    elif stratum in EXTRA:
        e = EXTRA[stratum]
        c = uniq[(uniq.top_emotion == e) & (uniq.top_share >= .7)]
    else:
        raise ValueError(stratum)
    return c.sort_values(["yrank", "rand"])


# extra (non-stimulus) items: stratum name -> clear, high-agreement emotion
EXTRA = {"practice": "contentment", "check_awe": "awe", "check_amusement": "amusement"}


def image_ok(path) -> bool:
    try:
        with Image.open(path) as im:
            return min(im.size) >= 300
    except Exception:  # noqa: BLE001
        return False


def save_image(sid, path, img_dir):
    img_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((1200, 1200))
        im.save(img_dir / f"{sid}.jpg", "JPEG", quality=90, optimize=True)


def public_rows(s: pd.DataFrame) -> pd.DataFrame:
    """stimuli.csv columns (no vote data) from selected holdout rows with stimulus_id, role, stratum."""
    return pd.DataFrame({
        "stimulus_id": s.stimulus_id, "role": s.role, "stratum": s.stratum, "artwork_id": s.artwork_id,
        "art_style": s.art_style.str.replace("_", " "), "artist": s.artist.map(artist_name),
        "title": s.painting.map(title_from_slug), "year": s.year.map(lambda y: "" if pd.isna(y) else str(int(y))),
        "year_known": s.year.notna(), "image": [f"stimuli/images/{i}.jpg" for i in s.stimulus_id]})


def vote_rows(s: pd.DataFrame) -> pd.DataFrame:
    votes = s[["stimulus_id", "artwork_id", "n_votes", *(f"n_{e}" for e in EMOTIONS), "top_emotions", "top_share"]].copy()
    for e in EMOTIONS:
        votes[f"p_{e}"] = s[f"n_{e}"] / s.n_votes
    return votes


def main(out_dir: Path = STUDY):
    d = load_pool()
    used_art, used_artist, rows = set(), set(), []

    def take(stratum, k, role="stimulus", label=None):
        cand = stratum_pool(d, stratum)
        cand = cand[~cand.artwork_id.isin(used_art) & ~cand.artist.isin(used_artist)]
        got = 0
        for _, r in cand.iterrows():
            if not image_ok(r.image_path):
                continue
            rows.append({**r.to_dict(), "stratum": label or stratum, "role": role})
            used_art.add(r.artwork_id); used_artist.add(r.artist)
            got += 1
            if got == k:
                break
        if got < k:
            print(f"WARNING: stratum {stratum}: only {got} of {k}")

    for e in EMOTIONS:
        take(f"{e}/high", N_HIGH)
        take(f"{e}/low", N_LOW)
    take("tied", N_TIED)
    take("dark", N_DARK)
    take("minimalist", N_MIN)
    n_stim = len(rows)
    # practice + two attention-check images: clear, high-agreement works outside the 60
    take("practice", 1, role="practice")
    take("check_awe", 1, role="check")
    take("check_amusement", 1, role="check")

    s = pd.DataFrame(rows)
    s.insert(0, "stimulus_id", [f"S{i + 1:02d}" for i in range(n_stim)] + ["P01", "C01", "C02"])
    for sid, p in zip(s.stimulus_id, s.image_path):
        save_image(sid, p, out_dir / "stimuli/images")
    out = public_rows(s)
    out.to_csv(out_dir / "stimuli.csv", index=False)
    (out_dir / "private").mkdir(parents=True, exist_ok=True)
    vote_rows(s).to_csv(out_dir / "private/votes.csv", index=False)
    st = out[out.role == "stimulus"]
    print(f"{len(st)} stimuli + {len(out) - len(st)} extra; year < 1950: {(st.year_known & (st.year.replace('', '9999').astype(int) < 1950)).sum()}, "
          f"year unknown: {(~st.year_known).sum()}")
    print(st.stratum.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY)
