"""Phase C fresh holdout: composition report. Reads only manifests and official labels (no model output, no logits).

Writes <prep>/composition.json and <prep>/composition.md:
  size, unique-top count and tied-vote fraction; excluded artworks by reason (already used, ambiguous painting
  name, undecodable, duplicate rule) plus unused artworks without an image in the WikiArt folder; class distribution
  (unique-top counts and mean vote shares) against the natural ArtEmis distribution (all aggregated artworks of the
  official release v0); artists absent from the 43K train and their artwork count.
usage: python holdout_composition.py [--prep DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from common import OUT, ROOT

sys.path.insert(0, str(ROOT))
import build_balanced_artemis as B  # noqa: E402

from build_fresh_holdout import OFFICIAL, key  # noqa: E402

E = B.EMOTIONS


def dist(df):
    ut = df[df.unique_top.astype(bool)]
    return {"n": len(df), "n_unique_top": len(ut), "tied_fraction": 1 - len(ut) / len(df),
            "unique_top_counts": {e: int((ut.top_emotion == e).sum()) for e in E},
            "unique_top_share": {e: float((ut.top_emotion == e).mean()) for e in E},
            "mean_vote_share": {e: float(df[f"p_{e}"].mean()) for e in E},
            "mean_votes_per_artwork": float(df.n_votes.mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prep", default=str(OUT / "fresh_holdout")); a = ap.parse_args()
    prep = Path(a.prep)
    h = pd.read_csv(prep / "holdout.csv")
    ex = pd.read_csv(prep / "excluded.csv")
    nat = B.aggregate(B.load_annotations(OFFICIAL))

    def cat(r):
        r = str(r)
        return ("already used" if r == "already used" else "ambiguous painting name (stem -> >1 artwork_id)"
                if r.startswith("painting stem maps") else "undecodable" if r.startswith("decode failure")
                else "duplicate rule" if r.startswith("duplicate rule") else r)
    ex["category"] = ex.reason.map(cat)
    by = ex.groupby("category").size().to_dict()
    # artworks (not stems) behind the ambiguous stems
    amb = set(ex[ex.category.str.startswith("ambiguous")].painting)
    nat["key"] = nat.painting.map(key)
    n_amb_artworks = int(nat.key.isin(amb).sum())
    eligible_pre_image = set(nat.key) - amb - set(ex[ex.category == "already used"].painting)
    accounted = set(h.painting.map(key)) | set(ex[ex.category.isin(["undecodable", "duplicate rule"])].painting)
    no_image = sorted(eligible_pre_image - accounted)

    dup = ex[ex.category == "duplicate rule"].reason.value_counts().to_dict()
    und = ex[ex.category == "undecodable"][["painting", "reason"]].to_dict("records")
    art = h.painting.str.split("_").str[0].str.lower()
    unseen = ~h.artist_in_43k_train.astype(bool)
    comp = {
        "holdout": dist(h),
        "natural_artemis_release_v0": dist(nat.drop(columns="key")),
        "exclusions": {
            "official_artworks_aggregated": len(nat),
            "ambiguous painting name: stems": by.get("ambiguous painting name (stem -> >1 artwork_id)", 0),
            "ambiguous painting name: artworks": n_amb_artworks,
            "already used": by.get("already used", 0),
            "unused, no image in the WikiArt folder": len(no_image),
            "undecodable": by.get("undecodable", 0),
            "undecodable_list": und,
            "duplicate rule": by.get("duplicate rule", 0),
            "duplicate rule by class": dup,
        },
        "artists": {"n_artists": int(art.nunique()),
                    "n_artists_absent_from_43k_train": int(art[unseen].nunique()),
                    "n_artworks_by_absent_artists": int(unseen.sum()),
                    "n_unique_top_artworks_by_absent_artists": int((unseen & h.unique_top.astype(bool)).sum())},
        "no_image_list": no_image,
    }
    (prep / "composition.json").write_text(json.dumps(comp, indent=1) + "\n")

    H, N = comp["holdout"], comp["natural_artemis_release_v0"]
    L = [f"# Fresh holdout composition ({prep})", "",
         f"Holdout: {H['n']:,} artworks; {H['n_unique_top']:,} unique-top; tied-vote fraction "
         f"{H['tied_fraction']:.3f} (natural ArtEmis: {N['tied_fraction']:.3f} of {N['n']:,}).", "",
         "| class | holdout unique-top n | holdout unique-top % | natural unique-top % | holdout mean share | "
         "natural mean share |", "|---|---:|---:|---:|---:|---:|"]
    for e in E:
        L.append(f"| {e} | {H['unique_top_counts'][e]:,} | {100 * H['unique_top_share'][e]:.1f} | "
                 f"{100 * N['unique_top_share'][e]:.1f} | {H['mean_vote_share'][e]:.3f} | {N['mean_vote_share'][e]:.3f} |")
    x = comp["exclusions"]
    L += ["", "Exclusions:", *[f"- {k}: {v}" for k, v in x.items() if k not in ("undecodable_list", "no_image_list")],
          *[f"  - {r['painting']}: {r['reason']}" for r in und], "",
          f"Artists: {comp['artists']['n_artists']:,} in the holdout; {comp['artists']['n_artists_absent_from_43k_train']:,} "
          f"absent from the 43K train, with {comp['artists']['n_artworks_by_absent_artists']:,} artworks "
          f"({comp['artists']['n_unique_top_artworks_by_absent_artists']:,} unique-top)."]
    (prep / "composition.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
