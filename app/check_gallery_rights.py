"""Image-rights check of the demo gallery (app/assets/examples/) with the study's rule, and replacement of failures.

Rule (study/artist_rights.py, used unchanged): the artist died before 1956, the work was created before 1931,
and it is 2D. Sources are the WikiArt artist and painting records (Wikidata P570 when WikiArt has no death date).

Replacement (the gallery rule of app/build_assets.py, continued): a failing example is replaced by the next
validation artwork in the same ranking (unique-top val artworks of the same human-voted top emotion, by annotator
agreement, highest first) that passes every rule, has an image of at least 300 px on its shorter side, is not in
the test split, and whose artist is not in the gallery (the original 12 artists included).

The WikiArt responses are cached in app/assets/examples/rights_artists.json and rights_works.json; entries
already fetched for the study (study/artist_meta.json, study/artwork_meta.json) are copied from there (read only;
study/ is never written). The outcome is written to app/assets/examples/rights_check.json.

run:  python app/check_gallery_rights.py            (report only)
      python app/check_gallery_rights.py --apply    (replace failing examples in examples.json; then rerun
                                                     app/precompute_gallery.py)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "study"))
import artist_rights as ar  # noqa: E402  (the study's rule, imported read-only)
from select_stimuli import slug_year  # noqa: E402

sys.path.insert(0, str(ROOT))
from artfae_balance import EMOTIONS  # noqa: E402

EX = APP / "assets/examples"
VAL, TEST = ROOT / "data/artemis_balanced/val.csv", ROOT / "data/artemis_balanced/test.csv"
MIN_SIDE = 300


class SeededCache(ar.Cache):
    """An app-owned cache that falls back to the study's cache (read only) before fetching."""

    def __init__(self, path, study_path):
        super().__init__(path)
        self.study = json.loads(study_path.read_text()) if study_path.exists() else {}

    def seed(self, key):
        """Copy the study's record for key (if any) before evaluate() looks it up, so it is not fetched again."""
        if key not in self.data and key in self.study:
            self.data[key] = self.study[key]


def artist_of(painting):
    return painting.split("_", 1)[0]


def rank(v, emotion):
    P = v[[f"p_{e}" for e in EMOTIONS]].to_numpy()
    g = v[(P.argmax(1) == EMOTIONS.index(emotion)) & v.unique_top.astype(bool)].copy()
    g["agree"] = P[g.index].max(1)
    return g.sort_values("agree", ascending=False, kind="stable")


def check(ac, wc, row):
    slug = artist_of(row.painting)
    ac.seed(slug); wc.seed(row.artwork_id)
    return ar.evaluate(ac, wc, row.artwork_id, slug, slug_year(row.painting))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    v, test = pd.read_csv(VAL), set(pd.read_csv(TEST).image_path)
    examples = json.loads((EX / "examples.json").read_text())
    ac = SeededCache(EX / "rights_artists.json", ROOT / "study/artist_meta.json")
    wc = SeededCache(EX / "rights_works.json", ROOT / "study/artwork_meta.json")
    gallery_artists = {artist_of(v.loc[e["val_row"], "painting"]) for e in examples}
    used_rows = {e["val_row"] for e in examples}
    report, new_examples = [], []
    for e in examples:
        row = v.loc[e["val_row"]]
        assert Path(row.image_path).name == e["file"], e["file"]
        ev = check(ac, wc, row)
        item = {"file": e["file"], "artwork_id": row.artwork_id, "annotator_top": e["annotator_top"], **ev,
                "passes": ev["reason"] is None}
        if ev["reason"] is None:
            new_examples.append(e)
        else:
            for idx, c in rank(v, e["annotator_top"]).iterrows():
                if idx in used_rows or artist_of(c.painting) in gallery_artists or c.image_path in test:
                    continue
                if not Path(c.image_path).exists():
                    continue
                with Image.open(c.image_path) as im:
                    if min(im.size) < MIN_SIDE:
                        continue
                cev = check(ac, wc, c)
                if cev["reason"] is not None:
                    continue
                name = Path(c.image_path).name
                title = c.painting.split("_", 1)[-1].replace("-", " ").strip()
                rep = {"file": name, "artwork_id": c.artwork_id, "painting": title[:1].upper() + title[1:],
                       "artist": name.split("_")[0].replace("-", " ").title(), "style": c.art_style.replace("_", " "),
                       "val_row": int(idx), "annotator_top": e["annotator_top"], "n_votes": int(sum(c[f"n_{k}"] for k in EMOTIONS)),
                       "annotators": {k: float(c[f"p_{k}"]) for k in EMOTIONS}}
                item["replacement"] = {"file": name, "artwork_id": c.artwork_id, "val_row": int(idx),
                                       "agreement": float(c[f"p_{e['annotator_top']}"]), **cev}
                used_rows.add(int(idx)); gallery_artists.add(artist_of(c.painting))
                new_examples.append(rep)
                break
            else:
                raise SystemExit(f"no passing replacement for {e['file']}")
        report.append(item)
        print(("PASS " if item["passes"] else "FAIL ") + e["file"] + ("" if item["passes"] else
              f"  ({ev['reason']})  -> {item['replacement']['file']}"), flush=True)
    ac.save(); wc.save()
    out = {"rule": {"death_before": ar.DEATH_BEFORE, "created_before": ar.CREATED_BEFORE, "form": "2D",
                    "source": "study/artist_rights.py evaluate()"}, "items": report}
    (EX / "rights_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    if a.apply:
        for old, new in zip(examples, new_examples):
            if old["file"] != new["file"]:
                shutil.copy2(v.loc[new["val_row"], "image_path"], EX / new["file"])
                (EX / old["file"]).unlink()
        (EX / "examples.json").write_text(json.dumps(new_examples, indent=1, ensure_ascii=False))
        print("examples.json updated; now rerun app/precompute_gallery.py")
    print(f"{sum(i['passes'] for i in report)}/{len(report)} pass")


if __name__ == "__main__":
    main()
