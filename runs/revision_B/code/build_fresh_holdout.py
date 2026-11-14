"""Phase C fresh holdout: ArtEmis artworks whose images are in the full WikiArt set but that were never used.

Stage `candidates` (run once the full WikiArt image folder exists):
  1. Aggregate the official ArtEmis release exactly as build_balanced_artemis.aggregate (9 emotions,
     "something else" -> something_else, soft targets, present_*, unique_top with isclose atol 1e-5).
  2. Match images exactly as build_balanced_artemis.index_images/attach_images (lower-cased filename stem,
     style-folder match first, then style-free).
  3. Exclude every artwork already used, keyed by lower-cased painting stem (style ignored, conservative):
     the 49,011 cleaned images (clean_artemis/images), every manifest ever used (43K train/val/test, v1 20K,
     v2 20K, paper 14K, locked20k, 30k), and the 13 cleaning-QC invalid images. Also exclude painting stems that
     map to more than one ArtEmis artwork_id (ambiguous identity).
  4. Full-decode every remaining image (artfae_balance.extract_features.load_image, plus a pinned full decode
     for the pHash); failures are excluded and listed.
  5. pHash (code/phash.py) every remaining image; candidate pairs = Hamming <= 8 against every reference image
     (43K train, evaluation val/test, all of v2 20K). Threshold 8 (not the QC's 4) because the pinned pHash
     reproduces 184/191 QC pairs at <= 4 and 190/191 at <= 8 (holdout_prep/phash_validation.csv).
  6. Contact sheet per candidate pair -> <prep>/duplicates/pair_NNNN.jpg; template classification.csv with an
     empty `classification` column.
Stage `finalize` (after every candidate has been classified visually as in revision_B Step 0):
  7. Every pair must have a class ("same artwork" / "same artwork, different reproduction" / "uncertain" /
     "different") and a one-line reason in the `note` column of duplicates/classification.csv (the contact-sheet
     log); finalize refuses otherwise. Holdout artworks in any pair NOT classified "different" are excluded, i.e.
     "uncertain" counts as a duplicate.
  8. Write holdout.csv (artemis_balanced columns + image_path + artist + artist_in_43k_train) and
     excluded.csv (every exclusion with its reason), print sha256 of both. Nothing is trained on or tuned with
     this set; features are then extracted with extract_holdout_features.sh (pinned settings).

usage: python build_fresh_holdout.py candidates --images /path/to/wikiart [--prep DIR]
       python build_fresh_holdout.py finalize [--prep DIR]
Dry-run support: --pretend-unused val treats the evaluation val split as unused (exercises the pipeline on real
images; output must go to a scratch --prep directory and is never a holdout).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT, ROOT

sys.path.insert(0, str(ROOT))
import build_balanced_artemis as B  # noqa: E402

from phash import hash_paths, pairs_between  # noqa: E402

THRESH = 8
OFFICIAL = ROOT / "artemis_official_data/official_data/artemis_dataset_release_v0.csv"
USED_MANIFESTS = ["data/artemis_balanced/train.csv", "data/artemis_balanced/val.csv", "data/artemis_balanced/test.csv",
                  "data/artworks_20k.csv", "data/paper14k/train.csv",
                  "data/artemis_balanced_locked20k/train.csv", "data/artemis_balanced_locked20k/val.csv",
                  "data/artemis_balanced_locked20k/test.csv", "data/artemis_balanced_30k/train.csv",
                  "data/artemis_balanced_30k/val.csv", "data/artemis_balanced_30k/test.csv",
                  "/home/dev/ArtFAE-Net/Data/artworks_20k.csv"]
CLASSES = {"same artwork", "same artwork, different reproduction", "uncertain", "different"}


def key(s):
    return Path(str(s)).stem.lower()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def used_keys(pretend_unused=None):
    k = {key(f) for f in (ROOT / "clean_artemis/images").iterdir()}
    for m in USED_MANIFESTS:
        p = Path(m) if Path(m).is_absolute() else ROOT / m
        k |= set(pd.read_csv(p).painting.map(key))
    k |= set(pd.read_csv(ROOT / "Data/artemis_full_v1_20261031/excluded_records.csv").painting.map(key))
    if pretend_unused == "val":
        k -= set(pd.read_csv(ROOT / "data/artemis_balanced/val.csv").painting.map(key))
    return k


def candidates(a):
    prep = Path(a.prep); (prep / "duplicates").mkdir(parents=True, exist_ok=False)
    art = B.aggregate(B.load_annotations(OFFICIAL))
    art["key"] = art.painting.map(key)
    excl = []
    multi = art.key.duplicated(keep=False)
    excl += [{"painting": k, "reason": "painting stem maps to >1 ArtEmis artwork_id"} for k in art.key[multi].unique()]
    art = art[~multi]
    used = used_keys(a.pretend_unused)
    excl += [{"painting": k, "reason": "already used"} for k in art.key[art.key.isin(used)]]
    art = art[~art.key.isin(used)]
    n_unused = len(art)
    art = B.attach_images(art, B.index_images(Path(a.images)))
    print(f"[holdout] {n_unused:,} unused ArtEmis artworks, {len(art):,} with an image in {a.images}")
    h, err = hash_paths(art.image_path) if len(art) else (np.zeros(0, np.uint64), [])
    from artfae_balance.extract_features import load_image
    bad = [i for i, (p, e) in enumerate(zip(art.image_path, err)) if e or not _decodes(load_image, p)]
    excl += [{"painting": art.key.iat[i], "reason": f"decode failure {err[i]}"} for i in bad]
    keep = np.setdiff1d(np.arange(len(art)), bad)
    art, h = art.iloc[keep].reset_index(drop=True), h[keep]
    art["phash"] = [f"{int(x):016x}" for x in h]
    ref = pd.read_csv(OUT / "holdout_prep/reference_phash.csv", keep_default_na=False)
    if a.pretend_unused == "val":
        ref = ref[ref.split != "val"].reset_index(drop=True)
    hr = np.array([int(x, 16) for x in ref.phash], dtype=np.uint64)
    pairs = pairs_between(h, hr, THRESH) if len(h) else []
    pdf = pd.DataFrame([{"pair": f"pair_{n + 1:04d}", "holdout_side": art.key.iat[i], "reference_side": ref.key.iat[j],
                         "reference_split": ref.split.iat[j], "hamming": int(d), "classification": "", "note": ""}
                        for n, (i, j, d) in enumerate(pairs)])
    art.drop(columns=["key"]).to_csv(prep / "holdout_candidates.csv", index=False)
    pd.DataFrame(excl).to_csv(prep / "excluded_stage1.csv", index=False)
    pdf.to_csv(prep / "duplicates/classification.csv", index=False)
    if len(pdf):
        _sheets(pdf, art.set_index(art.painting.map(key)).image_path, ref.set_index("key").path, prep / "duplicates")
    print(f"[holdout] {len(art):,} decodable candidate artworks; {len(pdf):,} pHash pairs (<= {THRESH}) to classify "
          f"in {prep / 'duplicates/classification.csv'}")


def _decodes(load_image, p):
    try:
        load_image(p).load()
        return True
    except Exception:
        return False


def _sheets(pdf, hold_path, ref_path, out):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    font, H = ImageFont.load_default(size=16), 520
    for r in pdf.itertuples():
        ims = []
        for k, p in ((r.reference_side, ref_path[r.reference_side]), (r.holdout_side, hold_path[r.holdout_side])):
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                ims.append((k, im.resize((max(1, round(im.width * H / im.height)), H))))
        sheet = Image.new("RGB", (max(sum(i.width for _, i in ims) + 30, 900), H + 70), "white")
        d, x = ImageDraw.Draw(sheet), 10
        for (k, im), lab in zip(ims, (f"reference ({r.reference_split})", f"holdout candidate, pHash {r.hamming}")):
            sheet.paste(im, (x, 60)); d.text((x, 8), k[:70], fill="black", font=font)
            d.text((x, 30), lab, fill="darkred", font=font); x += im.width + 10
        sheet.save(out / f"{r.pair}.jpg", quality=88)


def finalize(a):
    prep = Path(a.prep)
    art = pd.read_csv(prep / "holdout_candidates.csv")
    cls = pd.read_csv(prep / "duplicates/classification.csv", keep_default_na=False)
    if len(cls) and not cls.classification.isin(CLASSES).all():
        sys.exit(f"{(~cls.classification.isin(CLASSES)).sum()} pairs are not classified; classify every pair first")
    no_reason = cls.note.astype(str).str.strip().eq("") | cls.note.astype(str).str.contains("\n")
    if len(cls) and no_reason.any():
        sys.exit(f"{int(no_reason.sum())} pairs lack a one-line reason in `note`; record one for every pair")
    dup = set(cls[cls.classification.isin(CLASSES - {"different"})].holdout_side)
    excl = pd.read_csv(prep / "excluded_stage1.csv")
    why = {k: c for k, c in zip(cls.holdout_side, cls.classification) if c != "different"}
    excl = pd.concat([excl, pd.DataFrame([{"painting": k, "reason": f"duplicate rule: pair classified '{why[k]}'"}
                                          for k in sorted(dup)])])
    art = art[~art.painting.map(key).isin(dup)].reset_index(drop=True)
    tr = pd.read_csv(ROOT / "data/artemis_balanced/train.csv")
    seen = set(tr.painting.str.split("_").str[0].str.lower())
    art["artist"] = art.painting.str.split("_").str[0].str.lower()
    art["artist_in_43k_train"] = art.artist.isin(seen)
    art.to_csv(prep / "holdout.csv", index=False)
    excl.to_csv(prep / "excluded.csv", index=False)
    print(f"[holdout] final: {len(art):,} artworks ({int(art.unique_top.sum()):,} unique-top), "
          f"{int((~art.artist_in_43k_train).sum()):,} by artists absent from the 43K train; "
          f"{len(dup)} excluded under the duplicate rule (all classes except 'different', incl. uncertain)")
    print(f"holdout.csv sha256 {sha(prep / 'holdout.csv')}\nexcluded.csv sha256 {sha(prep / 'excluded.csv')}\n"
          f"classification.csv sha256 {sha(prep / 'duplicates/classification.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["candidates", "finalize"])
    ap.add_argument("--images")
    ap.add_argument("--prep", default=str(OUT / "fresh_holdout"))
    ap.add_argument("--pretend-unused", choices=["val"], default=None, help="dry-run only")
    a = ap.parse_args()
    if a.pretend_unused and Path(a.prep).resolve() == (OUT / "fresh_holdout").resolve():
        sys.exit("--pretend-unused is a dry run; use a scratch --prep directory")
    (candidates if a.stage == "candidates" else finalize)(a)


if __name__ == "__main__":
    main()
