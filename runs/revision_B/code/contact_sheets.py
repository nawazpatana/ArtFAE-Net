"""Step 0a: side-by-side contact sheets for every pHash pair flagged in revision_A
(runs/revision_A/t6_near_duplicate_cross_split.csv). Read-only on revision_A; writes runs/revision_B/duplicates/.

Each sheet shows the training-side image (left) and the evaluation-side image (right) at full aspect ratio,
scaled to 520 px height, with filename, split membership and pHash Hamming distance.
"""
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/revision_B/duplicates"
H = 520


def key(p):
    return Path(str(p)).stem.lower()


def main():
    flags = pd.read_csv(ROOT / "runs/revision_A/t6_near_duplicate_cross_split.csv")
    man = {"train43K": "data/artemis_balanced/train.csv", "train14K": "data/paper14k/train.csv",
           "val": "data/artemis_balanced/val.csv", "test": "data/artemis_balanced/test.csv"}
    member, path = {}, {}
    for s, f in man.items():
        df = pd.read_csv(ROOT / f)
        for k, p in zip(df.painting.map(key), df.image_path):
            member.setdefault(k, []).append(s)
            path.setdefault(k, p if Path(p).is_absolute() else str(ROOT / p))
    pairs = (flags.groupby(["a", "b", "hamming"])
             .apply(lambda g: ";".join(sorted(f"{t}-{e}" for t, e in zip(g["train set"], g["eval split"]))),
                    include_groups=False).reset_index(name="flagged_as"))
    font = ImageFont.load_default(size=16)
    rows = []
    for i, r in pairs.iterrows():
        a_eval = any(s in ("val", "test") for s in member[r.a])
        tr, ev = (r.b, r.a) if a_eval else (r.a, r.b)
        ims = []
        for k in (tr, ev):
            with Image.open(path[k]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                ims.append((k, im.resize((max(1, round(im.width * H / im.height)), H))))
        w = sum(im.width for _, im in ims) + 30
        sheet = Image.new("RGB", (max(w, 900), H + 70), "white")
        x = 10
        d = ImageDraw.Draw(sheet)
        for k, im in ims:
            sheet.paste(im, (x, 60))
            d.text((x, 8), k[:70], fill="black", font=font)
            d.text((x, 30), "+".join(member[k]), fill="darkred", font=font)
            x += im.width + 10
        pid = f"pair_{i + 1:02d}"
        sheet.save(OUT / f"{pid}.jpg", quality=88)
        rows.append({"pair": pid, "train_side": tr, "eval_side": ev, "train_side_splits": "+".join(member[tr]),
                     "eval_side_splits": "+".join(member[ev]), "hamming": r.hamming, "flagged_as": r["flagged_as"]})
    pd.DataFrame(rows).to_csv(OUT / "pairs.csv", index=False)
    print(f"{len(rows)} contact sheets -> {OUT}")


if __name__ == "__main__":
    main()
