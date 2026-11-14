"""Task 7 support: empirical ranges of the 28 attributes and frequency of the 27 describe() sentences,
from the cached attribute arrays (no image decoding). Also checks that attr_text has at most 27 distinct
rows, i.e. the text branch receives one of 27 fixed vectors.
"""
import itertools

import numpy as np
import pandas as pd

from common import OUT, ROOT

from artfae.features.aesthetic import GROUPS, NAMES, describe  # noqa: E402

SPLITS = {"train 43K": ("data/artemis_balanced/train.csv", "features/train"),
          "train paper 14K": ("data/paper14k/train.csv", "features_paper14k/train"),
          "val": ("data/artemis_balanced/val.csv", "features/val"),
          "test": ("data/artemis_balanced/test.csv", "features/test")}


def all_sentences():
    out = []
    for b, c, col in itertools.product(("Low", "Moderate", "High"), ("low", "moderate", "high"),
                                       ("predominantly warm", "predominantly cool", "mixed or neutral")):
        out.append(f"{b} overall brightness, {c} luminance contrast, and {col} colors.")
    return out


def main():
    sents = all_sentences()
    counts, ranges, uniq = {}, None, []
    for name, (csv, fdir) in SPLITS.items():
        a = np.load(ROOT / fdir / "attributes.npy")
        assert len(a) == len(pd.read_csv(ROOT / csv))
        desc = pd.Series([describe(x) for x in a])
        assert set(desc) <= set(sents)
        counts[name] = desc.value_counts().reindex(sents, fill_value=0)
        t = np.load(ROOT / fdir / "attr_text.npy").astype(np.float64)
        # rows sharing a sentence differ only by batch-padding float noise; measure it explicitly
        means = {s: t[desc.values == s].mean(0) for s in desc.unique()}
        within = max(np.abs(t[desc.values == s] - m).max() for s, m in means.items())
        mv = np.stack(list(means.values()))
        mv = mv / np.linalg.norm(mv, axis=1, keepdims=True)
        cos = mv @ mv.T
        np.fill_diagonal(cos, np.nan)
        uniq.append({"split": name, "n": len(a), "distinct sentences": desc.nunique(),
                     "max |attr_text - sentence mean|": within,
                     "min pairwise cosine between the sentence vectors": float(np.nanmin(cos)),
                     "max pairwise cosine between the sentence vectors": float(np.nanmax(cos))})
        if name == "train 43K":
            grp = {i: g for g, idx in GROUPS.items() for i in idx}
            ranges = pd.DataFrame({"index": range(28), "name": NAMES, "group": [grp.get(i, "-") for i in range(28)],
                                   "min": a.min(0), "p5": np.percentile(a, 5, 0), "mean": a.mean(0),
                                   "p95": np.percentile(a, 95, 0), "max": a.max(0)})
    cdf = pd.DataFrame(counts)
    cdf.index.name = "sentence"
    cdf.to_csv(OUT / "t7_sentence_counts.csv")
    ranges.to_csv(OUT / "t7_attribute_ranges_train43k.csv", index=False)
    pd.DataFrame(uniq).to_csv(OUT / "t7_attr_text_check.csv", index=False)
    print(pd.DataFrame(uniq).to_string()); print(cdf.to_string()); print(ranges.round(4).to_string())


if __name__ == "__main__":
    main()
