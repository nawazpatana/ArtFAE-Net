"""Empirical description of the v2 ("user-reviewed") 20K subset, whose selection procedure is unknown.

Per stratum (unique-top emotion, or tie; isclose atol 1e-5 on vote counts):
  N_artemis   artworks in the full official ArtEmis release (80,031)
  N_pool      of those, artworks with an image in the 49,011-image pool (clean_artemis/images)
  v1, v2      artworks in each 20K subset
  rate        = count / N (inclusion rate); enrichment = share in subset / share in population
  v1 kept     fraction of v1 artworks of the stratum that are also in v2; v2 from v1 = fraction of v2 from v1
Writes runs/revision_B/v2_inclusion_rates.csv and .md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import EMOTIONS, OUT, ROOT

sys.path.insert(0, str(ROOT))
import build_balanced_artemis as B  # noqa: E402


def strata(c):
    u = np.isclose(c, c.max(1, keepdims=True), atol=1e-5).sum(1) == 1
    return np.where(u, np.array(EMOTIONS)[c.argmax(1)], "tie")


def main():
    art = B.aggregate(B.load_annotations(ROOT / "artemis_official_data/official_data/artemis_dataset_release_v0.csv"))
    art["key"] = art.painting.str.lower()
    art["stratum"] = strata(art[[f"n_{e}" for e in EMOTIONS]].to_numpy(float))
    pool = {Path(f).stem.lower() for f in (ROOT / "clean_artemis/images").iterdir()}
    subsets = {}
    for name, f in (("v1", ROOT / "data/artworks_20k.csv"), ("v2", Path("/home/dev/ArtFAE-Net/Data/artworks_20k.csv"))):
        d = pd.read_csv(f)
        d["key"] = d.painting.str.lower()
        d["stratum"] = strata(d[[f"count_{e}" for e in EMOTIONS]].to_numpy(float))
        subsets[name] = d
    k1, k2 = set(subsets["v1"].key), set(subsets["v2"].key)
    order = EMOTIONS + ["tie"]
    pa = art[art.key.isin(pool)]
    rows = []
    for s in order + ["all"]:
        f = (lambda d: d) if s == "all" else (lambda d, s=s: d[d.stratum == s])
        n_a, n_p = len(f(art)), len(f(pa))
        v1, v2 = f(subsets["v1"]), f(subsets["v2"])
        rows.append({"stratum": s, "N_artemis": n_a, "N_pool": n_p, "v1": len(v1), "v2": len(v2),
                     "v1 rate (pool)": len(v1) / n_p, "v2 rate (pool)": len(v2) / n_p,
                     "v1 share": len(v1) / len(subsets["v1"]), "v2 share": len(v2) / len(subsets["v2"]),
                     "ArtEmis share": n_a / len(art), "pool share": n_p / len(pa),
                     "v1 enrich vs ArtEmis": (len(v1) / len(subsets["v1"])) / (n_a / len(art)),
                     "v2 enrich vs ArtEmis": (len(v2) / len(subsets["v2"])) / (n_a / len(art)),
                     "v2/v1 rate ratio": len(v2) / max(len(v1), 1),
                     "v1 kept in v2": len(set(v1.key) & k2) / max(len(v1), 1),
                     "v2 from v1": len(set(v2.key) & k1) / max(len(v2), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v2_inclusion_rates.csv", index=False)
    md = ["| stratum | N ArtEmis | N pool (49,011) | v1 | v2 | v1 rate (pool) | v2 rate (pool) | ArtEmis share | "
          "v1 share | v2 share | v1 enrich. | v2 enrich. | v2/v1 | v1 kept in v2 | v2 from v1 |",
          "|---|" + "---|" * 14]
    for r in df.itertuples(index=False):
        md.append(f"| {r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:,} | {r[4]:,} | {r[5]:.3f} | {r[6]:.3f} | {r[9]:.4f} | "
                  f"{r[7]:.4f} | {r[8]:.4f} | {r[11]:.2f} | {r[12]:.2f} | {r[13]:.2f} | {r[14]:.3f} | {r[15]:.3f} |")
    (OUT / "v2_inclusion_rates.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
