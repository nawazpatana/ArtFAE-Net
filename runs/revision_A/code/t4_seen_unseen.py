"""Task 4: test split by whether the artwork's artist / art_style occurs in each model's training set.

artist = the WikiArt painting slug before the first underscore (e.g. "pablo-picasso_untitled-1920-2"
-> "pablo-picasso"). art_style = manifest column. Training set = config.json train_manifest of the
run (R0n: data/paper14k/train.csv; all others: data/artemis_balanced/train.csv).

macro-F1 and acc on unique-top artworks of the group; macro-F1 averages over the classes with
support > 0 in that group (classes absent from a small group would otherwise count as F1 = 0).
softCE (T = 1) on all artworks of the group. Mean ± SD over seeds 42–46.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import MODELS, OUT, ROOT, SEEDS, Data, F, dist_metrics, fmt, softmax


def artist(p):
    return str(p).split("_")[0].lower()


def main():
    d = Data()
    tt, test = d.tt, d.test_df
    yt = tt.y.argmax(1)
    rows, counts = [], []
    for lab, (m, k) in MODELS.items():
        manifests = {json.loads((F.run_dir(m, s) / "config.json").read_text())["train_manifest"] for s in SEEDS}
        assert len(manifests) == 1
        tr = pd.read_csv(ROOT / manifests.pop())
        seen = {"artist": test.painting.map(artist).isin(set(tr.painting.map(artist))).to_numpy(),
                "art_style": test.art_style.isin(set(tr.art_style)).to_numpy()}
        for attr, mask_seen in seen.items():
            for grp, mask in (("seen", mask_seen), ("unseen", ~mask_seen)):
                u = mask & tt.uniq
                counts.append({"model": lab, "train set": f"{len(tr):,}", "attribute": attr, "group": grp,
                               "n test": int(mask.sum()), "n unique-top": int(u.sum()),
                               "n distinct": int(test.loc[mask, "painting"].map(artist).nunique() if attr == "artist"
                                                 else test.loc[mask, "art_style"].nunique())})
                for s in SEEDS:
                    p = softmax(d.logits(lab, s))
                    r = {"model": lab, "attribute": attr, "group": grp, "seed": s}
                    if u.sum():
                        labels = np.unique(yt[u])
                        pr = p[u].argmax(1)
                        r.update({"macro-F1": f1_score(yt[u], pr, labels=labels, average="macro", zero_division=0),
                                  "acc": float((pr == yt[u]).mean()), "n classes": len(labels)})
                    if mask.sum():
                        r.update(dist_metrics(p[mask], tt.y[mask]))
                    rows.append(r)
    df, cdf = pd.DataFrame(rows), pd.DataFrame(counts)
    df.to_csv(OUT / "t4_seen_unseen_per_seed.csv", index=False)
    cdf.to_csv(OUT / "t4_seen_unseen_counts.csv", index=False)

    md = ["## Task 4. Seen vs unseen artist / art_style (test)", "",
          "artist = painting slug before the first underscore; training set = the run's train manifest. "
          "macro-F1 / acc on unique-top artworks of the group (macro over classes present in the group); "
          "softCE (T = 1) on all artworks of the group. Mean ± SD over seeds 42–46.", ""]
    for attr in ("artist", "art_style"):
        md += [f"### {attr}", "",
               "| model | train set | group | n test | n unique-top | distinct | classes | macro-F1 | acc | softCE |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for lab in MODELS:
            for grp in ("seen", "unseen"):
                c = cdf[(cdf.model == lab) & (cdf.attribute == attr) & (cdf.group == grp)].iloc[0]
                g = df[(df.model == lab) & (df.attribute == attr) & (df.group == grp)]
                has = lambda col: col in g and g[col].notna().all() and len(g)
                md.append(f"| {lab} | {c['train set']} | {grp} | {c['n test']} | {c['n unique-top']} | {c['n distinct']} | "
                          f"{int(g['n classes'].iloc[0]) if has('n classes') else '–'} | "
                          + " | ".join(fmt(g[col]) if has(col) else "–" for col in ("macro-F1", "acc", "softCE")) + " |")
        md.append("")
    (OUT / "t4_seen_unseen.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
