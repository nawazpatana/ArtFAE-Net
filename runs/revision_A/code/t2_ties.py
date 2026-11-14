"""Task 2: softCE / KL / Brier on unique-top vs tied test artworks, per seed, mean ± SD over seeds.

Primary: raw probabilities (T = 1). Secondary: T fitted on val per seed and kind, as in final_eval_v2.
Integrity check first: macro-F1 and softCE(T) must reproduce runs/final_eval_v2/test_per_seed.csv.
"""
import numpy as np
import pandas as pd

from common import MODELS, OUT, ROOT, SEEDS, Data, dist_metrics, evaluate, fmt, softmax


def main():
    d = Data()
    tt = d.tt
    ref = pd.read_csv(ROOT / "runs/final_eval_v2/test_per_seed.csv")
    groups = {"all": np.ones(len(tt.y), bool), "unique-top": tt.uniq, "tied": ~tt.uniq}
    rows = []
    for lab, (m, k) in MODELS.items():
        for s in SEEDS:
            z, t = d.logits(lab, s), d.temperature(lab, s)
            r = ref[(ref.model == m) & (ref.seed == s) & (ref.kind == k)].iloc[0]
            mf1 = evaluate(softmax(z), tt.y, tt.present, tt.uniq)["macro_f1"]
            ce_t = dist_metrics(softmax(z / t), tt.y)["softCE"]
            assert abs(mf1 - r["macro-F1"]) < 1e-9 and abs(ce_t - r["softCE(T)"]) < 1e-9, (lab, s)
            for tname, tv in (("T=1", 1.0), ("T(val)", t)):
                p = softmax(z / tv)
                for g, mask in groups.items():
                    rows.append({"model": lab, "seed": s, "scaling": tname, "group": g, "n": int(mask.sum()),
                                 **dist_metrics(p[mask], tt.y[mask])})
    print("integrity: macro-F1 and softCE(T) reproduce runs/final_eval_v2/test_per_seed.csv for all 25 runs")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t2_ties_per_seed.csv", index=False)

    md = ["## Task 2. Distribution metrics on unique-top vs tied test artworks", "",
          f"Test: {int(tt.uniq.sum())} unique-top, {int((~tt.uniq).sum())} tied (top-vote tie, isclose atol 1e-5). "
          "Mean ± SD over seeds 42–46. Integrity check passed: macro-F1 and softCE(T) reproduce "
          "runs/final_eval_v2/test_per_seed.csv for all 25 model × seed runs.", ""]
    # the vote distribution itself: entropy differs strongly between groups, so give the floor
    ent = -(tt.y * np.log(tt.y + 1e-12)).sum(1)
    md += [f"Entropy of the human vote distribution (the softCE floor, = softCE − KL): unique-top "
           f"{ent[tt.uniq].mean():.4f}, tied {ent[~tt.uniq].mean():.4f}, all {ent.mean():.4f}.", ""]
    for tname in ("T=1", "T(val)"):
        md += [f"### {tname}" + (" (primary: raw probabilities)" if tname == "T=1" else
                                  " (temperature fitted on val per seed, as in final_eval_v2)"), "",
               "| model | softCE unique-top | softCE tied | KL unique-top | KL tied | Brier unique-top | Brier tied | "
               "softCE all | KL all | Brier all |", "|---|---|---|---|---|---|---|---|---|---|"]
        for lab in MODELS:
            g = df[(df.model == lab) & (df.scaling == tname)]
            cell = lambda grp, c: fmt(g[g.group == grp][c])
            md.append(f"| {lab} | " + " | ".join(cell(grp, c) for c in ("softCE", "KL", "Brier")
                                                  for grp in ("unique-top", "tied")) + " | " +
                      " | ".join(cell("all", c) for c in ("softCE", "KL", "Brier")) + " |")
        md.append("")
    (OUT / "t2_ties.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
