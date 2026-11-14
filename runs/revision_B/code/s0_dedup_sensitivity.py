"""Step 0b: SENSITIVITY ANALYSIS. Re-report the final_eval_v2 test summary with the test artworks confirmed as
duplicates of training artworks (duplicates/classification.csv: "same artwork" or "same artwork, different
reproduction") removed. Uses only the existing saved test logits; no model is run, nothing is selected or tuned.

Identical procedure to scripts/final_eval_v2.py `test`: T per seed fitted on val, prior correction from val,
seed-averaged ensembles, paired bootstrap (2,000 resamples, rng seed 0, now over the reduced test set),
Holm across the 4 comparisons per metric. The full-test values are recomputed the same way for the Δ column and
checked against runs/final_eval_v2/test_per_seed.csv.
"""
import json

import numpy as np
import pandas as pd

from common import OUT, ROOT, F, Sub, fmt, softmax


def test_logits(m, s):
    pat = F.MODELS[m][2]
    p = str(pat).format(s=s) if pat else ROOT / f"runs/final_eval_v2/test/{m}_s{s}_test_logits.npy"
    return np.load(p).astype(np.float64)


def main():
    cls = pd.read_csv(OUT / "duplicates/classification.csv")
    dup = set(cls[(cls.classification != "different") & (cls.eval_side_splits == "test")].eval_side)
    test_df = pd.read_csv(F.TEST_CSV)
    keep = ~test_df.painting.str.lower().isin(dup).to_numpy()
    assert (~keep).sum() == len(dup) == 15
    tv, tt = F.Targets(F.VAL_CSV), F.Targets(F.TEST_CSV)
    vl = F.val_logits_all(tv)
    subsets = {"full": tt, "dedup": Sub(tt, keep)}
    ref = pd.read_csv(ROOT / "runs/final_eval_v2/test_per_seed.csv")

    rows, ens_p = [], {}
    for m in F.MODELS:
        for s in F.SEEDS:
            raw = test_logits(m, s)
            for k in F.KINDS:
                lt = raw + (vl[(m, s)]["delta"] if k == "corrected" else 0)
                t = F.fit_t(vl[(m, s)][k], tv.y)
                for name, tg in subsets.items():
                    z = lt if name == "full" else lt[keep]
                    rows.append({"set": name, "model": m, "seed": s, "kind": k, "T": t, **F.metrics(z, t, tg)})
                r = ref[(ref.model == m) & (ref.seed == s) & (ref.kind == k)].iloc[0]
                assert abs(rows[-2]["macro-F1"] - r["macro-F1"]) < 1e-9 and abs(rows[-2]["softCE(T)"] - r["softCE(T)"]) < 1e-9
                ens_p.setdefault((m, k), {"val": [], "test": []})
                ens_p[(m, k)]["val"].append(softmax(vl[(m, s)][k])); ens_p[(m, k)]["test"].append(softmax(lt))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "s0_dedup_per_seed.csv", index=False)

    ens = {}
    for mk, v in ens_p.items():
        pv, pt = np.mean(v["val"], 0), np.mean(v["test"], 0)
        t = F.fit_t(np.log(pv + 1e-12), tv.y)
        ens[mk] = (pt, np.log(softmax(np.log(pt + 1e-12) / t) + 1e-12), t)

    boot = {}
    for name, tg in subsets.items():
        n = len(tg.y)
        sel = np.arange(len(keep)) if name == "full" else np.flatnonzero(keep)
        rng = np.random.default_rng(F.BOOT_SEED)
        idxs = [rng.integers(0, n, n) for _ in range(F.N_BOOT)]
        full = np.arange(n)
        res = {}
        for cname, a, b in F.COMPARISONS:
            pa, la, _ = ens[a]; pb, lb, _ = ens[b]
            pa, la, pb, lb = pa[sel], la[sel], pb[sel], lb[sel]
            obs = F.boot_stats(pa, la, tg, full) - F.boot_stats(pb, lb, tg, full)
            d = np.array([F.boot_stats(pa, la, tg, i) - F.boot_stats(pb, lb, tg, i) for i in idxs])
            lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
            p = np.minimum(1, 2 * np.minimum((d <= 0).sum(0) + 1, (d >= 0).sum(0) + 1) / (F.N_BOOT + 1))
            res[cname] = {mt: {"diff": float(obs[j]), "ci": [float(lo[j]), float(hi[j])], "p": float(p[j])}
                          for j, mt in enumerate(F.BOOT_METRICS)}
        for mt in F.BOOT_METRICS:
            for cname, ph in zip(res, F.holm([res[c][mt]["p"] for c in res])):
                res[cname][mt]["p_holm"] = float(ph)
        boot[name] = res
    stored = json.loads((ROOT / "runs/final_eval_v2/bootstrap.json").read_text())
    for c in stored:
        for mt in F.BOOT_METRICS:
            assert abs(stored[c][mt]["diff"] - boot["full"][c][mt]["diff"]) < 1e-12
            assert abs(stored[c][mt]["ci"][0] - boot["full"][c][mt]["ci"][0]) < 1e-12
    (OUT / "s0_dedup_bootstrap.json").write_text(json.dumps(boot["dedup"], indent=1))

    md = ["## Step 0b. SENSITIVITY ANALYSIS: final_eval_v2 test summary with confirmed duplicates excluded", "",
          f"Not a new evaluation. The 15 test artworks visually confirmed as the same artwork as a 43K-train artwork "
          f"(1 identical image, 14 different reproductions; duplicates/classification.csv) are removed, leaving "
          f"{int(keep.sum())} test artworks ({int(subsets['dedup'].uniq.sum())} unique-top). None of the pairs "
          f"flagged against the paper 14K train is a duplicate, so R0n's training set has no confirmed duplicate in "
          f"test; the same reduced test set is used for every model so comparisons stay paired. Existing test logits "
          f"only; procedure identical to final_eval_v2 (full-test values reproduce test_per_seed.csv and "
          f"bootstrap.json). The pHash candidate list (Hamming ≤ 4) is not exhaustive, so undetected duplicates may "
          f"remain.", "",
          "### Test, mean ± SD over seeds 42–46 (T fitted on val per seed); deduplicated set, Δ = dedup − full", "",
          "| model | logits | macro-F1 | Δ | acc | balAcc | softCE(T) | Δ | Brier(T) | mean anger+fear AP | Δ | "
          "disgust F1 | neg-sentiment F1 | Δ | anger AUROC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in F.MODELS:
        for k in F.KINDS:
            g = {s: df[(df.set == s) & (df.model == m) & (df.kind == k)] for s in subsets}
            dl = lambda c: f"{(g['dedup'][c].mean() - g['full'][c].mean()):+.4f}"
            md.append(f"| {F.MODELS[m][0]} | {k} | {fmt(g['dedup']['macro-F1'])} | {dl('macro-F1')} | "
                      f"{fmt(g['dedup']['acc'])} | {fmt(g['dedup']['balAcc'])} | {fmt(g['dedup']['softCE(T)'])} | "
                      f"{dl('softCE(T)')} | {fmt(g['dedup']['Brier(T)'])} | {fmt(g['dedup']['mean anger+fear AP'])} | "
                      f"{dl('mean anger+fear AP')} | {fmt(g['dedup']['disgust F1'])} | "
                      f"{fmt(g['dedup']['neg-sentiment F1'])} | {dl('neg-sentiment F1')} | {fmt(g['dedup']['anger AUROC'])} |")
    md += ["", "### Paired bootstrap, deduplicated test (2,000 resamples, rng seed 0); full-test values from "
               "final_eval_v2 for reference", "",
           "| comparison | metric | Δ dedup | 95% CI dedup | p (Holm) dedup | Δ full | 95% CI full | p (Holm) full |",
           "|---|---|---|---|---|---|---|---|"]
    for c in boot["dedup"]:
        for mt in F.BOOT_METRICS:
            d, f = boot["dedup"][c][mt], boot["full"][c][mt]
            md.append(f"| {c} | {mt} | {d['diff']:+.4f} | [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] | {d['p_holm']:.4f} | "
                      f"{f['diff']:+.4f} | [{f['ci'][0]:+.4f}, {f['ci'][1]:+.4f}] | {f['p_holm']:.4f} |")
    md += ["", "### Seed-averaged ensembles, deduplicated test", "",
           "| model | logits | macro-F1 | mean anger+fear AP | neg-sentiment F1 | softCE(T) |", "|---|---|---|---|---|---|"]
    tgd, sel = subsets["dedup"], np.flatnonzero(keep)
    for (m, k), (p, lp, t) in ens.items():
        v = F.boot_stats(p[sel], lp[sel], tgd, np.arange(len(sel)))
        md.append(f"| {F.MODELS[m][0]} | {k} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f} |")
    (OUT / "s0_dedup_sensitivity.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
