"""Step 2 tables: L/14 R1 (seeds 42-44) trained without the held-out artworks, evaluated on the unseen-artist
holdout and the seen-artist control holdout (both carved from the 43K train set; no val or test).

Metrics: softCE and KL (T = 1, all artworks), macro-F1 (unique-top artworks, 9 labels), mean anger+fear presence
AP and 9-class macro presence AP (all artworks). Gap = unseen − seen.
CI: cluster bootstrap by artist, 2,000 resamples (rng seed 0). In each resample, artists are drawn with
replacement independently within each holdout; the metric is computed for each seed and averaged over seeds;
gap = unseen − seen. 95% percentile interval.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import EMOTIONS, EPS, OUT, ROOT, evaluate, fmt, softmax

SEEDS = [42, 43, 44]
N_BOOT = 2000
A, FE = EMOTIONS.index("anger"), EMOTIONS.index("fear")


def stats(p, y, present, uniq):
    ce = -(y * np.log(p + EPS)).sum(1).mean()
    kl = (y * (np.log(y + EPS) - np.log(p + EPS))).sum(1).mean()
    yt, pt = y[uniq].argmax(1), p[uniq].argmax(1)
    cm = np.bincount(yt * 9 + pt, minlength=81).reshape(9, 9)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.where(den > 0, 2 * np.diag(cm) / np.maximum(den, 1), 0).mean()
    ap = [average_precision_score(present[:, c], p[:, c]) if present[:, c].any() else np.nan for c in range(9)]
    return np.array([ce, kl, f1, (ap[A] + ap[FE]) / 2, np.nanmean(ap)])


NAMES = ["softCE", "KL", "macro-F1", "mean anger+fear AP", "macro AP (9)"]


def main():
    sp = pd.read_csv(OUT / "step2/split.csv")
    tr = pd.read_csv(ROOT / "data/artemis_balanced/train.csv")
    H = {}
    for k in ("holdout_unseen", "holdout_seen"):
        m = (sp.role == k).to_numpy()
        df = tr[m].reset_index(drop=True)
        y = df[[f"p_{e}" for e in EMOTIONS]].to_numpy(float)
        H[k] = {"y": y, "present": df[[f"present_{e}" for e in EMOTIONS]].to_numpy(int),
                "uniq": df.unique_top.astype(bool).to_numpy(), "artist": sp.artist[m].to_numpy(),
                "p": {s: softmax(np.load(OUT / f"step2/R1_s{s}/{k}_logits.npy").astype(np.float64)) for s in SEEDS}}
    per = []
    for k, h in H.items():
        for s in SEEDS:
            per.append({"holdout": k, "seed": s, **dict(zip(NAMES, stats(h["p"][s], h["y"], h["present"], h["uniq"])))})
    per = pd.DataFrame(per)
    per.to_csv(OUT / "s2_per_seed.csv", index=False)

    rng = np.random.default_rng(0)
    groups = {k: [np.flatnonzero(h["artist"] == a) for a in np.unique(h["artist"])] for k, h in H.items()}
    boot = {k: np.empty((N_BOOT, len(NAMES))) for k in H}
    for b in range(N_BOOT):
        for k, h in H.items():
            g = groups[k]
            idx = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
            boot[k][b] = np.mean([stats(h["p"][s][idx], h["y"][idx], h["present"][idx], h["uniq"][idx])
                                  for s in SEEDS], 0)
    gap_b = boot["holdout_unseen"] - boot["holdout_seen"]
    point = {k: per[per.holdout == k][NAMES].mean().to_numpy() for k in H}
    gap = point["holdout_unseen"] - point["holdout_seen"]
    lo, hi = np.percentile(gap_b, [2.5, 97.5], 0)

    sp_tr = sp[sp.role == "train"]
    info = []
    for k, h in H.items():
        ent = -(h["y"] * np.log(h["y"] + EPS)).sum(1)
        info.append(f"| {k} | {len(h['y']):,} | {len(np.unique(h['artist']))} | {int(h['uniq'].sum()):,} | "
                    f"{ent.mean():.4f} | " + " ".join(f"{v:.3f}" for v in h["y"].mean(0)) + " |")
    md = ["## Step 2. Artist-held-out experiment (L/14 R1, seeds 42–44; no val or test used)", "",
          "Split (runs/revision_B/step2/split.csv, code/s2_split.py): 10% of the 43K-train artists, sampled within "
          "artwork-count strata, form the unseen-artist holdout (all their artworks). A seen-artist control of the "
          "same size is drawn at random from the other artists' artworks (each artist keeps ≥ 1 training "
          "artwork). A 5% selection split of the remaining artworks replaces val for early stopping / checkpoint "
          f"selection (min softCE). Training: {len(sp_tr):,} artworks, {sp_tr.artist.nunique()} artists. Recipe "
          "otherwise identical to Step 1 R1.", "",
          "### Holdout composition", "",
          "| holdout | artworks | artists | unique-top | mean vote entropy | mean vote share (amu awe con exc ang dis fea sad som) |",
          "|---|---|---|---|---|---|", *info, "",
          "### Metrics per holdout (mean ± SD over 3 seeds) and gap (unseen − seen) with artist-cluster bootstrap 95% CI", "",
          "| metric | unseen artists | seen-artist control | gap (unseen − seen) | 95% CI |", "|---|---|---|---|---|"]
    for j, n in enumerate(NAMES):
        md.append(f"| {n} | {fmt(per[per.holdout == 'holdout_unseen'][n])} | "
                  f"{fmt(per[per.holdout == 'holdout_seen'][n])} | {gap[j]:+.4f} | [{lo[j]:+.4f}, {hi[j]:+.4f}] |")
    md += ["", "Per-seed gaps: " + "; ".join(
        f"s{s}: " + ", ".join(f"{n} {per[(per.holdout == 'holdout_unseen') & (per.seed == s)][n].iloc[0] - per[(per.holdout == 'holdout_seen') & (per.seed == s)][n].iloc[0]:+.4f}"
                              for n in NAMES[:3]) for s in SEEDS) + "."]
    pd.DataFrame({"metric": NAMES, "gap": gap, "ci_lo": lo, "ci_hi": hi}).to_csv(OUT / "s2_gap.csv", index=False)
    (OUT / "s2_artist_holdout.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
