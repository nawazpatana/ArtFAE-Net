"""Task 3: 9x9 confusion matrices (per seed and seed-averaged), prediction counts, per-class F1 with
95% bootstrap CIs. Unique-top test artworks; rows = human top emotion, columns = predicted argmax.

Bootstrap: 2,000 resamples of test artworks drawn exactly as in final_eval_v2 (rng seed 0,
rng.integers(0, n, n) over all 2,999 artworks, then restricted to unique-top). In each resample the
per-class F1 is computed for every seed and averaged over seeds; 95% percentile interval.
"""
import json

import numpy as np
import pandas as pd

from common import EMOTIONS, MODELS, OUT, SEEDS, Data, softmax

N_BOOT, BOOT_SEED = 2000, 0
SHORT = ["amu", "awe", "con", "exc", "ang", "dis", "fea", "sad", "som"]


def cm_of(yt, pt):
    return np.bincount(yt * 9 + pt, minlength=81).reshape(9, 9)


def f1_from_cm(cm):
    tp, den = np.diag(cm), cm.sum(0) + cm.sum(1)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def cm_md(cm, title, nd=0):
    lines = [f"#### {title}", "", "| true \\ pred | " + " | ".join(SHORT) + " | support |",
             "|---|" + "---|" * 10]
    for i, e in enumerate(EMOTIONS):
        lines.append(f"| {e} | " + " | ".join(f"{v:.{nd}f}" for v in cm[i]) + f" | {cm[i].sum():.{nd}f} |")
    lines.append("| **#pred** | " + " | ".join(f"{v:.{nd}f}" for v in cm.sum(0)) + f" | {cm.sum():.{nd}f} |")
    return "\n".join(lines)


def main():
    d = Data()
    tt = d.tt
    n = len(tt.y)
    yt_all = tt.y.argmax(1)
    rng = np.random.default_rng(BOOT_SEED)
    idxs = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    out_json, md_cm, pc_rows, cnt_rows = {}, [], [], []
    for lab in MODELS:
        preds = {s: softmax(d.logits(lab, s)).argmax(1) for s in SEEDS}
        cms = {s: cm_of(yt_all[tt.uniq], preds[s][tt.uniq]) for s in SEEDS}
        mean_cm = np.mean([cms[s] for s in SEEDS], 0)
        out_json[lab] = {"per_seed": {str(s): cms[s].tolist() for s in SEEDS}, "seed_mean": mean_cm.tolist()}
        f1_seed = np.array([f1_from_cm(cms[s]) for s in SEEDS])  # (5, 9)
        boot = np.empty((N_BOOT, 9))
        for b, idx in enumerate(idxs):
            u = idx[tt.uniq[idx]]
            boot[b] = np.mean([f1_from_cm(cm_of(yt_all[u], preds[s][u])) for s in SEEDS], 0)
        lo, hi = np.percentile(boot, [2.5, 97.5], 0)
        for i, e in enumerate(EMOTIONS):
            pc_rows.append({"model": lab, "emotion": e, "support": int(round(mean_cm[i].sum())),
                            "F1 mean over seeds": f1_seed[:, i].mean(), "F1 SD over seeds": f1_seed[:, i].std(ddof=1),
                            "95% CI lo": lo[i], "95% CI hi": hi[i]})
            for s in SEEDS:
                cnt_rows.append({"model": lab, "seed": s, "emotion": e,
                                 "n_pred unique-top": int(cms[s][:, i].sum()),
                                 "n_pred all test": int((preds[s] == i).sum())})
        md_cm.append(f"### {lab}\n\n" + cm_md(mean_cm, "seed-averaged (mean of the 5 per-seed count matrices)", 1))
        for s in SEEDS:
            md_cm.append(cm_md(cms[s], f"{lab}, seed {s}"))
    (OUT / "t3_confusion_matrices.json").write_text(json.dumps(
        {"rows": "human top emotion (unique-top test artworks)", "cols": "predicted argmax",
         "class_order": EMOTIONS, "models": out_json}, indent=1))
    pc, cnt = pd.DataFrame(pc_rows), pd.DataFrame(cnt_rows)
    pc.to_csv(OUT / "t3_per_class_f1_ci.csv", index=False)
    cnt.to_csv(OUT / "t3_prediction_counts_per_seed.csv", index=False)
    (OUT / "t3_confusion_matrices.md").write_text(
        "## Task 3. Confusion matrices, unique-top test artworks (n = 2,246)\n\n"
        "Rows: human top emotion. Columns: predicted argmax. Last row: number of predictions per class.\n\n"
        + "\n\n".join(md_cm) + "\n")

    md = ["## Task 3. Per-class F1 (unique-top test) with 95% bootstrap CIs", "",
          "Point estimate: mean (± SD) of per-seed F1 over seeds 42–46. CI: 2,000 paired artwork resamples "
          "(rng seed 0, same draws as final_eval_v2), per-class F1 averaged over seeds within each resample, "
          "95% percentile interval.", "",
          "| emotion | support | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for e in EMOTIONS:
        cells = []
        for lab in MODELS:
            r = pc[(pc.model == lab) & (pc.emotion == e)].iloc[0]
            cells.append(f"{r['F1 mean over seeds']:.3f} ± {r['F1 SD over seeds']:.3f} "
                         f"[{r['95% CI lo']:.3f}, {r['95% CI hi']:.3f}]")
        md.append(f"| {e} | {int(pc[pc.emotion == e].support.iloc[0])} | " + " | ".join(cells) + " |")
    md += ["", "### Prediction counts per class, mean ± SD over seeds (unique-top test, n = 2,246; "
               "true support in the first column)", "",
           "| emotion | true support | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for e in EMOTIONS:
        sup = int(pc[pc.emotion == e].support.iloc[0])
        cells = []
        for lab in MODELS:
            v = cnt[(cnt.model == lab) & (cnt.emotion == e)]["n_pred unique-top"]
            cells.append(f"{v.mean():.1f} ± {v.std(ddof=1):.1f}")
        md.append(f"| {e} | {sup} | " + " | ".join(cells) + " |")
    md += ["", "### Prediction counts per class on all 2,999 test artworks (incl. tied), mean ± SD over seeds", "",
           "| emotion | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for e in EMOTIONS:
        md.append(f"| {e} | " + " | ".join(
            f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" for v in
            (cnt[(cnt.model == lab) & (cnt.emotion == e)]["n_pred all test"] for lab in MODELS)) + " |")
    (OUT / "t3_per_class.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
