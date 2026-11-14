"""Task 1: reference baselines on val and test (constant predictors, no fitting on val/test)."""
import json

import numpy as np
import pandas as pd

from common import EMOTIONS, OUT, TEST_CSV, TRAIN_14K, TRAIN_43K, VAL_CSV, F, dist_metrics, evaluate, md_table


def train_mean(csv):
    return pd.read_csv(csv)[[f"p_{e}" for e in EMOTIONS]].to_numpy(float).mean(0)


def main():
    maj = np.eye(9)[EMOTIONS.index("contentment")]
    preds = {
        "majority class (contentment, one-hot)": maj,
        "mean train vote distribution, 43K": train_mean(TRAIN_43K),
        "mean train vote distribution, paper 14K": train_mean(TRAIN_14K),
    }
    rows, priors = [], {k: dict(zip(EMOTIONS, np.round(v, 6).tolist())) for k, v in preds.items()}
    for split, csv in (("val", VAL_CSV), ("test", TEST_CSV)):
        tg = F.Targets(csv)
        for name, q in preds.items():
            p = np.tile(q, (len(tg.y), 1))
            r = evaluate(p, tg.y, tg.present, tg.uniq)
            rows.append({"split": split, "baseline": name, "n": len(tg.y), "n unique-top": int(tg.uniq.sum()),
                         "acc": r["accuracy"], "macro-F1": r["macro_f1"], "balAcc": r["balanced_accuracy"],
                         **dist_metrics(p, tg.y)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_baselines.csv", index=False)
    (OUT / "t1_train_priors.json").write_text(json.dumps(priors, indent=1))
    md = ["## Task 1. Reference baselines (constant predictors)", "",
          "Hard metrics on unique-top artworks; softCE, KL, Brier on all artworks, eps = 1e-12 as in "
          "artfae_balance.evaluate. The one-hot majority predictor assigns probability 0 to eight classes, "
          "so its softCE and KL are dominated by log(1e-12) ≈ −27.6 and are not meaningful as calibration "
          "numbers; they are reported only for completeness.", "",
          md_table(df), "", "Mean training vote distributions (class order " + ", ".join(EMOTIONS) + "):", ""]
    md += [f"- {k}: " + ", ".join(f"{v:.4f}" for v in preds[k]) for k in list(preds)[1:]]
    (OUT / "t1_baselines.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
