"""Evaluation with minority-class metrics (Steps 6-7). NumPy only.

Usage:
  python -m artfae_balance.evaluate --manifest data/artemis_balanced/test.csv \
      --probs runs/m1_balanced/test_probs.npy --out runs/m1_balanced/test_report.json

Post-hoc logit adjustment (tune on val, apply once on test):
  python -m artfae_balance.evaluate --manifest .../val.csv --logits val_logits.npy \
      --train-manifest .../train.csv --tune-tau
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

try:
    from . import EMOTIONS
except ImportError:
    from artfae_balance import EMOTIONS


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_targets(manifest):
    df = pd.read_csv(manifest)
    y_soft = df[[f"p_{e}" for e in EMOTIONS]].to_numpy(float)
    present = df[[f"present_{e}" for e in EMOTIONS]].to_numpy(int)
    uniq = df["unique_top"].astype(bool).to_numpy()
    return df, y_soft, present, uniq


def evaluate(probs, y_soft, present, uniq):
    eps = 1e-12
    r = {}
    # distribution metrics on ALL artworks
    r["soft_ce"] = float(-(y_soft * np.log(probs + eps)).sum(1).mean())
    r["kl"] = float((y_soft * (np.log(y_soft + eps) - np.log(probs + eps))).sum(1).mean())
    r["brier"] = float(((probs - y_soft) ** 2).sum(1).mean())
    # hard metrics on unique-top artworks
    yt, pt = y_soft[uniq].argmax(1), probs[uniq].argmax(1)
    labels = list(range(len(EMOTIONS)))
    r["n_unique_top"] = int(uniq.sum())
    r["accuracy"] = float((yt == pt).mean())
    r["macro_f1"] = float(f1_score(yt, pt, labels=labels, average="macro", zero_division=0))
    r["balanced_accuracy"] = float(balanced_accuracy_score(yt, pt))
    p, rc, f, s = precision_recall_fscore_support(yt, pt, labels=labels, zero_division=0)
    top2 = np.argsort(-probs[uniq], 1)[:, :2]
    r["per_class"] = {}
    for i, e in enumerate(EMOTIONS):
        m = yt == i
        r["per_class"][e] = {
            "support": int(s[i]), "precision": float(p[i]), "recall": float(rc[i]), "f1": float(f[i]),
            "top2_recall": float((top2[m] == i).any(1).mean()) if m.any() else None,
            "n_predicted": int((pt == i).sum()),
            # detection: does the model rank artworks where ANY annotator chose e higher?
            "present_pos": int(present[:, i].sum()),
            "present_auroc": float(roc_auc_score(present[:, i], probs[:, i]))
                             if 0 < present[:, i].sum() < len(present) else None,
            "present_ap": float(average_precision_score(present[:, i], probs[:, i]))
                          if present[:, i].sum() > 0 else None,
            "prior_ap": float(present[:, i].mean()),   # AP of a random ranker, for reference
        }
    r["confusion_matrix"] = confusion_matrix(yt, pt, labels=labels).tolist()
    return r


def tune_tau(logits, y_soft, uniq, log_prior, grid=np.arange(0, 2.01, 0.1)):
    """Post-hoc: probs = softmax(logits - tau*log_prior); pick tau by val macro-F1."""
    yt = y_soft[uniq].argmax(1)
    best = (None, -1)
    for t in grid:
        pt = (logits - t * log_prior)[uniq].argmax(1)
        f = f1_score(yt, pt, labels=list(range(len(EMOTIONS))), average="macro", zero_division=0)
        if f > best[1]:
            best = (float(t), float(f))
    return best


def print_report(r):
    print(f"unique-top n={r['n_unique_top']}  acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  "
          f"balAcc={r['balanced_accuracy']:.4f}")
    print(f"softCE={r['soft_ce']:.4f}  KL={r['kl']:.4f}  Brier={r['brier']:.4f}")
    print(f"{'emotion':15s}{'sup':>5s}{'P':>7s}{'R':>7s}{'F1':>7s}{'top2R':>7s}{'#pred':>6s}"
          f"{'AUROC':>7s}{'AP':>7s}{'chance':>7s}")
    for e, c in r["per_class"].items():
        fmt = lambda v: f"{v:7.3f}" if v is not None else "      -"
        print(f"{e:15s}{c['support']:5d}{fmt(c['precision'])}{fmt(c['recall'])}{fmt(c['f1'])}"
              f"{fmt(c['top2_recall'])}{c['n_predicted']:6d}{fmt(c['present_auroc'])}"
              f"{fmt(c['present_ap'])}{fmt(c['prior_ap'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probs")
    g.add_argument("--logits")
    ap.add_argument("--train-manifest", help="needed for --tune-tau / --tau")
    ap.add_argument("--tune-tau", action="store_true")
    ap.add_argument("--tau", type=float, default=0.0, help="apply a fixed post-hoc tau (from val)")
    ap.add_argument("--out")
    a = ap.parse_args()

    _, y_soft, present, uniq = load_targets(a.manifest)
    if a.probs:
        probs = np.load(a.probs)
    else:
        logits = np.load(a.logits)
        assert len(logits) == len(y_soft), "predictions and manifest rows differ"
        if a.tune_tau or a.tau:
            tr = pd.read_csv(a.train_manifest)
            log_prior = np.log(tr[[f"p_{e}" for e in EMOTIONS]].to_numpy().mean(0) + 1e-8)
            if a.tune_tau:
                tau, f = tune_tau(logits, y_soft, uniq, log_prior)
                print(f"[tune] best tau={tau:.1f} (val macro-F1 {f:.4f}); use --tau {tau:.1f} on test")
                a.tau = tau
            logits = logits - a.tau * log_prior
        probs = softmax(logits)
    assert len(probs) == len(y_soft), "predictions and manifest rows differ"
    r = evaluate(probs, y_soft, present, uniq)
    r["post_hoc_tau"] = a.tau
    print_report(r)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
