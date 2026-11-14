"""Anger analysis from annotations and the saved test logits only (no model is run).

Writes runs/final_eval/analysis/anger_analysis.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artfae_balance import EMOTIONS  # noqa: E402
from artfae_balance.evaluate import softmax  # noqa: E402

T = ROOT / "runs/final_eval/test"
OUT = ROOT / "runs/final_eval/analysis"
A = EMOTIONS.index("anger")
SEEDS = [42, 43, 44, 45, 46]
SENT = {"positive": ["amusement", "awe", "contentment", "excitement"],
        "negative": ["anger", "disgust", "fear", "sadness"], "something_else": ["something_else"]}


def probs(name):
    if name == "R0":
        return [softmax(np.load(T / "R0_test_logits.npy").astype(float))]
    return [softmax(np.load(T / f"{name}_s{s}_test_logits.npy").astype(float)) for s in SEEDS]


def votes_section(df, label):
    n = df[[f"n_{e}" for e in EMOTIONS]].to_numpy()
    top = n.max(1, keepdims=True); is_top = n == top
    uniq_anger = (is_top.sum(1) == 1) & (n.argmax(1) == A)
    tie_anger = (is_top.sum(1) > 1) & is_top[:, A]
    any_a = n[:, A] >= 1
    share = df["p_anger"].to_numpy()
    lines = [f"#### {label} ({len(df):,} artworks)", "",
             f"- anger is the unique plurality: **{uniq_anger.sum()}** ({uniq_anger.mean():.2%}); "
             f"anger tied for the top: {tie_anger.sum()}",
             f"- artworks with ≥1 anger vote: {any_a.sum():,} ({any_a.mean():.1%}); mean anger share among them: "
             f"**{share[any_a].mean():.3f}** (median {np.median(share[any_a]):.3f}, mean votes per artwork "
             f"{n[any_a].sum(1).mean():.1f})",
             f"- artworks with ≥2 anger votes: {(n[:, A] >= 2).sum():,}", ""]
    m = n[:, A] >= 2
    winner = np.where(is_top[m].sum(1) == 1, np.array(EMOTIONS)[n[m].argmax(1)], "tie")
    vc = pd.Series(winner).value_counts().reindex(EMOTIONS + ["tie"], fill_value=0)
    # co-vote table: mean share of each emotion on artworks with >=2 anger votes
    co = df.loc[m, [f"p_{e}" for e in EMOTIONS]].mean()
    lines += [f"Artworks with ≥2 anger votes (n={m.sum()}): which emotion wins, and the mean vote share of each "
              "emotion on those artworks", "", "| emotion | wins (unique top) | share of wins | mean vote share |",
              "|---|---|---|---|"]
    for e in EMOTIONS + ["tie"]:
        lines.append(f"| {e} | {vc[e]} | {vc[e] / m.sum():.1%} | " + (f"{co[f'p_{e}']:.3f}" if e != "tie" else "–") + " |")
    return lines + [""]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(ROOT / "data/artemis_balanced/train.csv")
    te = pd.read_csv(ROOT / "data/artemis_balanced/test.csv")
    L = ["# Anger analysis (annotations + saved test logits only)", "", "## 1. Anger in the votes", ""]
    L += votes_section(tr, "New 43K train") + votes_section(te, "Test")

    y = te[[f"p_{e}" for e in EMOTIONS]].to_numpy()
    uniq = te["unique_top"].astype(bool).to_numpy()
    present = te[[f"present_{e}" for e in EMOTIONS]].to_numpy()
    P = {k: probs(k) for k in ("R0", "R1", "R4b")}
    avg = {k: np.mean(v, 0) for k, v in P.items()}

    # 2. the unique-top anger test cases
    case = uniq & (y.argmax(1) == A)
    L += [f"## 2. The {case.sum()} unique-top anger test artworks", "",
          "Rank of anger in the seed-averaged predicted distribution (R0: single checkpoint) and the "
          "predicted class instead.", "",
          "| artwork | anger votes / total | anger share | 2nd emotion (votes) | R0 rank | R1 rank | R1 predicts | R4b rank | R4b predicts | R4b p(anger) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    n = te[[f"n_{e}" for e in EMOTIONS]].to_numpy()
    rank = {k: (np.argsort(-avg[k], 1) == A).argmax(1) + 1 for k in avg}
    for i in np.where(case)[0]:
        o = np.argsort(-n[i]); second = EMOTIONS[o[1]]
        L.append(f"| {te.painting[i]} | {n[i, A]} / {n[i].sum()} | {y[i, A]:.2f} | {second} ({n[i, o[1]]}) | "
                 f"{rank['R0'][i]} | {rank['R1'][i]} | {EMOTIONS[avg['R1'][i].argmax()]} | {rank['R4b'][i]} | "
                 f"{EMOTIONS[avg['R4b'][i].argmax()]} | {avg['R4b'][i, A]:.3f} |")
    L += ["", "| model | anger ranked 1st | 2nd | 3rd | lower | predicted instead (counts) |", "|---|---|---|---|---|---|"]
    for k in ("R0", "R1", "R4b"):
        r = rank[k][case]
        inst = pd.Series([EMOTIONS[j] for j in avg[k][case].argmax(1)]).value_counts()
        L.append(f"| {k} | {(r == 1).sum()} | {(r == 2).sum()} | {(r == 3).sum()} | {(r > 3).sum()} | "
                 + ", ".join(f"{e} {c}" for e, c in inst.items()) + " |")

    # 3. anger presence ranking
    pos = present[:, A]
    L += ["", f"## 3. Anger presence detection on test ({pos.sum()} artworks with ≥1 anger vote; "
          f"chance AP = {pos.mean():.3f})", "",
          "| model | AUROC (seed-averaged) | AP (seed-averaged) | AUROC per seed, mean ± SD | AP per seed, mean ± SD |",
          "|---|---|---|---|---|"]
    for k in ("R0", "R1", "R4b"):
        au = [roc_auc_score(pos, p[:, A]) for p in P[k]]; ap = [average_precision_score(pos, p[:, A]) for p in P[k]]
        sd = lambda v: f"{np.mean(v):.4f}" + (f" ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else "")
        L.append(f"| {k} | {roc_auc_score(pos, avg[k][:, A]):.4f} | {average_precision_score(pos, avg[k][:, A]):.4f} "
                 f"| {sd(au)} | {sd(ap)} |")

    # 4. sentiment level
    idx = {s: [EMOTIONS.index(e) for e in es] for s, es in SENT.items()}
    ys = np.stack([y[:, v].sum(1) for v in idx.values()], 1)
    top = ys.max(1, keepdims=True); su = np.isclose(ys, top, atol=1e-5).sum(1) == 1
    yt = ys[su].argmax(1)
    L += ["", "## 4. Sentiment level (positive = amusement+awe+contentment+excitement; negative = "
          "anger+disgust+fear+sadness; something_else)", "",
          f"Targets are the summed vote shares; hard metrics on the {su.sum():,} artworks with a unique top sentiment "
          f"(of {len(te):,}). Predictions sum the model's class probabilities per sentiment. Per-seed mean ± SD; "
          "R0 is a single checkpoint.", "",
          "| model | macro-F1 | accuracy | F1 positive | F1 negative | F1 something_else | soft CE (3-way) |",
          "|---|---|---|---|---|---|---|"]
    for k in ("R0", "R1", "R4b"):
        rows = []
        for p in P[k]:
            ps = np.stack([p[:, v].sum(1) for v in idx.values()], 1)
            pt = ps[su].argmax(1)
            f = f1_score(yt, pt, labels=[0, 1, 2], average=None, zero_division=0)
            rows.append([f.mean(), (pt == yt).mean(), *f, -(ys * np.log(ps + 1e-12)).sum(1).mean()])
        rows = np.array(rows)
        cell = lambda j: f"{rows[:, j].mean():.4f}" + (f" ± {rows[:, j].std(ddof=1):.4f}" if len(rows) > 1 else "")
        L.append(f"| {k} | " + " | ".join(cell(j) for j in range(6)) + " |")
    L.append(f"\nTest sentiment support (unique-top): " + ", ".join(
        f"{s} {int((yt == j).sum())}" for j, s in enumerate(SENT)))
    (OUT / "anger_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
