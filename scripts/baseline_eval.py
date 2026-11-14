"""SECOND test evaluation, baselines only: data-effect baseline R0n (paper M1 config on the paper's 14K
train, new features, seeds 42-46) vs R1 (same config on the new 43K train). No model selection.

R1's test logits are the ones already saved by the first test evaluation (runs/final_eval/test).
Output: runs/final_eval/baseline_test/ (refuses to run twice).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_eval as F  # noqa: E402

OUT = F.ROOT / "runs/final_eval/baseline_test"
I_FEAR = F.I_FEAR


def stats(probs, y_soft, present, uniq, idx):
    u = uniq[idx]
    yt, pt = y_soft[idx][u].argmax(1), probs[idx][u].argmax(1)
    cm = np.bincount(yt * 9 + pt, minlength=81).reshape(9, 9)
    tp = np.diag(cm); denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0)
    ap = np.mean([F.average_precision_score(present[idx, c], probs[idx, c]) for c in (F.I_ANGER, I_FEAR)])
    return np.array([f1.mean(), f1[I_FEAR], ap])


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    test_csv = F.ROOT / "data/artemis_balanced/test.csv"
    _, y_val, _, _ = F.load_targets(F.ROOT / "data/artemis_balanced/val.csv")
    _, y, present, uniq = F.load_targets(test_csv)
    rows, probs = [], {"R0n": [], "R1": []}
    for seed in F.SEEDS:
        for name, d, logits in (("R0n", F.ROOT / f"runs/R0n_paper14k_s{seed}", None),
                                ("R1", F.run_dir("R1", seed), np.load(F.OUT / f"test/R1_s{seed}_test_logits.npy"))):
            if logits is None:
                logits = F.test_logits(d)
                np.save(OUT / f"R0n_s{seed}_test_logits.npy", logits)
                out = subprocess.run([sys.executable, "-m", "artfae_balance.evaluate", "--manifest", str(test_csv),
                                      "--logits", str(OUT / f"R0n_s{seed}_test_logits.npy"),
                                      "--out", str(OUT / f"R0n_s{seed}_test_report.json")],
                                     cwd=F.ROOT, capture_output=True, text=True, check=True).stdout
                (OUT / f"R0n_s{seed}_test_report.txt").write_text(out)
                r = json.loads((OUT / f"R0n_s{seed}_test_report.json").read_text())
            else:
                r = json.loads((F.OUT / f"test/R1_s{seed}_test_report.json").read_text())
            t = F.fit_temperature(np.load(d / "val_logits.npy").astype(np.float64), y_val)
            rows.append({"run": name, "seed": seed, **F.metrics_row(r, *F.calibrated(logits.astype(np.float64), y, t), t)})
            probs[name].append(F.softmax(logits.astype(np.float64)))
    pd.DataFrame(rows).to_csv(OUT / "per_seed.csv", index=False)
    table = F.summary_table(rows, "SECOND test evaluation (baselines only): mean ± SD over seeds 42–46")

    avg = {k: np.mean(v, 0) for k, v in probs.items()}
    rng = np.random.default_rng(0)
    full = np.arange(len(y))
    point = stats(avg["R1"], y, present, uniq, full) - stats(avg["R0n"], y, present, uniq, full)
    d = np.empty((2000, 3))
    for i in range(2000):
        idx = rng.integers(0, len(y), len(y))
        d[i] = stats(avg["R1"], y, present, uniq, idx) - stats(avg["R0n"], y, present, uniq, idx)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    names = ["macro-F1", "fear F1", "mean anger+fear AP"]
    bl = ["### Paired bootstrap R1 − R0n (2,000 resamples of test artworks, seed-averaged probabilities)", "",
          "| metric | R0n (seed-avg) | R1 (seed-avg) | Δ R1 − R0n | 95% CI | P(Δ ≤ 0) |", "|---|---|---|---|---|---|"]
    a0, a1 = stats(avg["R0n"], y, present, uniq, full), stats(avg["R1"], y, present, uniq, full)
    for j, n in enumerate(names):
        bl.append(f"| {n} | {a0[j]:.4f} | {a1[j]:.4f} | {point[j]:+.4f} | [{lo[j]:+.4f}, {hi[j]:+.4f}] | "
                  f"{(d[:, j] <= 0).mean():.3f} |")
    (OUT / "bootstrap.json").write_text(json.dumps({n: {"diff": point[j], "ci": [lo[j], hi[j]],
                                                        "p_le_0": float((d[:, j] <= 0).mean())}
                                                    for j, n in enumerate(names)}, indent=1))
    report = table + "\n" + "\n".join(bl) + "\n"
    (OUT / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
