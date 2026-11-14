"""Task 8: equivalence test (TOST, margin ±0.01 macro-F1) for comparison (d) L/14 R3c − L/14 R3.

Uses the existing bootstrap of final_eval_v2 exactly: seed-averaged probabilities, 2,000 paired
resamples of test artworks drawn with rng seed 0, macro-F1 on unique-top artworks via
final_eval_v2.boot_stats. The regenerated observed difference and 95% CI are checked against
runs/final_eval_v2/bootstrap.json before the TOST is computed.

TOST at alpha = 0.05: equivalence iff the 90% percentile CI lies inside (−0.01, +0.01).
Bootstrap p-values: p_low = (#{d* <= −0.01} + 1)/(B + 1) for H0: Δ <= −0.01,
p_high = (#{d* >= +0.01} + 1)/(B + 1) for H0: Δ >= +0.01; p_TOST = max(p_low, p_high).
Secondary: paired t-TOST on the 5 per-seed differences (seed-level variability, n = 5).
"""
import json

import numpy as np
from scipy import stats

from common import OUT, ROOT, SEEDS, Data, F, softmax

MARGIN = 0.01
NAME = "(d) L/14 R3c − L/14 R3"


def main():
    d = Data()
    tv, tt = d.tv, d.tt
    ens = {}
    for lab in ("L/14 R3c", "L/14 R3"):
        pv = np.mean([softmax(d.logits(lab, s, "val")) for s in SEEDS], 0)
        pt = np.mean([softmax(d.logits(lab, s)) for s in SEEDS], 0)
        t = F.fit_t(np.log(pv + 1e-12), tv.y)
        ens[lab] = (pt, np.log(softmax(np.log(pt + 1e-12) / t) + 1e-12))
    rng = np.random.default_rng(F.BOOT_SEED)
    n = len(tt.y)
    idxs = [rng.integers(0, n, n) for _ in range(F.N_BOOT)]
    (pa, la), (pb, lb) = ens["L/14 R3c"], ens["L/14 R3"]
    full = np.arange(n)
    obs = F.boot_stats(pa, la, tt, full) - F.boot_stats(pb, lb, tt, full)
    dist = np.array([F.boot_stats(pa, la, tt, i) - F.boot_stats(pb, lb, tt, i) for i in idxs])

    ref = json.loads((ROOT / "runs/final_eval_v2/bootstrap.json").read_text())[NAME]
    for j, mt in enumerate(F.BOOT_METRICS):
        lo, hi = np.percentile(dist[:, j], [2.5, 97.5])
        assert abs(obs[j] - ref[mt]["diff"]) < 1e-12 and abs(lo - ref[mt]["ci"][0]) < 1e-12 \
            and abs(hi - ref[mt]["ci"][1]) < 1e-12, mt
    print("integrity: regenerated bootstrap reproduces bootstrap.json (diff and 95% CI, all 4 metrics)")

    dm = dist[:, F.BOOT_METRICS.index("macro-F1")]
    o = float(obs[F.BOOT_METRICS.index("macro-F1")])
    B = len(dm)
    ci90, ci95 = np.percentile(dm, [5, 95]), np.percentile(dm, [2.5, 97.5])
    p_low = (np.sum(dm <= -MARGIN) + 1) / (B + 1)
    p_high = (np.sum(dm >= MARGIN) + 1) / (B + 1)
    boot = {"observed_diff": o, "ci90": ci90.tolist(), "ci95": ci95.tolist(), "p_lower": float(p_low),
            "p_upper": float(p_high), "p_tost": float(max(p_low, p_high)),
            "equivalent_at_0.05": bool(ci90[0] > -MARGIN and ci90[1] < MARGIN),
            "frac_boot_inside_margin": float(np.mean(np.abs(dm) < MARGIN)),
            "frac_boot_below_minus_margin": float(np.mean(dm <= -MARGIN))}

    # secondary: per-seed paired differences of macro-F1 (unique-top), t-TOST with df = 4
    per = []
    for s in SEEDS:
        f = [F.metrics(d.logits(lab, s), 1.0, tt)["macro-F1"] for lab in ("L/14 R3c", "L/14 R3")]
        per.append(f[0] - f[1])
    per = np.array(per)
    m, se, df_ = per.mean(), per.std(ddof=1) / np.sqrt(len(per)), len(per) - 1
    t_low, t_high = (m + MARGIN) / se, (m - MARGIN) / se
    seed = {"per_seed_diff": dict(zip(map(str, SEEDS), per.round(6).tolist())), "mean": float(m), "se": float(se),
            "ci90": [float(m - stats.t.ppf(0.95, df_) * se), float(m + stats.t.ppf(0.95, df_) * se)],
            "p_lower": float(1 - stats.t.cdf(t_low, df_)), "p_upper": float(stats.t.cdf(t_high, df_))}
    seed["p_tost"] = max(seed["p_lower"], seed["p_upper"])
    seed["equivalent_at_0.05"] = bool(seed["ci90"][0] > -MARGIN and seed["ci90"][1] < MARGIN)
    res = {"comparison": NAME, "metric": "macro-F1 (unique-top test)", "margin": MARGIN, "alpha": 0.05,
           "bootstrap_tost_primary": boot, "seed_paired_t_tost_secondary": seed}
    (OUT / "t8_tost.json").write_text(json.dumps(res, indent=1))

    md = ["## Task 8. Equivalence test (TOST), comparison (d) L/14 R3c − L/14 R3, macro-F1, margin ±0.01", "",
          "Primary: the existing final_eval_v2 bootstrap (seed-averaged probabilities, 2,000 paired artwork "
          "resamples, rng seed 0), regenerated and checked to reproduce bootstrap.json exactly. Equivalence at "
          "α = 0.05 requires the 90% CI to lie inside (−0.01, +0.01).", "",
          "| analysis | Δ macro-F1 | 90% CI | 95% CI | p (H0: Δ ≤ −0.01) | p (H0: Δ ≥ +0.01) | p TOST | equivalent? |",
          "|---|---|---|---|---|---|---|---|",
          f"| bootstrap, 5-seed ensemble (primary) | {o:+.4f} | [{ci90[0]:+.4f}, {ci90[1]:+.4f}] | "
          f"[{ci95[0]:+.4f}, {ci95[1]:+.4f}] | {p_low:.4f} | {p_high:.4f} | {max(p_low, p_high):.4f} | "
          f"{'yes' if boot['equivalent_at_0.05'] else 'no'} |",
          f"| paired t over 5 seeds (secondary, df = 4) | {m:+.4f} | [{seed['ci90'][0]:+.4f}, {seed['ci90'][1]:+.4f}] | – | "
          f"{seed['p_lower']:.4f} | {seed['p_upper']:.4f} | {seed['p_tost']:.4f} | "
          f"{'yes' if seed['equivalent_at_0.05'] else 'no'} |", "",
          f"Share of bootstrap differences inside ±0.01: {boot['frac_boot_inside_margin']:.3f}; at or below −0.01: "
          f"{boot['frac_boot_below_minus_margin']:.3f}. Per-seed differences: "
          + ", ".join(f"s{k} {v:+.4f}" for k, v in seed["per_seed_diff"].items()) + "."]
    (OUT / "t8_tost.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
