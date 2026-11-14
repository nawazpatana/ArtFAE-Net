"""Step 6 (val group analysis, models a-e) and step 7 (presence head: calibration of the chips) on VALIDATION.
Uses the raw val logits reproduced in s1 (val_raw_*.npy, identical to each run's saved val_logits.npy)."""
import json

import numpy as np
import pandas as pd
import torch

from common_diag import (EMOTIONS, MODELS, NAMES, OUT, forward_set, load_set, soft_ce, ensemble, val_df,
                         val_features)

v = val_df()
Y = v[[f"p_{e}" for e in EMOTIONS]].to_numpy()
PRES = v[[f"present_{e}" for e in EMOTIONS]].to_numpy()
N = v[[f"n_{e}" for e in EMOTIONS]].to_numpy()
A = np.load(OUT.parent.parent / "features_l14/val/attributes.npy")
col = lambda k: A[:, list(NAMES).index(k)]  # noqa: E731
AN = EMOTIONS.index("anger")

raw = {k: np.load(OUT / f"val_raw_{k}.npy") for k in ("R3_MF1", "R3", "R1", "PROBE")}
probs = {m: ensemble(raw[c["key"]], c["correct"]) for m, c in MODELS.items()}
np.savez(OUT / "val_probs_a_to_e.npz", **probs)

lum, sat, edge, lstd = col("luminance"), col("saturation"), col("edge_density"), col("luminance_std")
q = lambda x, f: np.quantile(x, f)  # noqa: E731  (val quantiles)
styles = v.art_style.isin(["Color_Field_Painting", "Minimalism", "Abstract_Expressionism"]).to_numpy()
dark_or_flat = (lum < q(lum, .25)) | (sat < q(sat, .25))
uniform = (edge < q(edge, .25)) & (lstd < q(lstd, .25))
groups = {
    "G1 abstract styles (CFP+Min+AbEx), all": styles,
    "G2 abstract styles AND (lum<P25 or sat<P25)  [requested group]": styles & dark_or_flat,
    "G3 any style, dark (lum<P25) AND near-uniform (edge<P25, lum_std<P25)": (lum < q(lum, .25)) & uniform,
    "G4 any style, dark (lum<P25) AND low saturation (sat<P25)": (lum < q(lum, .25)) & (sat < q(sat, .25)),
    "G5 any style, near-uniform (edge<P25, lum_std<P25)": uniform,
    "ALL val": np.ones(len(v), bool),
}
thresholds = {k: float(q(x, .25)) for k, x in
              {"luminance": lum, "saturation": sat, "edge_density": edge, "luminance_std": lstd}.items()}
mx = Y.max(1, keepdims=True)
human_anger_plural_unique = (np.isclose(Y, mx, atol=1e-5).sum(1) == 1) & np.isclose(Y[:, AN], mx[:, 0], atol=1e-5)
human_anger_plural_any = np.isclose(Y[:, AN], mx[:, 0], atol=1e-5)

res = {"thresholds_val_P25": thresholds, "groups": {}}
for gname, m in groups.items():
    r = {"n": int(m.sum()), "human_mean_anger_share": float(Y[m, AN].mean()),
         "human_anger_present_rate": float(PRES[m, AN].mean()),
         "human_anger_unique_plurality_rate": float(human_anger_plural_unique[m].mean()),
         "human_anger_plurality_incl_ties_rate": float(human_anger_plural_any[m].mean()), "models": {}}
    for k, p in probs.items():
        r["models"][k] = {"mean_pred_anger": float(p[m, AN].mean()),
                          "anger_top1_rate": float((p[m].argmax(1) == AN).mean()),
                          "softCE": float(soft_ce(Y[m], p[m]).mean()),
                          "mean_pred_something_else": float(p[m, -1].mean())}
    res["groups"][gname] = r
    res["groups"][gname]["indices"] = np.where(m)[0].tolist() if m.sum() <= 60 else "omitted"

# where does the deployed app say anger on val, and what did humans say there?
pa = probs["a_app"]
top_a = pa.argmax(1) == AN
res["app_anger_top1_on_val"] = {
    "n": int(top_a.sum()), "human_mean_anger_share_there": float(Y[top_a, AN].mean()),
    "human_anger_present_rate_there": float(PRES[top_a, AN].mean()),
    "human_anger_unique_plurality_there": float(human_anger_plural_unique[top_a].mean()),
    "n_without_correction_b": int((probs["b_mf1_raw"].argmax(1) == AN).sum()),
    "n_r3_softce_c": int((probs["c_r3_ce"].argmax(1) == AN).sum()),
    "n_r1_d": int((probs["d_r1"].argmax(1) == AN).sum()),
    "n_probe_e": int((probs["e_probe"].argmax(1) == AN).sum()),
    "n_human_unique_plurality_anger": int(human_anger_plural_unique.sum()),
    "lum_of_app_anger_top1_median": float(np.median(lum[top_a])), "lum_val_median": float(np.median(lum)),
    "rows": np.where(top_a)[0].tolist(),
}
# correlation of predicted anger with darkness, per model (does the model tie anger to low luminance?)
res["spearman_pred_anger_vs_luminance"] = {}
spear = lambda x, y: float(pd.Series(x).rank().corr(pd.Series(y).rank()))  # noqa: E731
res["spearman_human_anger_vs_luminance"] = spear(Y[:, AN], lum)
for k, p in probs.items():
    res["spearman_pred_anger_vs_luminance"][k] = spear(p[:, AN], lum)
# anger mass by luminance decile
dec = pd.qcut(lum, 10, labels=False)
res["anger_by_luminance_decile"] = {
    "human": [float(Y[dec == d, AN].mean()) for d in range(10)],
    **{k: [float(p[dec == d, AN].mean()) for d in range(10)] for k, p in probs.items()}}

# ---------------------------------------------------------------- step 7: presence head
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g, a, t, pp = val_features()
_, sig = forward_set(load_set("R3_MF1", dev), g, a, t, pp, dev)
pres_mean = sig.mean(0)
np.save(OUT / "val_presence_R3_MF1.npy", sig)
pres = {"present_equals_any_vote": bool(((N > 0).astype(int) == PRES).all()),
        "per_class": {}}
for c, e in enumerate(EMOTIONS):
    s = pres_mean[:, c]
    shown = s > .5
    pres["per_class"][e] = {
        "empirical_present_rate": float(PRES[:, c].mean()), "mean_sigmoid": float(s.mean()),
        "rate_sigmoid_gt_0.5": float(shown.mean()),
        "empirical_present_rate_when_shown": float(PRES[shown, c].mean()) if shown.any() else None,
        "mean_sigmoid_when_shown": float(s[shown].mean()) if shown.any() else None,
        "mean_n_votes_when_shown": float(v.n_votes[shown].mean()) if shown.any() else None,
    }
# reliability by bins, pooled over classes
bins = np.linspace(0, 1, 11)
S, T = pres_mean.ravel(), PRES.ravel()
pres["reliability_pooled"] = [{"bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}", "n": int(((S >= bins[i]) & (S < bins[i + 1])).sum()),
                               "mean_sigmoid": float(S[(S >= bins[i]) & (S < bins[i + 1])].mean()) if ((S >= bins[i]) & (S < bins[i + 1])).any() else None,
                               "empirical_rate": float(T[(S >= bins[i]) & (S < bins[i + 1])].mean()) if ((S >= bins[i]) & (S < bins[i + 1])).any() else None}
                              for i in range(10)]
pres["n_votes_distribution_val"] = v.n_votes.describe().round(2).to_dict()
res["presence"] = pres
(OUT / "s6_groups.json").write_text(json.dumps(res, indent=1))

# compact print
for gname, r in res["groups"].items():
    print(f"\n{gname}  n={r['n']}  human anger share {r['human_mean_anger_share']:.4f}  "
          f"uniq-plural {r['human_anger_unique_plurality_rate']:.3f}  plural+ties {r['human_anger_plurality_incl_ties_rate']:.3f}")
    for k, mm in r["models"].items():
        print(f"   {k:10s} pred anger {mm['mean_pred_anger']:.4f}  anger top1 {mm['anger_top1_rate']:.3f}  softCE {mm['softCE']:.4f}")
print(json.dumps({k: v for k, v in res["app_anger_top1_on_val"].items() if k != "rows"}, indent=1))
print("spearman", res["spearman_human_anger_vs_luminance"], res["spearman_pred_anger_vs_luminance"])
print("decile", {k: np.round(v, 4).tolist() for k, v in res["anger_by_luminance_decile"].items()})
print(json.dumps(pres["per_class"], indent=0))
print(json.dumps(pres["reliability_pooled"]))
print("G2 idx", res["groups"]["G2 abstract styles AND (lum<P25 or sat<P25)  [requested group]"]["indices"])
