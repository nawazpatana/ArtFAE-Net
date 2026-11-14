"""Steps 1 and 3: configuration of the deployed app, reproduction of every run's saved val logits,
label-order checks and a one-hot test through the display path. Writes s1_config.json, s3_labels.json."""
import ast
import json
import re

import numpy as np
import pandas as pd
import torch

from common_diag import (DELTA, EMOTIONS, OUT, PRI, ROOT, SEL, SEEDS, TRAIN_CSV, VAL_CSV, forward_set, load_run,
                         load_set, sha, val_features)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg_out = {}

# ---------------------------------------------------------------- 1. what the app loads
reg = SEL["checkpoint_sha256"]
rows = []
for s in SEEDS:
    app_d, run = ROOT / f"app/assets/models/s{s}", f"runs/L14_R3_lam0.3_s{s}"
    ck = torch.load(app_d / "best.pt", map_location="cpu", weights_only=False)
    hist = json.loads((ROOT / run / "history.json").read_text())
    hist = hist if isinstance(hist, list) else hist["history"]
    best_mf1 = max(hist, key=lambda h: h["val_macro_f1"])
    best_ce = min(hist, key=lambda h: h["val_soft_ce"])
    rows.append({
        "seed": s, "app_path": str(app_d.relative_to(ROOT) / "best.pt"),
        "sha256": sha(app_d / "best.pt"),
        "registry_entry": f"{run}/best.pt",
        "matches_R3_MF1_registry": sha(app_d / "best.pt") == reg[f"{run}/best.pt"],
        "config_matches_registry": sha(app_d / "config.json") == reg[f"{run}/config.json"],
        "saved_epoch": ck["epoch"], "epoch_max_val_macroF1": best_mf1["epoch"],
        "val_macroF1_at_saved": round(best_mf1["val_macro_f1"], 4),
        "val_softCE_at_saved": round(next(h for h in hist if h["epoch"] == ck["epoch"])["val_soft_ce"], 4),
        "epoch_min_val_softCE": best_ce["epoch"], "min_val_softCE": round(best_ce["val_soft_ce"], 4),
        "n_epochs_run": len(hist),
    })
c42 = json.loads((ROOT / "app/assets/models/s42/config.json").read_text())
cfgs = [json.loads((ROOT / f"app/assets/models/s{s}/config.json").read_text()) for s in SEEDS]
same = all({k: v for k, v in c.items() if k != "seed"} == {k: v for k, v in c42.items() if k != "seed"} for c in cfgs)

# priors: recompute from the manifests
tr, va = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)
P = lambda df: df[[f"p_{e}" for e in EMOTIONS]].to_numpy().mean(0)  # noqa: E731
tp, vp = P(tr), P(va)
delta_re = np.log(vp + 1e-8) - np.log(tp + 1e-8)
# effective class prior the sampler actually shows the model (sampler_weight, with replacement)
w = tr["sampler_weight"].to_numpy(np.float64)
eff = (w[:, None] * tr[[f"p_{e}" for e in EMOTIONS]].to_numpy()).sum(0) / w.sum()
pres_rate = tr[[f"present_{e}" for e in EMOTIONS]].to_numpy(np.float32).mean(0)
pos_weight = np.minimum((1 - pres_rate) / np.maximum(pres_rate, 1e-4), 20.0)

cfg_out["checkpoints"] = rows
cfg_out["selection_rule"] = {"registry_model": "R3_MF1", "registry_label": SEL["models"]["R3_MF1"]["label"],
                             "checkpoint_selection": SEL["models"]["R3_MF1"]["checkpoint_selection"]}
cfg_out["config_identical_across_seeds_except_seed"] = same
cfg_out["config_seed42"] = c42
cfg_out["dims"] = json.loads((ROOT / "app/assets/models/s42/dims.json").read_text())
cfg_out["normalizer_sha256_all_seeds"] = sorted({sha(ROOT / f"app/assets/models/s{s}/normalizer.json") for s in SEEDS})
cfg_out["loss"] = ("soft CE (tau_train 0 -> artfae.training.losses.soft_cross_entropy, no logit adjustment) "
                   "+ 0.3 * PresenceBCE(pos_weight = (1-rate)/rate, clamp 20)")
cfg_out["sampler"] = "WeightedRandomSampler(train.csv sampler_weight = inverse dominant-class frequency, replacement)"
cfg_out["class_weights_in_loss"] = "none (no class-weighted CE); only the presence BCE pos_weight"
cfg_out["presence_pos_weight"] = dict(zip(EMOTIONS, pos_weight.round(3).tolist()))
cfg_out["priors"] = {
    "emotions": PRI["emotions"],
    "train_prior": dict(zip(EMOTIONS, np.round(tp, 5).tolist())),
    "val_prior": dict(zip(EMOTIONS, np.round(vp, 5).tolist())),
    "sampler_effective_prior": dict(zip(EMOTIONS, np.round(eff, 5).tolist())),
    "delta_app": dict(zip(EMOTIONS, np.round(DELTA, 4).tolist())),
    "delta_recomputed_max_abs_diff": float(np.abs(delta_re - DELTA).max()),
    "odds_multiplier_exp_delta": dict(zip(EMOTIONS, np.round(np.exp(DELTA), 3).tolist())),
    "delta_if_sampler_prior_used": dict(zip(EMOTIONS, np.round(np.log(vp) - np.log(eff), 4).tolist())),
}
cfg_out["ensemble"] = ("per seed: softmax(raw_logits + delta); then arithmetic mean of the 5 probability vectors "
                       "(probability averaging, not logit averaging). Presence: mean of the 5 sigmoids.")
cfg_out["clip"] = {"model": "openai/clip-vit-large-patch14-336", "revision": "ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
                   "dtype": "fp32", "processor": "CLIPProcessor use_fast=False", "pool_grid": 7}

# ---------------------------------------------------------------- reproduce every run's saved val logits
g, a, t, p = val_features()
repro = {}
for key in ("R3_MF1", "R3", "R1", "PROBE"):
    ms = load_set(key, dev)
    raw, _ = forward_set(ms, g, a, t, p, dev)
    for (run, *_), z in zip(ms, raw):
        saved = np.load(ROOT / run / "val_logits.npy")
        repro[run] = {"shape": list(saved.shape), "max_abs_diff": float(np.abs(saved - z).max()),
                      "val_logits_sha_matches_registry": sha(ROOT / run / "val_logits.npy") == reg[f"{run}/val_logits.npy"]}
    np.save(OUT / f"val_raw_{key}.npy", raw)
cfg_out["val_logit_reproduction"] = repro
# R3_s45 and L14_R3_lam0.3_s45 share a val_logits hash in the registry; are the weights the same?
a45 = torch.load(ROOT / "runs/L14_R3_lam0.3_s45/best.pt", map_location="cpu", weights_only=False)
b45 = torch.load(ROOT / "runs/revision_B/step1/R3_s45/best.pt", map_location="cpu", weights_only=False)
cfg_out["s45_note"] = {
    "val_logits_identical": bool(np.array_equal(np.load(ROOT / "runs/L14_R3_lam0.3_s45/val_logits.npy"),
                                                np.load(ROOT / "runs/revision_B/step1/R3_s45/val_logits.npy"))),
    "weights_identical": all(torch.equal(a45["model"][k], b45["model"][k]) for k in a45["model"]),
    "mf1_saved_epoch": a45["epoch"],
}
(OUT / "s1_config.json").write_text(json.dumps(cfg_out, indent=1))

# ---------------------------------------------------------------- 3. label order
import i18n  # noqa: E402
import inference as inf  # noqa: E402
import ui  # noqa: E402

lab = {"artfae_balance.EMOTIONS": EMOTIONS, "app inference.EMOTIONS": inf.EMOTIONS,
       "priors.json emotions": PRI["emotions"]}
for s in SEEDS:
    lab[f"app best.pt s{s} emotions"] = torch.load(ROOT / f"app/assets/models/s{s}/best.pt", map_location="cpu",
                                                   weights_only=False)["emotions"]
for key in ("R3", "R1", "PROBE"):
    for run in SEL["models"][key]["runs"]:
        lab[f"{run} emotions"] = load_run(run, "cpu")[2]["emotions"]
lab["i18n EMOTION_NAMES en keys"] = list(i18n.EMOTION_NAMES["en"])
lab["i18n EMOTION_NAMES zh keys"] = list(i18n.EMOTION_NAMES["zh"])
lab["ui PALETTE light keys"] = list(ui.PALETTE["light"])
lab["ui PALETTE dark keys"] = list(ui.PALETTE["dark"])
lab["inference SENTIMENT keys"] = list(inf.SENTIMENT)
lab["train.csv p_* column order"] = [c[2:] for c in tr.columns if c.startswith("p_")]
lab["val.csv p_* column order"] = [c[2:] for c in va.columns if c.startswith("p_")]
# training code reads targets by name: df[[f"p_{e}" for e in EMOTIONS]]
src = (ROOT / "artfae_balance/train.py").read_text()
lab_res = {k: {"order": v, "same_as_EMOTIONS": list(v) == EMOTIONS} for k, v in lab.items()}
lab_res["_train_targets_by_name"] = 'df[[f"p_{e}" for e in EMOTIONS]]' in src

# one-hot and a full permutation test through the display path
disp = []
for m in ("light", "dark"):
    for lang in ("en", "zh"):
        labels = [i18n.emotion(lang, e) for e in EMOTIONS]
        sents = [i18n.sentiment(lang, inf.SENTIMENT[e]) for e in EMOTIONS]
        for c, e in enumerate(EMOTIONS):
            oh = np.zeros(9); oh[c] = 1.0
            fig = ui.distribution_chart(m, EMOTIONS, labels, oh, sents, "p")
            b = fig.data[0]
            j = [i for i, x in enumerate(b.x) if x == 1.0]
            ok = (len(j) == 1 and b.y[j[0]] == labels[c] and b.marker.color[j[0]] == ui.PALETTE[m][e]
                  and b.text[j[0]] == "100.0%" and b.customdata[j[0]] == sents[c])
            top = inf.EMOTIONS[int(oh.argmax())]
            card = ui.result_card(m, top, i18n.emotion(lang, top), float(oh.max()), inf.SENTIMENT[top],
                                  i18n.sentiment(lang, inf.SENTIMENT[top]), "t", "p", "s")
            ok_card = f'>{i18n.emotion(lang, e)}<' in card and ui.PALETTE[m][e] in card and "100.0%" in card
            disp.append({"mode": m, "lang": lang, "emotion": e, "chart_ok": bool(ok), "card_ok": bool(ok_card)})
        # distinct values: every bar must carry its own value
        vals = (np.arange(9) + 1) / 45.0
        b = ui.distribution_chart(m, EMOTIONS, labels, vals, sents, "p").data[0]
        perm_ok = all(abs(x - vals[labels.index(y)]) < 1e-15 for x, y in zip(b.x, b.y))
        colour_ok = all(col == ui.PALETTE[m][EMOTIONS[labels.index(y)]] for col, y in zip(b.marker.color, b.y))
        cc = ui.compare_chart(m, EMOTIONS, labels, [("x", vals)]).data[0]
        cmp_ok = list(cc.x) == labels and np.allclose(cc.y, vals)
        disp.append({"mode": m, "lang": lang, "emotion": "ALL distinct values", "chart_ok": bool(perm_ok and colour_ok),
                     "card_ok": bool(cmp_ok)})
# presence chips: zip(EMOS, r["presence"]) -> chips
pres = np.linspace(.1, .9, 9)
also = sorted([(e, i18n.emotion("en", e), float(q)) for e, q in zip(inf.EMOTIONS, pres) if q > .5 and e != "anger"],
              key=lambda x: -x[2])
chip_html = ui.chips("light", also)
chips_ok = all(f'{i18n.emotion("en", e)}<b>{q * 100:.0f}%</b>' in chip_html for e, _, q in also)
lab_res["_display_tests"] = disp
lab_res["_all_display_tests_pass"] = all(d["chart_ok"] and d["card_ok"] for d in disp) and chips_ok
lab_res["_chips_ok"] = chips_ok
(OUT / "s3_labels.json").write_text(json.dumps(lab_res, indent=1, ensure_ascii=False))

print(json.dumps({"checkpoints": rows, "repro_max": max(v["max_abs_diff"] for v in repro.values()),
                  "repro_bad": [k for k, v in repro.items() if v["max_abs_diff"] > 1e-4 or not v["val_logits_sha_matches_registry"]],
                  "s45": cfg_out["s45_note"], "labels_all_same": all(v["same_as_EMOTIONS"] for k, v in lab_res.items() if not k.startswith("_")),
                  "display_ok": lab_res["_all_display_tests_pass"], "priors": cfg_out["priors"]}, indent=1))
