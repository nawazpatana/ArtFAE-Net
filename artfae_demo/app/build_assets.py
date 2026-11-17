"""Build app/assets/ from the frozen runs (read-only): run once on the training server.

  models/{r1,r3}/s{seed}/  best.pt, normalizer.json, config.json (+ dims.json from the config) of
                    runs/revision_B/step1/{R1,R3}_s{seed} (both softCE-selected), hashes checked against
                    runs/final_selection_v3.txt
  priors.json       val prior (mean val p_*), train prior (mean train p_*), delta = log(val) - log(train)
  presence.json     log(pos_weight) of the training PresenceBCE (undone before the presence sigmoid)
  attr_percentiles.json   1,001 quantiles of each raw attribute over the training split
  examples/         12 val artworks (bytes copied unchanged) + examples.json; never test
  results.json      fresh-holdout results of R1 / R3 (runs/final_eval_v3/per_seed.csv) and the earlier
                    test-split results (runs/final_eval_v2), for the About page

usage: python app/build_assets.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artfae_balance import EMOTIONS  # noqa: E402

ASSETS = ROOT / "app/assets"
# app mode -> (final_selection_v3 model key, run directory pattern); both selected by min val softCE
MODEL_RUNS = {"r1": ("R1", "runs/revision_B/step1/R1_s{s}"), "r3": ("R3", "runs/revision_B/step1/R3_s{s}")}
SEEDS = [42, 43, 44, 45, 46]
VAL, TRAIN, TEST = (ROOT / f"data/artemis_balanced/{s}.csv" for s in ("val", "train", "test"))
# gallery rule (annotator agreement only, model never consulted): the highest-agreement unique-top val
# artwork of every emotion, plus the runner-up for the three negative minority emotions
EXTRA = ["anger", "disgust", "fear"]


def prior(csv):
    return pd.read_csv(csv)[[f"p_{e}" for e in EMOTIONS]].to_numpy().mean(0)


def presence_offsets():
    """presence.json: log(pos_weight) of the training PresenceBCE, so the app can undo it.

    BCE with pos_weight w is minimised at logit = logit(P(present)) + log(w), so the raw presence sigmoid is not
    the probability the app labels it as. pos_weight is rebuilt with the training code itself (no fitting)."""
    import torch
    from artfae_balance.losses import PresenceBCE
    rate = torch.as_tensor(pd.read_csv(TRAIN)[[f"present_{e}" for e in EMOTIONS]].to_numpy(np.float32).mean(0))
    pw = PresenceBCE(rate).pos_weight.double().numpy()  # exactly as in artfae_balance.train
    (ASSETS / "presence.json").write_text(json.dumps({
        "emotions": EMOTIONS, "train_present_rate": rate.double().numpy().tolist(), "pos_weight": pw.tolist(),
        "offset": np.log(pw).tolist(), "formula": "P(present) = sigmoid(presence_logit - log(pos_weight))",
        "source": "artfae_balance.losses.PresenceBCE on data/artemis_balanced/train.csv present_*"}, indent=1))


def main():
    import hashlib
    sel = json.loads((ROOT / "runs/final_selection_v3.txt").read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    for mode, (key, pattern) in MODEL_RUNS.items():
        for s in SEEDS:
            run = pattern.format(s=s)
            assert run in sel["models"][key]["runs"] and sel["models"][key]["checkpoint_selection"] == "min val softCE"
            src, dst = ROOT / run, ASSETS / f"models/{mode}/s{s}"
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("best.pt", "config.json"):
                assert sha(src / f) == sel["checkpoint_sha256"][f"{run}/{f}"], f"{run}/{f} does not match the registry"
            for f in ("best.pt", "normalizer.json", "config.json"):
                shutil.copy2(src / f, dst / f)
            cfg = json.loads((src / "config.json").read_text())
            assert cfg["train_manifest"] == "data/artemis_balanced/train.csv" and cfg["condition"] == key
            (dst / "dims.json").write_text(json.dumps(cfg["dims"]))

    vp, tp = prior(VAL), prior(TRAIN)
    delta = np.log(vp + 1e-8) - np.log(tp + 1e-8)  # identical to scripts/final_eval_v2.py delta()
    (ASSETS / "priors.json").write_text(json.dumps({
        "emotions": EMOTIONS, "val_prior": vp.tolist(), "train_prior": tp.tolist(), "delta": delta.tolist(),
        "formula": "logits + log(val_prior + 1e-8) - log(train_prior + 1e-8)",
        "source": {"val": "data/artemis_balanced/val.csv", "train": "data/artemis_balanced/train.csv"}}, indent=1))
    presence_offsets()

    from artfae.features.aesthetic import NAMES
    attrs = np.load(ROOT / "features_l14/train/attributes.npy")
    levels = np.linspace(0, 1, 1001)
    (ASSETS / "attr_percentiles.json").write_text(json.dumps({
        "names": list(NAMES), "levels": levels.tolist(), "n_train": int(len(attrs)),
        "quantiles": np.quantile(attrs, levels, axis=0).T.round(7).tolist()}))

    v = pd.read_csv(VAL)
    test_paths = set(pd.read_csv(TEST).image_path)
    P = v[[f"p_{e}" for e in EMOTIONS]].to_numpy()
    v["top"], v["agree"] = P.argmax(1), P.max(1)
    ex_dir = ASSETS / "examples"; ex_dir.mkdir(exist_ok=True)
    examples = []
    for i, e in enumerate(EMOTIONS):
        g = v[(v.top == i) & v.unique_top.astype(bool)].sort_values("agree", ascending=False, kind="stable")
        for _, r in g.head(2 if e in EXTRA else 1).iterrows():
            assert r.image_path not in test_paths
            name = Path(r.image_path).name
            shutil.copy2(r.image_path, ex_dir / name)
            title = r.painting.split("_", 1)[-1].replace("-", " ").strip()
            examples.append({"file": name, "artwork_id": r.artwork_id, "painting": title[:1].upper() + title[1:], "artist": name.split("_")[0].replace("-", " ").title(),
                             "style": r.art_style.replace("_", " "), "val_row": int(r.name),
                             "annotator_top": e, "n_votes": int(sum(r[f"n_{k}"] for k in EMOTIONS)),
                             "annotators": {k: float(r[f"p_{k}"]) for k in EMOTIONS}})
    (ex_dir / "examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False))

    per_seed = pd.read_csv(ROOT / "runs/final_eval_v2/test_per_seed.csv")
    rows = [("R0n", "raw", "baseline"), ("B32_R1", "raw", "b32"), ("L14_R1", "raw", "l14_r1"), ("L14_R3", "corrected", "primary")]
    cols = {"macro-F1": "macro_f1", "acc": "acc", "mean anger+fear AP": "anger_fear_ap", "neg-sentiment F1": "neg_f1",
            "F1 anger": "anger_f1"}
    res = {"seeds": SEEDS, "n_test": 2999, "rows": []}
    for m, k, key in rows:
        g = per_seed[(per_seed.model == m) & (per_seed.kind == k)]
        res["rows"].append({"key": key, **{c2: [float(g[c].mean()), float(g[c].std(ddof=1))] for c, c2 in cols.items()}})
    boot = json.loads((ROOT / "runs/final_eval_v2/bootstrap.json").read_text())["(a) L/14 R3c − R0n"]["macro-F1"]
    res["primary_vs_baseline_macro_f1"] = boot
    ens = (ROOT / "runs/final_eval_v2/test_summary.md").read_text().split("Secondary: seed-averaged")[1]
    line = next(l for l in ens.splitlines() if l.startswith("| R3   L/14@336 | corrected |"))
    res["ensemble_macro_f1_primary"] = float(line.split("|")[3])  # secondary 5-seed ensemble table
    res["anger_test_support"] = 14
    hp = pd.read_csv(ROOT / "runs/final_eval_v3/per_seed.csv")
    hcols = {"softCE": "softce", "macro-F1": "macro_f1", "mean anger+fear AP": "anger_fear_ap",
             "neg-sentiment F1": "neg_f1"}
    res["holdout"] = {"n": json.loads((ROOT / "runs/final_eval_v3/results.json").read_text())["primary"]["n"],
                      "source": "runs/final_eval_v3/per_seed.csv (mean ± SD over 5 seeds, raw logits)", "rows": {}}
    for mode, (key, _) in MODEL_RUNS.items():
        g = hp[hp.model == key]
        assert len(g) == 5
        res["holdout"]["rows"][mode] = {c2: [float(g[c].mean()), float(g[c].std(ddof=1))] for c, c2 in hcols.items()}
    (ASSETS / "results.json").write_text(json.dumps(res, indent=1))
    print(f"assets -> {ASSETS}: {len(examples)} examples, delta {np.round(delta, 3).tolist()}")


if __name__ == "__main__":
    main()
