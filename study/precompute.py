"""Precompute everything the study shows, so the study itself needs no GPU and no model code.

For each stimulus (and the practice item) it runs the demo's own pipeline (app/inference.py, imported, unchanged):
  R1 and R3 seed-ensemble distributions (raw outputs; R1 without prior correction, as the demo default),
  the "10 viewers" counts (largest remainder) for R1, and the R1 attention overlay for R1's top emotion.
It cross-checks each distribution against the saved pre-registered holdout logits
(runs/final_eval_v3/{R1_R1,R3_R3}_s42-46_holdout_logits.npy, row-aligned with holdout.csv) and stops on a mismatch.

Outputs: study/precomputed/outputs.json, study/precomputed/attention/<id>.jpg, study/precomputed/maps.npz

run (GPU optional):  artfae_demo/.venv/bin/python study/precompute.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(ROOT / "app"))
from common import EMOTIONS, people_counts  # noqa: E402
import inference as inf  # noqa: E402

TOL = 2e-3  # max |p_app - p_saved|; fp32 CLIP on GPU vs the cached-feature evaluation
ALPHA = 0.55  # overlay opacity, the demo's default


def saved_ensemble(prefix, rows):
    ps = []
    for s in inf.SEEDS:
        z = np.load(ROOT / f"runs/final_eval_v3/{prefix}_s{s}_holdout_logits.npy")[rows].astype(np.float64)
        e = np.exp(z - z.max(1, keepdims=True))
        ps.append(e / e.sum(1, keepdims=True))
    return np.mean(ps, 0)


def main():
    assert list(inf.EMOTIONS) == EMOTIONS
    stim = pd.read_csv(STUDY / "stimuli.csv", keep_default_na=False)
    stim = stim[stim.role.isin(["stimulus", "practice"])]
    hold = pd.read_csv(ROOT / "runs/revision_B/fresh_holdout/holdout.csv", usecols=["artwork_id"])
    row = {a: i for i, a in enumerate(hold.artwork_id)}
    rows = [row[a] for a in stim.artwork_id]
    saved = {"r1": saved_ensemble("R1_R1", rows), "r3": saved_ensemble("R3_R3", rows)}
    holdout_paths = pd.read_csv(ROOT / "runs/revision_B/fresh_holdout/holdout.csv",
                                usecols=["artwork_id", "image_path"]).set_index("artwork_id").image_path

    eng = inf.load_engine()
    out_dir = STUDY / "precomputed"
    (out_dir / "attention").mkdir(parents=True, exist_ok=True)
    outputs, maps, worst = {}, {}, 0.0
    for j, (sid, aid) in enumerate(zip(stim.stimulus_id, stim.artwork_id)):
        img = inf.load_image(holdout_paths[aid])  # the original WikiArt file, as in the evaluation
        r1 = inf.predict(eng, img, mode="r1", prior_correction=False)
        r3 = inf.predict(eng, img, mode="r3")
        for k, r in (("r1", r1), ("r3", r3)):
            d = float(np.abs(r["probs"] - saved[k][j]).max())
            worst = max(worst, d)
            if d > TOL:
                sys.exit(f"{sid} {k}: app pipeline differs from the saved holdout logits by {d:.4g}")
        top = r1["top"]
        m = r1["maps"][EMOTIONS.index(top)]
        inf.attention_overlay(img, m, ALPHA).save(out_dir / "attention" / f"{sid}.jpg", "JPEG", quality=88)
        maps[sid] = m
        outputs[sid] = {"r1": [round(float(x), 6) for x in r1["probs"]], "r3": [round(float(x), 6) for x in r3["probs"]],
                        "r1_counts": [int(c) for c in people_counts(r1["probs"])], "r1_top": top, "r3_top": r3["top"],
                        "attention_emotion": top, "attention_image": f"precomputed/attention/{sid}.jpg"}
    meta = {"emotions": EMOTIONS, "models": {"r1": "runs/revision_B/step1/R1_s42-46 (app/assets/models/r1)",
                                             "r3": "runs/revision_B/step1/R3_s42-46 (app/assets/models/r3)"},
            "ensemble": "mean of the five seeds' softmax; raw logits (no prior correction)",
            "attention": f"R1 ECSA map of the R1 top emotion, centre square, overlay opacity {ALPHA}",
            "max_abs_diff_vs_saved_holdout_logits": worst, "device": eng.device.type}
    (out_dir / "outputs.json").write_text(json.dumps({"meta": meta, "stimuli": outputs}, indent=1))
    np.savez_compressed(out_dir / "maps.npz", **maps)
    print(f"{len(outputs)} stimuli; max |p_app - p_saved| = {worst:.2e}; device {eng.device.type}")


if __name__ == "__main__":
    main()
