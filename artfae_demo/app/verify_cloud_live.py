"""Verify the cloud_live engine (CLIP vision tower + precomputed text embeddings, CPU) against the full pipeline.

run on the server (full environment, validation images available):
  HF_HUB_OFFLINE=1 python app/verify_cloud_live.py [--n 20]

For n validation images (evenly spaced rows of data/artemis_balanced/val.csv, gallery examples excluded), every
model setting (R1, R1 + prior correction, R3) is predicted by
  live      inference.load_engine(device="cpu", vision_only=True)       what Streamlit Cloud runs
  full_cpu  inference.load_engine(device="cpu")                         the full pipeline, same device
  full_gpu  inference.load_engine() on CUDA (if available)              the server's usual device
Pass: max |probability difference| < 1e-5 against both, identical descriptions, attributes and top emotions,
and attention maps identical (bit for bit) to the full pipeline on the same device. Differences to the GPU run
are reported (float32 GPU vs CPU arithmetic differs in the last bits). Writes app/assets/cloud_live_check.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import inference as inf  # noqa: E402

TOL = 1e-5
SETTINGS = (("r1", False), ("r1", True), ("r3", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    import torch
    v = pd.read_csv(core.ROOT / "data/artemis_balanced/val.csv")
    gallery_rows = {e["val_row"] for e in core.load_examples()}
    rows = [int(i) for i in np.linspace(0, len(v) - 1, a.n + len(gallery_rows)).round() if int(i) not in gallery_rows][:a.n]
    engines = {"live": inf.load_engine(device="cpu", vision_only=True), "full_cpu": inf.load_engine(device="cpu")}
    if torch.cuda.is_available():
        engines["full_gpu"] = inf.load_engine(device="cuda")
    worst = {k: {"probs": 0.0, "maps": 0.0, "presence": 0.0, "raw_logits": 0.0} for k in engines if k != "live"}
    maps_identical_cpu = True
    for r in rows:
        img = inf.load_image(core.ROOT / v.image_path[r])
        for mode, prior in SETTINGS:
            out = {k: inf.predict(e, img, mode=mode, prior_correction=prior) for k, e in engines.items()}
            live = out["live"]
            for k in worst:
                ref = out[k]
                assert ref["description"] == live["description"] and ref["top"] == live["top"], (r, mode, k)
                assert np.array_equal(ref["attrs"], live["attrs"]), (r, k)
                for f in worst[k]:
                    if ref[f] is not None:
                        worst[k][f] = max(worst[k][f], float(np.abs(np.asarray(ref[f], np.float64) -
                                                                    np.asarray(live[f], np.float64)).max()))
            maps_identical_cpu &= bool(np.array_equal(out["full_cpu"]["maps"], live["maps"]))
        print(f"  row {r}: ok", flush=True)
    ok = all(w["probs"] < TOL for w in worst.values()) and maps_identical_cpu
    rep = {"n_images": len(rows), "val_rows": rows, "settings": [f"{m}{'+prior' if p else ''}" for m, p in SETTINGS],
           "tolerance": TOL, "max_abs_diff": worst, "maps_identical_to_full_cpu": maps_identical_cpu, "passed": ok,
           "torch": torch.__version__}
    (core.ASSETS / "cloud_live_check.json").write_text(json.dumps(rep, indent=1) + "\n")
    for k, w in worst.items():
        print(f"live vs {k}: " + ", ".join(f"{f} {x:.2e}" for f, x in w.items()))
    print(f"attention maps identical to full_cpu: {maps_identical_cpu}; verification {'PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
