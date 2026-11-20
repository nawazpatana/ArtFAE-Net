"""Precompute the example gallery for gallery mode (APP_MODE=gallery) and verify it against live inference.

run (full environment, with weights and the CLIP cache):
  HF_HUB_OFFLINE=1 python app/precompute_gallery.py

Writes app/assets/precomputed/gallery.json. For each of the 12 examples and both models it stores what the
results page shows: the 5 seed logits (float64; probabilities, R1 prior correction, people counts and
sentiment are derived from them exactly as in full mode), the seed-mean ECSA maps of all 9 emotions, the R3
presence estimates, the 28 attributes and the description, plus the palette and the descriptive aesthetic
properties (app/precompute_visual.py, which can also add those two to an existing file without torch).

Verification: the file is reloaded from disk through gallery.result() and compared with a second, independent
live inference.predict() run (R1, R1 + prior correction, R3). Every array must agree within 1e-5, the
description and top emotion exactly, and the attention overlays of all 9 emotions pixel for pixel. core.load_image
must give the same pixels as artfae_balance.extract_features.load_image. The script exits non-zero otherwise.
"""
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import gallery  # noqa: E402
import inference as inf  # noqa: E402
import precompute_visual  # noqa: E402  (palette + aesthetic properties, torch-free)

TOL = 1e-5
OUT = core.ASSETS / gallery.FILE


def f32(a):
    """float32 values as the shortest decimal that reads back to the same float32."""
    return [float(np.format_float_positional(np.float32(x), unique=True)) for x in np.asarray(a, np.float32).ravel()]


def f64(a):
    return [[float(x) for x in row] for row in np.asarray(a, np.float64)]


def live(engine, data, mode, prior=False):
    return inf.predict(engine, inf.load_image(io.BytesIO(data)), mode=mode, prior_correction=prior)


def build(engine, examples):
    out = {}
    for ex in examples:
        data = Path(ex["path"]).read_bytes()
        rec = {"sha256": gallery.sha256(data)}
        for mode in core.MODES:
            r = live(engine, data, mode)
            rec.setdefault("attrs", f32(r["attrs"]))
            rec.setdefault("description", r["description"])
            assert rec["description"] == r["description"]
            rec[mode] = {"raw_logits": f64(r["raw_logits"]), "maps": f32(r["maps"]),
                         "presence": None if r["presence"] is None else f32(r["presence"]),
                         "seconds": round(float(r["seconds"]), 3)}
        rec.update(precompute_visual.visual_record(core.load_upload(data)))  # as app/precompute_visual.py
        out[ex["file"]] = rec
        print(f"  computed {ex['file']}", flush=True)
    return out


def verify(engine, examples):
    gallery._data.cache_clear(); gallery._delta.cache_clear()
    worst = {k: 0.0 for k in ("probs", "probs_per_seed", "raw_logits", "presence", "maps", "attrs")}
    overlay_pixels = 0
    for ex in examples:
        data = Path(ex["path"]).read_bytes()
        a = np.asarray(core.load_image(io.BytesIO(data)))
        b = np.asarray(inf.load_image(io.BytesIO(data)))
        assert a.shape == b.shape and np.array_equal(a, b), f"core.load_image differs: {ex['file']}"
        img = core.load_image(io.BytesIO(data))
        for mode, prior in (("r1", False), ("r1", True), ("r3", False)):
            want, got = live(engine, data, mode, prior), gallery.result(data, mode, prior)
            assert got["description"] == want["description"] and got["top"] == want["top"], ex["file"]
            assert got["mode"] == want["mode"] and got["prior_correction"] == want["prior_correction"]
            for k in worst:
                if want[k] is None:
                    assert got[k] is None, (ex["file"], mode, k)
                    continue
                worst[k] = max(worst[k], float(np.max(np.abs(np.asarray(got[k], np.float64) -
                                                             np.asarray(want[k], np.float64)))))
            for e in range(9):
                o1 = np.asarray(core.attention_overlay(img, got["maps"][e], .55, size=320))
                o2 = np.asarray(inf.attention_overlay(img, want["maps"][e], .55, size=320))
                overlay_pixels += int(np.count_nonzero(o1 != o2))
    return worst, overlay_pixels


def main():
    engine = inf.load_engine()
    examples = core.load_examples()
    assert len(examples) == 12
    print(f"device {engine.device.type}; computing {len(examples)} examples x {len(core.MODES)} models")
    doc = {"format": gallery.FORMAT, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "emotions": core.EMOTIONS, "seeds": core.SEEDS, "models": {m: f"app/assets/models/{m}/s42-46" for m in core.MODES},
           "clip": f"{inf.CLIP_MODEL}@{inf.CLIP_REVISION}", "device": engine.device.type,
           "examples": build(engine, examples)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    worst, bad_px = verify(engine, examples)
    ok = all(v < TOL for v in worst.values()) and bad_px == 0
    print("max |gallery - live|: " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()) + f"; overlay pixels differing: {bad_px}")
    print(f"{OUT.relative_to(core.ROOT)}: {OUT.stat().st_size:,} bytes; verification {'PASSED' if ok else 'FAILED'} (tol {TOL})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
