"""Runs tests/test_inference.py without pytest (not installed in the app venv) + before/after presence check."""
import io, json, sys, types, traceback
from pathlib import Path
import numpy as np
import transformers  # noqa: F401  (import before the pytest stub)
ROOT = Path(__file__).resolve().parents[3]
sys.modules["pytest"] = types.SimpleNamespace(fixture=lambda **k: (lambda f: f),
                                              mark=types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f), skipif=lambda *a, **k: (lambda f: f)))
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "runs/app_diagnosis/code")]
import test_inference as T  # noqa: E402
from common_diag import EMOTIONS  # noqa: E402
eng = T.inf.load_engine()
for name, call in [("test_description_table_is_complete", T.test_description_table_is_complete),
                   ("test_i18n_parity", T.test_i18n_parity), ("test_smoke", lambda: T.test_smoke(eng)),
                   ("test_consistency_with_saved_val_logits", lambda: T.test_consistency_with_saved_val_logits(eng)),
                   ("test_bundle_matches_frozen_runs", T.test_bundle_matches_frozen_runs),
                   ("test_app_ui[en]", lambda: T.test_app_ui("en")), ("test_app_ui[zh]", lambda: T.test_app_ui("zh"))]:
    try:
        call(); print("PASS", name, flush=True)
    except Exception:
        print("FAIL", name); traceback.print_exc()
R = json.load(open(ROOT / "runs/app_diagnosis/s245_images.json"))
for r in R:
    out = T.inf.predict(eng, T.inf.load_image(io.BytesIO((ROOT / r["image_path"]).read_bytes())))
    before = np.array([r["presence_all"][e] for e in EMOTIONS])
    chips_after = {e: round(float(q), 3) for e, q in zip(EMOTIONS, out["presence"]) if q > .5 and e != out["top"]}
    pa = np.array([r["probs"]["a_app"][e] for e in EMOTIONS])
    print(f"{str(r['id'])[:28]:28s} top {out['top']} (unchanged {out['top'] == r['app_top']}, probs max diff {np.abs(out['probs'] - pa).max():.1e})")
    print(f"    anger presence {before[4]:.3f} -> {out['presence'][4]:.3f}")
    print(f"    chips before {r['chips_shown']}\n    chips after  {chips_after}")
