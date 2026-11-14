"""Steps 2, 4, 5: per-image votes (+ utterances), app-vs-extraction preprocessing equality, decomposition a-e.

Images: VALIDATION proxies (the suspicious images are not in the app gallery or in val) and synthetic
stimuli built to match the reported paintings (no labels). No holdout or test image is opened."""
import io
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from common_diag import (EMOTIONS, MODELS, NAMES, OUT, ROOT, forward_set, load_set, soft_ce, ensemble, val_df,
                         val_features)
import inference as inf  # app/inference.py

VAL_ROWS = [369, 6, 1430, 1428, 731]
STIM = OUT / "stimuli"; STIM.mkdir(exist_ok=True)


def rothko(size, bg, bands, blur=18, noise=6, seed=0):
    """bands: [(y0, y1, rgb)] in fractions of height; soft edges, canvas grain."""
    w, h = size
    rng = np.random.default_rng(seed)
    a = np.ones((h, w, 3)) * np.asarray(bg, float)
    for y0, y1, c in bands:
        m = int(.07 * w)
        a[int(y0 * h):int(y1 * h), m:w - m] = c
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(im, float) + rng.normal(0, noise, (h, w, 1))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


stimuli = {
    "S1_tall_purple_monochrome": rothko((700, 1600), (74, 36, 98), [(.04, .96, (86, 42, 112))], blur=40),
    "S2_navy_black_bands": rothko((1100, 1500), (12, 12, 18), [(.06, .34, (22, 30, 70)), (.40, .60, (8, 8, 12)),
                                                              (.66, .94, (26, 34, 78))]),
    "S3_maroon_on_black": rothko((1000, 1400), (16, 10, 12), [(.08, .45, (70, 16, 24)), (.55, .92, (40, 10, 16))]),
    "C1_uniform_black": rothko((900, 1200), (14, 14, 14), [], noise=3),
    "C2_uniform_mid_grey": rothko((900, 1200), (128, 128, 128), [], noise=3),
    "C3_uniform_bright_yellow": rothko((900, 1200), (236, 206, 70), [], noise=3),
}
for k, im in stimuli.items():
    im.save(STIM / f"{k}.jpg", quality=92)

v = val_df()
art = pd.read_csv(ROOT / "artemis_official_data/official_data/artemis_dataset_release_v0.csv")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eng = inf.load_engine(device=str(dev))
sets = {"R3_MF1": load_set("R3_MF1", dev), "R3": load_set("R3", dev), "R1": load_set("R1", dev),
        "PROBE": load_set("PROBE", dev)}
proc = eng.processor


def crop_region(w, h, size=336):
    """CLIPImageProcessor: resize shortest edge to 336 (bicubic), centre crop 336x336; box in original pixels."""
    s = size / min(w, h)
    rw, rh = round(w * s), round(h * s)
    left, top = (rw - size) // 2, (rh - size) // 2
    return {"resized": [rw, rh], "crop_in_resized": [left, top, left + size, top + size],
            "crop_in_original": [round(left / s, 1), round(top / s, 1), round((left + size) / s, 1), round((top + size) / s, 1)],
            "fraction_of_long_side_seen": round(min(w, h) / max(w, h), 3)}


def decompose(f):
    out, raws = {}, {}
    for key, ms in sets.items():
        raw, sig = forward_set(ms, f["g"].cpu().numpy(), f["attrs"][None].astype(np.float32),
                               f["t"].cpu().numpy(), f["p"].cpu().numpy(), dev)
        raws[key] = (raw[:, 0], None if sig is None else sig[:, 0])
    for m, c in MODELS.items():
        out[m] = ensemble(raws[c["key"]][0][:, None], c["correct"])[0]
    return out, raws


records = []
for kind, ident in [("val", r) for r in VAL_ROWS] + [("synthetic", k) for k in stimuli]:
    rec = {"kind": kind}
    if kind == "val":
        r = v.iloc[ident]
        path = ROOT / r.image_path
        rec.update(id=int(ident), art_style=r.art_style, painting=r.painting, image_path=r.image_path,
                   n_votes=int(r.n_votes), votes={e: int(r[f"n_{e}"]) for e in EMOTIONS},
                   shares={e: round(float(r[f"p_{e}"]), 4) for e in EMOTIONS})
        u = art[(art.art_style == r.art_style) & (art.painting == r.painting)]
        rec["utterances"] = [{"emotion": e, "utterance": t} for e, t in zip(u.emotion, u.utterance)]
        rec["official_release_vote_count"] = int(len(u))
        y = np.array([r[f"p_{e}"] for e in EMOTIONS])
    else:
        path = STIM / f"{ident}.jpg"
        rec.update(id=ident, image_path=str(path.relative_to(ROOT)))
        y = None
    # app path, exactly as streamlit_app.analyse_bytes: load_image(BytesIO(bytes)) -> predict
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as raw_im:
        rec["file_size_px"] = list(raw_im.size)
    img = inf.load_image(io.BytesIO(data))
    rec["loaded_size_px_after_draft"] = list(img.size)
    rec["crop"] = crop_region(*img.size)
    app = inf.predict(eng, img)
    f = inf.encode(eng, img)
    rec["attributes"] = {n: round(float(x), 5) for n, x in zip(NAMES, f["attrs"])}
    rec["description"] = f["description"]
    probs, raws = decompose(f)
    rec["app_predict_vs_decomposition_a_max_abs_diff"] = float(np.abs(app["probs"] - probs["a_app"]).max())
    rec["app_presence_vs_R3MF1_sigmoid_mean_max_abs_diff"] = float(np.abs(app["presence"] - raws["R3_MF1"][1].mean(0)).max())
    rec["app_top"] = app["top"]
    rec["chips_shown"] = {e: round(float(q), 3) for e, q in zip(EMOTIONS, app["presence"]) if q > .5 and e != app["top"]}
    rec["presence_all"] = {e: round(float(q), 3) for e, q in zip(EMOTIONS, app["presence"])}
    rec["per_seed_top_app"] = [EMOTIONS[int(q.argmax())] for q in app["probs_per_seed"]]
    rec["probs"] = {m: {e: round(float(x), 4) for e, x in zip(EMOTIONS, p)} for m, p in probs.items()}
    rec["top"] = {m: EMOTIONS[int(p.argmax())] for m, p in probs.items()}
    if y is not None:
        rec["softCE"] = {m: round(float(soft_ce(y, p)), 4) for m, p in probs.items()}
    if kind == "val":  # step 4: compare with the saved features_l14 val row and the saved val logits
        g, a, t, p = val_features([ident])
        rec["step4"] = {
            "global_max_abs_diff": float(np.abs(f["g"].cpu().numpy() - g).max()),
            "attr_text_max_abs_diff": float(np.abs(f["t"].cpu().numpy() - t).max()),
            "patches_max_abs_diff": float(np.abs(f["p"].cpu().numpy() - p).max()),
            "attributes_max_abs_diff": float(np.abs(f["attrs"] - a[0]).max()),
            "processor_pixel_shape": list(proc(images=[img], return_tensors="pt")["pixel_values"].shape),
        }
        lo = []
        for key in ("R3_MF1", "R3", "R1", "PROBE"):
            for (run, *_), z in zip(sets[key], raws[key][0]):
                lo.append(float(np.abs(np.load(ROOT / run / "val_logits.npy")[ident] - z).max()))
        rec["step4"]["logits_vs_saved_val_logits_max_abs_diff_all_20_runs"] = max(lo)
        rec["step4"]["app_raw_logits_vs_saved_R3_MF1_max_abs_diff"] = float(max(
            np.abs(np.load(ROOT / f"runs/L14_R3_lam0.3_s{s}/val_logits.npy")[ident] - app["raw_logits"][j]).max()
            for j, s in enumerate(inf.SEEDS)))
    # anger mass bookkeeping
    an = {m: probs[m]["anger"] if isinstance(probs[m], dict) else float(probs[m][4]) for m in probs}
    rec["anger_path"] = {m: round(float(p[4]), 4) for m, p in probs.items()}
    rec["probs"] = {m: {e: round(float(x), 4) for e, x in zip(EMOTIONS, p)} for m, p in probs.items()}
    records.append(rec)
    print(rec["id"], rec.get("painting", ""), "top", rec["top"], "anger", rec["anger_path"], "softCE", rec.get("softCE"),
          "step4", rec.get("step4"), "a-diff", rec["app_predict_vs_decomposition_a_max_abs_diff"], flush=True)

(OUT / "s245_images.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))
