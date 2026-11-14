"""Demo app tests: smoke, consistency with the evaluated pipeline, i18n coverage, UI in both languages.

  pytest tests/test_inference.py -v
The consistency check needs the research data (val manifest, images, saved val logits) and is skipped
in the laptop bundle, where those are not shipped.
"""
import hashlib
import io
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
import i18n  # noqa: E402
import inference as inf  # noqa: E402
from artfae.features.aesthetic import NAMES, describe  # noqa: E402

VAL = ROOT / "data/artemis_balanced/val.csv"
RUNS = ROOT / "runs"
HAVE_DATA = VAL.exists() and (RUNS / "L14_R3_lam0.3_s42/val_logits.npy").exists()


@pytest.fixture(scope="module")
def engine():
    return inf.load_engine()


def all_descriptions():
    out = set()
    i = {n: k for k, n in enumerate(NAMES)}
    for lum, std, (warm, cool) in itertools.product([.1, .5, .9], [.05, .2, .4], [(.6, .1), (.1, .6), (.3, .3)]):
        a = np.zeros(len(NAMES), np.float32)
        a[i["luminance"]], a[i["luminance_std"]], a[i["warm"]], a[i["cool"]] = lum, std, warm, cool
        out.add(describe(a))
    return out


def test_description_table_is_complete():
    d = all_descriptions()
    assert len(d) == 27
    assert d == set(i18n.DESCRIPTIONS_ZH)
    for zh in i18n.DESCRIPTIONS_ZH.values():
        assert not re.search(r"[A-Za-z]", zh) and zh.endswith("。")


def test_i18n_parity():
    assert set(i18n.STRINGS["en"]) == set(i18n.STRINGS["zh"])
    for table in (i18n.EMOTION_NAMES, i18n.SENTIMENT_NAMES, i18n.ATTRIBUTE_NAMES):
        assert set(table["en"]) == set(table["zh"])
    assert set(i18n.ATTRIBUTE_NAMES["en"]) == set(NAMES)
    assert [i18n.EMOTION_NAMES["zh"][e] for e in inf.EMOTIONS] == \
        ["趣味", "敬畏", "滿足", "興奮", "憤怒", "厭惡", "恐懼", "悲傷", "其他"]
    assert [i18n.SENTIMENT_NAMES["zh"][s] for s in ("positive", "negative", "other")] == ["正面", "負面", "其他"]
    for lang in i18n.LANGS:  # every template formats with the arguments the app passes
        for k, s in i18n.STRINGS[lang].items():
            assert s.count("{") == s.count("}"), k


def test_smoke(engine):
    ex = inf.load_examples()
    assert len(ex) == 12 and {e["annotator_top"] for e in ex} == set(inf.EMOTIONS)
    r = inf.predict_path(engine, ex[0]["path"])
    assert r["probs"].shape == (9,) and abs(r["probs"].sum() - 1) < 1e-9
    assert r["probs_per_seed"].shape == (5, 9) and r["raw_logits"].shape == (5, 9)
    assert r["maps"].shape == (9, 7, 7) and np.allclose(r["maps"].reshape(9, -1).sum(1), 1, atol=1e-5)
    assert ((r["presence"] > 0) & (r["presence"] < 1)).all()
    assert r["description"] in i18n.DESCRIPTIONS_ZH and r["top"] in inf.EMOTIONS
    p = inf.percentiles(r["attrs"], inf.load_json("attr_percentiles.json"))
    assert p.shape == (28,) and ((p >= 0) & (p <= 100)).all()
    from PIL import Image
    ov = inf.attention_overlay(inf.load_image(ex[0]["path"]), r["maps"][0], .5, size=200)
    assert ov.size == (200, 200)
    buf = io.BytesIO(); Image.new("RGBA", (300, 200), (10, 120, 200, 128)).save(buf, "PNG")
    assert inf.predict(engine, inf.load_image(io.BytesIO(buf.getvalue())))["probs"].shape == (9,)  # PNG upload


@pytest.mark.skipif(not HAVE_DATA, reason="research data not shipped in the bundle")
def test_consistency_with_saved_val_logits(engine):
    """20 val images through the app pipeline reproduce every seed's saved val logits.

    Criterion: max |diff| < 1e-4 on CUDA, the device the cached features and logits were computed on.
    On CPU, CLIP's vision tower runs in true fp32 where the GPU cache used cuDNN's default TF32
    convolution; the tiny differences cross float16 rounding boundaries in a few % of the stored patch
    values and give ~3e-4. (With the cached features the model itself reproduces to ~2e-7 on CPU.)
    So CPU is held to 1e-3 and identical per-seed top classes, reported separately.
    """
    v = pd.read_csv(VAL)
    saved = {s: np.load(RUNS / f"L14_R3_lam0.3_s{s}/val_logits.npy") for s in inf.SEEDS}
    diffs, pdiff, same_top = np.zeros((20, 5)), 0.0, True
    for i in range(20):
        r = inf.predict_path(engine, v.image_path[i])
        ref = []
        for k, s in enumerate(inf.SEEDS):
            diffs[i, k] = np.abs(r["raw_logits"][k] - saved[s][i]).max()
            same_top &= int(r["raw_logits"][k].argmax()) == int(saved[s][i].argmax())
            c = saved[s][i].astype(np.float64) + engine.delta
            ref.append(np.exp(c - c.max()) / np.exp(c - c.max()).sum())
        pdiff = max(pdiff, np.abs(r["probs"] - np.mean(ref, 0)).max())
    print(f"\nconsistency on {engine.device.type}: max |logit diff| = {diffs.max():.2e} "
          f"(per seed {np.round(diffs.max(0), 7).tolist()}); max |ensemble prob diff| = {pdiff:.2e}; "
          f"same top class for every seed: {same_top}")
    assert same_top
    assert diffs.max() < (1e-4 if engine.device.type == "cuda" else 1e-3)


@pytest.mark.skipif(not HAVE_DATA, reason="research data not shipped in the bundle")
def test_bundle_matches_frozen_runs():
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    for s in inf.SEEDS:
        for f in ("best.pt", "normalizer.json", "dims.json", "config.json"):
            assert sha(APP / f"assets/models/s{s}/{f}") == sha(RUNS / f"L14_R3_lam0.3_s{s}/{f}")
    pri = inf.load_json("priors.json")
    cols = [f"p_{e}" for e in inf.EMOTIONS]
    vp = pd.read_csv(VAL)[cols].to_numpy().mean(0)
    tp = pd.read_csv(ROOT / "data/artemis_balanced/train.csv")[cols].to_numpy().mean(0)
    assert np.allclose(pri["delta"], np.log(vp + 1e-8) - np.log(tp + 1e-8), atol=1e-12)
    test_paths = set(pd.read_csv(ROOT / "data/artemis_balanced/test.csv").image_path.map(lambda p: Path(p).name))
    assert not {e["file"] for e in inf.load_examples()} & test_paths


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_app_ui(lang):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=600)
    at.run()
    assert not at.exception
    at.sidebar.radio(key="lang").set_value(lang).run()
    at.button(key="ex_10").click().run()  # sadness example
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    names = i18n.EMOTION_NAMES[lang]
    assert "af-card af-result" in html and any(f'af-top">{n}<' in html for n in names.values())
    assert i18n.t(lang, "also_feel") in html and i18n.t(lang, "sentiment_summary") in html
    assert [t.label for t in at.tabs] == [i18n.t(lang, k) for k in ("tab_attention", "tab_aesthetics", "tab_details")]
    assert any(i18n.t(lang, "attention_caption") == c.value for c in at.caption)
    assert any(i18n.description(lang, d) in html for d in i18n.DESCRIPTIONS_ZH)
    if lang == "zh":
        assert not any(k in html for k in i18n.DESCRIPTIONS_ZH)  # no untranslated description
    at.selectbox(key=at.selectbox[0].key).set_value("fear").run()
    at.slider(key="att_alpha").set_value(0.8).run()
    assert not at.exception
    at.sidebar.radio(key="page").set_value("compare").run()
    at.multiselect(key="cmp_examples").set_value([0, 4, 6]).run()
    assert not at.exception
    assert sum("af-card af-result" in m.value for m in at.markdown) == 3
    at.sidebar.radio(key="page").set_value("about").run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "0.385" in html and "0.325" in html and "Nawazish Hussain" in html
