"""Demo app tests: smoke, consistency with the evaluated pipeline, i18n coverage, UI in both languages.

  pytest tests/test_inference.py -v
The consistency check needs the research data (val manifest, images, saved val logits) and is skipped
in the laptop bundle, where those are not shipped.
"""
import hashlib
from html import escape as html_escape
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
RUN_DIRS = {"r1": "revision_B/step1/R1_s{s}", "r3": "revision_B/step1/R3_s{s}"}  # app mode -> frozen runs
HAVE_DATA = VAL.exists() and all((RUNS / d.format(s=42) / "val_logits.npy").exists() for d in RUN_DIRS.values())


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
    for mode in inf.MODES:
        r = inf.predict_path(engine, ex[0]["path"], mode=mode)
        assert r["mode"] == mode and not r["prior_correction"]
        assert r["probs"].shape == (9,) and abs(r["probs"].sum() - 1) < 1e-9
        assert r["probs_per_seed"].shape == (5, 9) and r["raw_logits"].shape == (5, 9)
        assert r["maps"].shape == (9, 7, 7) and np.allclose(r["maps"].reshape(9, -1).sum(1), 1, atol=1e-5)
        if mode == "r3":
            assert r["presence"].shape == (9,) and ((r["presence"] > 0) & (r["presence"] < 1)).all()
        else:
            assert r["presence"] is None  # R1 has no presence head
        assert r["description"] in i18n.DESCRIPTIONS_ZH and r["top"] in inf.EMOTIONS
    assert inf.predict_path(engine, ex[0]["path"])["mode"] == "r1"  # default
    rc = inf.predict_path(engine, ex[0]["path"], mode="r1", prior_correction=True)
    assert rc["prior_correction"] and not np.allclose(rc["probs"], inf.predict_path(engine, ex[0]["path"])["probs"])
    with pytest.raises(ValueError):
        inf.predict_path(engine, ex[0]["path"], mode="r3", prior_correction=True)  # no correction on the R3 path
    p = inf.percentiles(r["attrs"], inf.load_json("attr_percentiles.json"))
    assert p.shape == (28,) and ((p >= 0) & (p <= 100)).all()
    from PIL import Image
    ov = inf.attention_overlay(inf.load_image(ex[0]["path"]), r["maps"][0], .5, size=200)
    assert ov.size == (200, 200)
    buf = io.BytesIO(); Image.new("RGBA", (300, 200), (10, 120, 200, 128)).save(buf, "PNG")
    assert inf.predict(engine, inf.load_image(io.BytesIO(buf.getvalue())))["probs"].shape == (9,)  # PNG upload


@pytest.mark.skipif(not HAVE_DATA, reason="research data not shipped in the bundle")
@pytest.mark.parametrize("mode,prior", [("r1", False), ("r1", True), ("r3", False)])
def test_consistency_with_saved_val_logits(engine, mode, prior):
    """20 val images through the app pipeline reproduce every seed's saved val logits.

    Criterion: max |diff| < 1e-4 on CUDA, the device the cached features and logits were computed on.
    On CPU, CLIP's vision tower runs in true fp32 where the GPU cache used cuDNN's default TF32
    convolution; the tiny differences cross float16 rounding boundaries in a few % of the stored patch
    values and give ~3e-4. (With the cached features the model itself reproduces to ~2e-7 on CPU.)
    So CPU is held to 1e-3 and identical per-seed top classes, reported separately.
    """
    v = pd.read_csv(VAL)
    saved = {s: np.load(RUNS / RUN_DIRS[mode].format(s=s) / "val_logits.npy") for s in inf.SEEDS}
    diffs, pdiff, same_top = np.zeros((20, 5)), 0.0, True
    for i in range(20):
        r = inf.predict_path(engine, v.image_path[i], mode=mode, prior_correction=prior)
        ref = []
        for k, s in enumerate(inf.SEEDS):
            diffs[i, k] = np.abs(r["raw_logits"][k] - saved[s][i]).max()
            same_top &= int(r["raw_logits"][k].argmax()) == int(saved[s][i].argmax())
            c = saved[s][i].astype(np.float64) + (engine.delta if prior else 0.0)
            ref.append(np.exp(c - c.max()) / np.exp(c - c.max()).sum())
        pdiff = max(pdiff, np.abs(r["probs"] - np.mean(ref, 0)).max())
    print(f"\nconsistency [{mode}, prior correction {prior}] on {engine.device.type}: max |logit diff| = {diffs.max():.2e} "
          f"(per seed {np.round(diffs.max(0), 7).tolist()}); max |ensemble prob diff| = {pdiff:.2e}; "
          f"same top class for every seed: {same_top}")
    assert same_top
    assert diffs.max() < (1e-4 if engine.device.type == "cuda" else 1e-3)


@pytest.mark.skipif(not HAVE_DATA, reason="research data not shipped in the bundle")
def test_bundle_matches_frozen_runs():
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    import json
    import torch
    from artfae_balance.losses import PresenceBCE
    sel = json.loads((RUNS / "final_selection_v3.txt").read_text())
    for mode, key in (("r1", "R1"), ("r3", "R3")):
        assert sel["models"][key]["checkpoint_selection"] == "min val softCE"
        for s in inf.SEEDS:
            run = RUN_DIRS[mode].format(s=s)
            assert f"runs/{run}" in sel["models"][key]["runs"]
            for f in ("best.pt", "normalizer.json", "config.json"):
                assert sha(APP / f"assets/models/{mode}/s{s}/{f}") == sha(RUNS / run / f)
            for f in ("best.pt", "config.json"):
                assert sha(APP / f"assets/models/{mode}/s{s}/{f}") == sel["checkpoint_sha256"][f"runs/{run}/{f}"]
            cfg = json.loads((RUNS / run / "config.json").read_text())
            assert json.loads((APP / f"assets/models/{mode}/s{s}/dims.json").read_text()) == cfg["dims"]
    tr = pd.read_csv(ROOT / "data/artemis_balanced/train.csv")
    rate = torch.as_tensor(tr[[f"present_{e}" for e in inf.EMOTIONS]].to_numpy(np.float32).mean(0))
    pres = inf.load_json("presence.json")
    assert pres["emotions"] == inf.EMOTIONS
    assert np.allclose(pres["offset"], np.log(PresenceBCE(rate).pos_weight.double().numpy()), atol=1e-12)
    pri = inf.load_json("priors.json")
    cols = [f"p_{e}" for e in inf.EMOTIONS]
    vp = pd.read_csv(VAL)[cols].to_numpy().mean(0)
    tp = pd.read_csv(ROOT / "data/artemis_balanced/train.csv")[cols].to_numpy().mean(0)
    assert np.allclose(pri["delta"], np.log(vp + 1e-8) - np.log(tp + 1e-8), atol=1e-12)
    test_paths = set(pd.read_csv(ROOT / "data/artemis_balanced/test.csv").image_path.map(lambda p: Path(p).name))
    assert not {e["file"] for e in inf.load_examples()} & test_paths


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("mode", ["r1", "r3"])
def test_app_ui(lang, mode):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=600)
    at.run()
    assert not at.exception
    assert at.session_state["model"] == "r1" and at.session_state["prior"] is False  # defaults
    at.radio(key="lang").set_value(lang).run()
    at.sidebar.radio(key="model").set_value(mode).run()
    checkboxes = [c.key for c in at.sidebar.checkbox]
    assert checkboxes == (["prior"] if mode == "r1" else [])  # prior correction only offered for R1
    at.button(key="ex_10").click().run()  # sadness example
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    names = i18n.EMOTION_NAMES[lang]
    model_name = i18n.t(lang, f"model_{mode}")
    assert any(c.value == i18n.t(lang, "model_used", name=model_name) for c in at.caption)
    assert any(c.value == i18n.t(lang, f"model_{mode}_help") for c in at.sidebar.caption)
    assert "af-card af-result" in html and any(f'af-top">{n}<' in html for n in names.values())
    assert i18n.t(lang, "sentiment_summary") in html
    if mode == "r3":
        assert i18n.t(lang, "also_feel") in html and i18n.t(lang, "also_feel_r3") in html
    else:
        assert i18n.t(lang, "also_feel") not in html  # R1 has no presence head
        at.sidebar.checkbox(key="prior").check().run()
        assert not at.exception, at.exception
        assert any(c.value == i18n.t(lang, "model_used_prior", name=model_name) for c in at.caption)
        at.sidebar.checkbox(key="prior").uncheck().run()
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
    assert "1.8357" in html and "1.8449" in html  # fresh-holdout soft CE of R1 and R3
    assert html_escape(i18n.t(lang, "row_h_r1")) in html and i18n.t(lang, "about_changelog_h") in html and "pos_weight" in html
    assert "R1_s42–46" in " ".join(m.value for m in at.markdown)


class Grab:
    """Collect the records of the app's "artfae.app" logger (it does not propagate to the root logger)."""

    def __enter__(self):
        import logging
        records = self.records = []

        class H(logging.Handler):
            def emit(self, record):
                records.append(record)

        self.log, self.h = logging.getLogger("artfae.app"), H()
        self.log.addHandler(self.h)
        return self

    def __exit__(self, *exc):
        self.log.removeHandler(self.h)

    def find(self, prefix):
        return [r for r in self.records if r.getMessage().startswith(prefix) and r.exc_info]


def full_app(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("APP_MODE", raising=False)
    st.cache_data.clear(); st.cache_resource.clear()  # earlier tests cached results and the loaded engine
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=600)
    at.run()
    assert not at.exception and at.file_uploader  # full mode: uploads available
    return at


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_full_mode_missing_weights_reports_error_model(monkeypatch, tmp_path, lang):
    """Weights missing (as in a fresh clone): the real torch.load FileNotFoundError is logged with traceback and
    the page says the model could not be loaded, never that the image is unreadable."""
    import streamlit as st
    real = inf.load_model
    monkeypatch.setattr(inf, "load_model", lambda d, mode, dev: real(tmp_path / "no-weights" / Path(d).name, mode, dev))
    try:
        with Grab() as g:
            at = full_app(monkeypatch)
            at.radio(key="lang").set_value(lang).run()
            at.button(key="ex_0").click().run()
            assert not at.exception, at.exception
            assert [e.value for e in at.error] == [i18n.t(lang, "error_model")]
            assert i18n.t(lang, "error_image") not in [e.value for e in at.error]
            rec = g.find("model loading failed")
            assert rec and isinstance(rec[0].exc_info[1], FileNotFoundError) and "best.pt" in str(rec[0].exc_info[1])
            assert "app_mode=full" in rec[0].getMessage() and not g.find("image decoding failed")
            assert not any("best.pt" in m.value for m in at.markdown)  # the details stay in the log
            at.sidebar.radio(key="page").set_value("compare").run()
            at.multiselect(key="cmp_examples").set_value([0, 1]).run()
            assert not at.exception and [e.value for e in at.error] == [i18n.t(lang, "error_model")]  # once
    finally:
        st.cache_resource.clear()  # never leave a half-initialised engine behind


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_full_mode_undecodable_upload_reports_error_image(monkeypatch, lang):
    """Only a real decoding failure shows error_image: an upload that is not an image (before any model load)."""
    with Grab() as g:
        at = full_app(monkeypatch)
        at.radio(key="lang").set_value(lang).run()
        at.file_uploader(key="upload").set_value(("not-an-image.jpg", b"this is not a JPEG", "image/jpeg")).run()
        assert not at.exception, at.exception
        assert [e.value for e in at.error] == [i18n.t(lang, "error_image")]
        rec = g.find("image decoding failed")
        assert rec and type(rec[0].exc_info[1]).__name__ == "UnidentifiedImageError"
        assert not g.find("model loading failed") and not g.find("inference failed")
