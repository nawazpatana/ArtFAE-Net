"""Gallery mode (APP_MODE=gallery, the Streamlit Cloud version): no torch, precomputed examples, uploads disabled.

  pytest tests/test_gallery.py -v
Runs without torch, transformers or the model weights: in a venv with only streamlit_cloud/requirements.txt and
pytest. The equivalence with live inference is checked by app/precompute_gallery.py (full environment).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
import core  # noqa: E402
import gallery  # noqa: E402
import i18n  # noqa: E402

EXAMPLES = core.load_examples()
HEAVY = ("torch", "transformers", "torchvision")


def test_precomputed_file_covers_every_example():
    d = core.load_json(gallery.FILE)
    assert d["format"] == gallery.FORMAT and d["emotions"] == core.EMOTIONS and d["seeds"] == core.SEEDS
    assert set(d["examples"]) == {e["file"] for e in EXAMPLES} and len(EXAMPLES) == 12
    for ex in EXAMPLES:
        rec = d["examples"][ex["file"]]
        assert rec["sha256"] == gallery.sha256(Path(ex["path"]).read_bytes()), "stale: rerun app/precompute_gallery.py"
        assert len(rec["attrs"]) == len(core.NAMES) and rec["description"] in i18n.DESCRIPTIONS_ZH
        for mode in core.MODES:
            m = rec[mode]
            assert np.shape(m["raw_logits"]) == (5, 9) and len(m["maps"]) == 9 * core.POOL_GRID ** 2
            assert (m["presence"] is None) == (mode == "r1")


def test_result_matches_predict_contract():
    data = Path(EXAMPLES[0]["path"]).read_bytes()
    for mode, prior in (("r1", False), ("r1", True), ("r3", False)):
        r = gallery.result(data, mode, prior)
        assert r["probs"].shape == (9,) and abs(r["probs"].sum() - 1) < 1e-12 and r["probs_per_seed"].shape == (5, 9)
        assert r["maps"].shape == (9, 7, 7) and r["attrs"].shape == (len(core.NAMES),)
        assert r["top"] == core.EMOTIONS[int(r["probs"].argmax())] and r["mode"] == mode
        assert (r["presence"] is None) == (mode == "r1") and r["precomputed"]
    assert not np.allclose(gallery.result(data, "r1", True)["probs"], gallery.result(data, "r1")["probs"])
    with pytest.raises(ValueError):
        gallery.result(data, "r3", True)  # no prior correction on the R3 path, as in full mode
    with pytest.raises(gallery.NotInGallery):
        gallery.result(b"not an example", "r1")


def test_app_mode(monkeypatch):
    monkeypatch.delenv("APP_MODE", raising=False)
    assert core.app_mode() == "full" and core.app_mode({}) == "full"  # local default
    assert core.app_mode({"APP_MODE": "gallery"}) == "gallery"  # Streamlit secret
    monkeypatch.setenv("APP_MODE", "Gallery")
    assert core.app_mode({"APP_MODE": "full"}) == "gallery"  # the environment wins
    monkeypatch.setenv("APP_MODE", "demo")
    with pytest.raises(ValueError):
        core.app_mode()


@pytest.mark.parametrize("entry", ["app/streamlit_app.py", "streamlit_cloud/streamlit_app.py"])
def test_gallery_mode_never_imports_torch(entry):
    """In a fresh interpreter: render the gallery and an example; no heavy module may be imported."""
    code = f"""
import sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(ROOT / entry)!r}, default_timeout=120)
at.run(); assert not at.exception, at.exception
at.button(key="ex_0").click().run(); assert not at.exception, at.exception
heavy = sorted(m for m in sys.modules if m.split(".")[0] in {HEAVY!r})
assert not heavy, heavy
print("ok")
"""
    env = {**os.environ, "APP_MODE": "gallery"} if entry.startswith("app/") else \
        {k: v for k, v in os.environ.items() if k != "APP_MODE"}  # the cloud entry point sets it itself
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-3000:]


SWEEP = """
import sys
from streamlit.testing.v1 import AppTest
ERR = {ERR!r}
at = AppTest.from_file({entry!r}, default_timeout=120); at.run()
assert not at.exception, at.exception
assert not at.get("file_uploader"), "upload must be disabled"
bad = []
for lang in ("en", "zh"):
    at.radio(key="lang").set_value(lang).run()
    for mode in ("r1", "r3"):
        at.sidebar.radio(key="model").set_value(mode).run()
        for i in range(12):
            at.button(key=f"ex_{{i}}").click().run()
            errs = [e.value for e in at.error]
            if at.exception or errs or not any("af-card af-result" in m.value for m in at.markdown):
                bad.append((lang, mode, i, errs, [str(e.message) for e in at.exception]))
    at.sidebar.radio(key="page").set_value("compare").run()
    at.multiselect(key="cmp_examples").set_value([0, 4, 6, 10]).run()
    if at.exception or at.error or sum("af-card af-result" in m.value for m in at.markdown) != 4:
        bad.append((lang, "compare"))
    at.sidebar.radio(key="page").set_value("analyse").run()
assert not bad, bad
print("swept ok")
"""


@pytest.mark.parametrize("start", ["repository root", "another folder"])
def test_cloud_entry_point_renders_every_example(start, tmp_path):
    """streamlit_cloud/streamlit_app.py as Streamlit Cloud starts it (from the repository root), and from an
    unrelated folder: all 12 examples x EN/zh-TW x R1/R3 and Compare, without error_image or a logged failure."""
    entry = str(ROOT / "streamlit_cloud" / "streamlit_app.py")
    errors = [i18n.t(lang, "error_image") for lang in i18n.LANGS]
    env = {k: v for k, v in os.environ.items() if k != "APP_MODE"}  # the entry point sets APP_MODE itself
    out = subprocess.run([sys.executable, "-c", SWEEP.format(ERR=errors, entry=entry)], env=env,
                         cwd=ROOT if start == "repository root" else tmp_path, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith("swept ok"), out.stderr[-4000:]
    assert "analyse failed" not in out.stderr and "compare failed" not in out.stderr, out.stderr[-4000:]


def test_error_image_is_logged_with_traceback(monkeypatch):
    """A failure behind the friendly error_image message reaches the log with its traceback, not the page."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("APP_MODE", "gallery")

    def boom(*a, **k):
        raise RuntimeError("boom-for-test")

    monkeypatch.setattr(gallery, "result", boom)
    import logging
    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record)

    log = logging.getLogger("artfae.app")
    h = Grab(); log.addHandler(h)
    import streamlit as st
    st.cache_data.clear()  # results are cached per process: an earlier test may hold this example already
    try:
        at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=120)
        at.run()
        at.button(key="ex_3").click().run()
        assert [e.value for e in at.error] == [i18n.t("en", "error_image")]
        assert not any("boom-for-test" in m.value for m in at.markdown)  # not shown to users
        rec = [r for r in records if r.getMessage().startswith("analyse failed")]
        assert rec and rec[0].exc_info and "boom-for-test" in str(rec[0].exc_info[1])
        assert "app_mode=gallery" in rec[0].getMessage()
    finally:
        log.removeHandler(h)


def uploaders(at):
    return [e for e in at.get("file_uploader")]


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("APP_MODE", "gallery")
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("lang", ["en", "zh"])  # zh = 繁體中文 (zh-TW)
def test_every_example_renders(app, lang):
    at = app
    at.radio(key="lang").set_value(lang).run()
    note = i18n.t(lang, "gallery_mode_note")
    assert not uploaders(at) and any(i.value == note for i in at.info)  # upload disabled, friendly note instead
    for mode in core.MODES:
        at.sidebar.radio(key="model").set_value(mode).run()
        model_name = i18n.t(lang, f"model_{mode}")
        for i, ex in enumerate(EXAMPLES):
            at.button(key=f"ex_{i}").click().run()
            assert not at.exception, (lang, mode, ex["file"], at.exception)
            r = gallery.result(Path(ex["path"]).read_bytes(), mode)
            html = " ".join(m.value for m in at.markdown)
            assert "af-card af-result" in html and f'af-top">{i18n.emotion(lang, r["top"])}<' in html, ex["file"]
            assert i18n.t(lang, "sentiment_summary") in html and i18n.t(lang, "ten_people") in html
            assert (i18n.t(lang, "also_feel") in html) == (mode == "r3")  # presence chips: R3 only
            assert i18n.description(lang, r["description"]) in html
            if lang == "zh":
                assert r["description"] not in html  # translated
            captions = [c.value for c in at.caption]
            assert i18n.t(lang, "model_used", name=model_name) in captions
            assert i18n.t(lang, "precomputed_time", seconds=r["seconds"], device=r["device"].upper()) in captions
            assert [t.label for t in at.tabs] == [i18n.t(lang, k) for k in ("tab_attention", "tab_aesthetics", "tab_details")]
            assert not uploaders(at) and any(i.value == note for i in at.info)
        # attention for every emotion and the opacity slider, on the last example
        for e in core.EMOTIONS:
            at.selectbox(key=at.selectbox[0].key).set_value(e).run()
            assert not at.exception, (lang, mode, e)
        at.slider(key="att_alpha").set_value(0.8).run()
        assert not at.exception
    at.sidebar.radio(key="model").set_value("r1").run()
    at.sidebar.checkbox(key="prior").check().run()  # R1 prior correction is precomputable: same logits + delta
    assert not at.exception
    assert i18n.t(lang, "model_used_prior", name=i18n.t(lang, "model_r1")) in [c.value for c in at.caption]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_compare_and_about_with_examples_only(app, lang):
    at = app
    at.radio(key="lang").set_value(lang).run()
    at.sidebar.radio(key="page").set_value("compare").run()
    assert not at.exception and not uploaders(at)
    assert any(i.value == i18n.t(lang, "gallery_mode_note") for i in at.info)
    at.multiselect(key="cmp_examples").set_value([0, 4, 6, 10]).run()
    assert not at.exception, at.exception
    assert sum("af-card af-result" in m.value for m in at.markdown) == 4
    at.sidebar.radio(key="page").set_value("about").run()
    assert not at.exception
    assert "1.8357" in " ".join(m.value for m in at.markdown)


def test_cloud_requirements_have_no_torch():
    req = (ROOT / "streamlit_cloud/requirements.txt").read_text()
    names = {line.split("==")[0].strip().lower() for line in req.splitlines() if line.strip() and not line.startswith("#")}
    assert names == {"streamlit", "numpy", "pandas", "pillow", "plotly"}
