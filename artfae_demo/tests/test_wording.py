"""Milestone 1 (interface and wording, Assoc. Prof. Shih-Yun Lu's recommendations): the result card's wording and
dispersed message, the ten-person caption and 0-people list, model names and claims, the JSON record, the scope
notice, the attention caveat, and text contrast. Runs without torch (gallery mode).

  pytest tests/test_wording.py -v
"""
import json
import re
import sys
from html import escape
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
import core  # noqa: E402
import gallery  # noqa: E402
import i18n  # noqa: E402
import ui  # noqa: E402

EXAMPLES = core.load_examples()
DISPERSED_EX = 4   # Leonardo, Studies for the heads of two soldiers: top two within 5 points in every setting
CLEAR_EX = 0       # Somov: 53% vs 9%
LANGS = ["en", "zh"]


def result(i, mode="r1", prior=False):
    return gallery.result(Path(EXAMPLES[i]["path"]).read_bytes(), mode, prior)


# ---------------------------------------------------------------- dispersed distribution
def test_threshold_is_a_named_constant_of_five_points():
    assert core.DISPERSED_MAX_GAP == 0.05
    base = np.full(9, 0.0)
    for gap, want in ((0.049, True), (0.05, True), (0.051, False), (0.2, False)):
        p = base.copy(); p[0], p[1] = 0.40, 0.40 - gap; p[2:] = (1 - p[0] - p[1]) / 7
        assert core.is_dispersed(p) is want, gap
    assert core.top_two(result(DISPERSED_EX)["probs"])[0][0] == "sadness"


@pytest.mark.parametrize("lang", LANGS)
def test_dispersed_card_text(lang):
    r = result(DISPERSED_EX)
    (a, pa), (b, pb) = core.top_two(r["probs"])
    assert core.is_dispersed(r["probs"]) and pa - pb < 0.05
    msg = i18n.t(lang, "dispersed_msg", a=i18n.emotion(lang, a), b=i18n.emotion(lang, b))
    html = ui.result_card("light", eyebrow=i18n.t(lang, "dispersed_eyebrow"), top_key=a, top_label=i18n.emotion(lang, a),
                          top_prob=pa, category_text="c", totals_label="t", totals=[("positive", "P", .1)],
                          dispersed=(msg, [(a, i18n.emotion(lang, a), pa), (b, i18n.emotion(lang, b), pb)],
                                     i18n.t(lang, "dispersed_note")))
    assert escape(msg) in html and escape(i18n.t(lang, "dispersed_note")) in html
    assert f"{pa * 100:.1f}%" in html and f"{pb * 100:.1f}%" in html and 'af-top">' not in html
    if lang == "en":
        assert msg == ("The predicted emotion distribution is dispersed, with similar proportions for Sadness and Fear.")
        assert i18n.t(lang, "dispersed_note") == ("These values represent the model's predicted emotion distribution, "
                                                   "not the feelings of every viewer.")


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    monkeypatch.setenv("APP_MODE", "gallery")
    st.cache_data.clear()
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def page_html(at):
    return " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("lang", LANGS)
def test_card_wording_in_the_app(app, lang):
    at = app
    at.radio(key="lang").set_value(lang).run()
    at.button(key=f"ex_{CLEAR_EX}").click().run()
    html = page_html(at)
    r = result(CLEAR_EX)
    assert escape(i18n.t(lang, "top_emotion")) in html and f'af-top">{i18n.emotion(lang, r["top"])}<' in html
    assert f'{r["probs"].max() * 100:.1f}%' in html
    cat = i18n.sentiment(lang, core.SENTIMENT[r["top"]])
    assert escape(i18n.t(lang, "top_category", cat=cat)) in html
    tot = core.sentiment_totals(r["probs"])
    assert escape(i18n.t(lang, "total_predicted")) in html
    for k in ("positive", "negative", "other"):
        assert f'{escape(i18n.sentiment(lang, k))} <b>{tot[k] * 100:.1f}%</b>' in html
    assert i18n.t(lang, "top_emotion") not in ("Most likely emotion", "最可能的情緒")
    # the dispersed example: the message replaces the single top emotion
    at.button(key=f"ex_{DISPERSED_EX}").click().run()
    html = page_html(at)
    (a, pa), (b, pb) = core.top_two(result(DISPERSED_EX)["probs"])
    assert escape(i18n.t(lang, "dispersed_msg", a=i18n.emotion(lang, a), b=i18n.emotion(lang, b))) in html
    assert f"{pa * 100:.1f}%" in html and f"{pb * 100:.1f}%" in html and "af-dispersed" in html
    # settings beside the result
    want = i18n.t(lang, "settings_line", model=i18n.t(lang, "model_r1"), prior=i18n.t(lang, "off"), version=core.APP_VERSION)
    assert escape(want) in html
    assert len(at.get("download_button")) == 1  # "Download result (JSON)"; its content: test_json_record_*


# ---------------------------------------------------------------- ten-person graphic
def test_zero_people_helper():
    p = np.array([.31, .29, .2, .1, .05, .03, .015, .004, .001])
    counts = ui.people_counts(p)
    assert counts.sum() == 10
    zero = ui.zero_people(p)
    assert zero == [i for i in np.argsort(-p) if counts[i] == 0] and all(counts[i] == 0 for i in zero)
    assert zero == [5, 6, 7, 8]


@pytest.mark.parametrize("lang", LANGS)
def test_ten_person_caption_and_zero_list_in_the_app(app, lang):
    at = app
    at.radio(key="lang").set_value(lang).run()
    at.button(key=f"ex_{DISPERSED_EX}").click().run()
    html = page_html(at)
    probs = result(DISPERSED_EX)["probs"]
    assert escape(i18n.t(lang, "ten_people_note")) in html
    if lang == "en":
        assert i18n.t(lang, "ten_people_note") == ("Illustrative ten-person representation of the predicted "
                                                    "distribution; not actual participant responses.")
    zero = ui.zero_people(probs)
    assert zero, "the dispersed example has emotions that round to 0 people"
    assert escape(i18n.t(lang, "zero_people", items="").rstrip()) in html
    for i in zero:
        assert f"{escape(i18n.emotion(lang, core.EMOTIONS[i]))} {probs[i] * 100:.1f}%" in html
    counts = ui.people_counts(probs)
    for i in range(9):  # every emotion's exact percentage is visible (legend or the 0-people list)
        assert f"{probs[i] * 100:.1f}%" in html


# ---------------------------------------------------------------- JSON record
def test_json_record_for_a_gallery_example_and_an_upload():
    ex = EXAMPLES[DISPERSED_EX]
    r = result(DISPERSED_EX, "r3")
    rec = core.result_record(r, app_mode="gallery", artwork_id=ex["artwork_id"])
    assert rec["input"] == {"type": "gallery_example", "artwork_id": ex["artwork_id"]}
    assert rec["app_version"] == core.APP_VERSION and rec["app_mode"] == "gallery"
    assert rec["model"]["key"] == "r3" and rec["model"]["seeds"] == core.SEEDS and "R3" in rec["model"]["checkpoints"]
    assert rec["calibration"]["prior_correction"] is False
    dist = rec["emotion_distribution"]
    assert list(dist) == core.EMOTIONS and abs(sum(dist.values()) - 1) < 1e-5
    assert np.allclose([dist[e] for e in core.EMOTIONS], r["probs"], atol=1e-6)
    assert rec["dispersed"]["is_dispersed"] is core.is_dispersed(r["probs"]) and rec["dispersed"]["threshold_points"] == 5
    assert set(rec["presence_r3"]["values"]) == set(core.EMOTIONS)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", rec["timestamp_utc"])
    data = Path(ex["path"]).read_bytes()
    up = core.result_record(result(CLEAR_EX, "r1", True), app_mode="cloud_live", image_sha256=gallery.sha256(data))
    assert up["input"] == {"type": "upload", "sha256": gallery.sha256(data)} and up["calibration"]["prior_correction"]
    assert "presence_r3" not in up
    for record in (rec, up):  # never the image: small, no long base64-like strings
        text = json.dumps(record)
        assert len(text) < 6000 and not re.search(r"[A-Za-z0-9+/=]{200,}", text)
        assert "image" not in {k for k in record if k != "input"} and "data" not in record["input"]
    with pytest.raises(ValueError):
        core.result_record(r, app_mode="gallery")
    with pytest.raises(ValueError):
        core.result_record(r, app_mode="gallery", artwork_id="x", image_sha256="y")


# ---------------------------------------------------------------- model names and claims
FORBIDDEN = ["most accurate model overall", "typical group of viewers", "Everyday viewers", "Rare-emotion lens",
             "Human-vote distribution", "一般觀者", "整體最準確", "少見情緒視角", "人類投票分布"]


def test_model_names_and_claims():
    assert i18n.t("en", "model_r1_plain") == "Emotion distribution (recommended)"
    assert i18n.t("en", "model_r3_plain") == "Rare-emotion sensitivity"
    assert i18n.t("en", "model_r1_help") == ("Predicts how ArtEmis annotators' votes were distributed; lowest prediction "
                                            "error (soft cross-entropy) on a fresh set of 30,280 ArtEmis paintings.")
    assert i18n.t("en", "model_r3_help") == ("Trained to notice rarer emotions such as anger and fear; it overstates them "
                                            "and is less faithful to the overall distribution.")
    assert i18n.t("en", "also_feel_r3") == ("Estimated chance that at least one annotator names this emotion (a separate "
                                           "output; not part of the distribution above).")
    texts = " ".join(str(v) for lang in i18n.LANGS for v in i18n.STRINGS[lang].values()
                     if not str(v).startswith("2026-11-06"))
    for phrase in FORBIDDEN:
        assert phrase not in texts, phrase
    assert "呂世" not in Path(APP / "i18n.py").read_text(encoding="utf-8")  # never guess Prof. Lu's Chinese name


@pytest.mark.parametrize("lang", LANGS)
def test_scope_notice_lists_the_nine_emotions(app, lang):
    at = app
    at.radio(key="lang").set_value(lang).run()
    html = page_html(at)
    sep = "、" if lang == "zh" else ", "
    names = sep.join(i18n.emotion(lang, e) for e in core.EMOTIONS)
    assert escape(i18n.t(lang, "scope_note", emotions=names)) in html


@pytest.mark.parametrize("lang", LANGS)
def test_attention_caveat_labels_toggle_and_side_by_side(app, lang):
    at = app
    at.radio(key="lang").set_value(lang).run()
    at.button(key=f"ex_{DISPERSED_EX}").click().run()
    html = page_html(at)
    assert escape(i18n.t(lang, "attention_caption")) in html and escape(i18n.t(lang, "attention_note")) in html
    (a, _), (b, _) = core.top_two(result(DISPERSED_EX)["probs"])
    for e in (a, b):
        assert escape(i18n.t(lang, "attention_map_label", emotion=i18n.emotion(lang, e))) in html
    assert [s.label for s in at.selectbox][-2:] == [i18n.t(lang, "attention_compare_a"), i18n.t(lang, "attention_compare_b")]
    at.toggle(key="att_original").set_value(True).run()
    assert not at.exception and escape(i18n.t(lang, "attention_original_label")) in page_html(at)
    assert at.slider(key="att_alpha").disabled
    tabs = [t.label for t in at.tabs]
    assert i18n.t(lang, "tab_aesthetics") in tabs and i18n.t(lang, "tab_aesthetics") != "Aesthetics"


@pytest.mark.parametrize("lang", LANGS)
def test_visual_attributes_key_rows_and_plain_percentiles(app, lang):
    at = app
    at.radio(key="lang").set_value(lang).run()
    at.button(key=f"ex_{CLEAR_EX}").click().run()
    html = page_html(at)
    r = result(CLEAR_EX)
    pct = core.percentiles(r["attrs"], core.load_json("attr_percentiles.json"))
    p = int(round(dict(zip(core.NAMES, pct))["luminance"]))
    assert escape(i18n.t(lang, "pct_luminance", p=p, n=43012)) in html
    if lang == "en":
        assert f"Brighter than {p}% of the 43,012 training paintings (WikiArt, ArtEmis)" in html
    assert [e.label for e in at.expander if e.label == i18n.t(lang, "attrs_more")]


# ---------------------------------------------------------------- accessibility
def contrast(a, b):
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= .03928 else ((x + .055) / 1.055) ** 2.4 for x in c]
        return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]
    hi, lo = sorted([lum(a), lum(b)], reverse=True)
    return (hi + .05) / (lo + .05)


def test_explanatory_text_meets_wcag_aa():
    for m, v in ui.THEME.items():
        for bg in ("bg", "card", "track"):
            assert contrast(v["muted"], v[bg]) >= 4.5, (m, bg)
            assert contrast(v["text"], v[bg]) >= 4.5, (m, bg)
    css = ui.css("light")
    assert '[data-testid="stCaptionContainer"] {{ opacity: 1 !important; }}'.replace("{{", "{").replace("}}", "}") in css
    sizes = [float(x) for x in re.findall(r"\.af-(?:muted|disp-note|zero|ksent|settings|note)[^{]*\{[^}]*font-size: ([\d.]+)rem", css)]
    assert sizes and min(sizes) >= 1.0, sizes


# ---------------------------------------------------------------- model vs. real ArtEmis viewers
def test_examples_carry_their_annotator_count():
    import pandas as pd
    manifest = ROOT / "data/artemis_balanced/val.csv"
    for e in EXAMPLES:
        n = e["n_votes"]
        assert isinstance(n, int) and n >= 1
        shares = np.array([e["annotators"][k] for k in core.EMOTIONS])
        assert np.allclose(shares * n, np.round(shares * n), atol=1e-6)  # whole annotators
    if manifest.exists():  # the research server: the count is the manifest's sum of n_<emotion>
        v = pd.read_csv(manifest)
        for e in EXAMPLES:
            assert e["n_votes"] == int(sum(v.loc[e["val_row"], f"n_{k}"] for k in core.EMOTIONS))


@pytest.mark.parametrize("lang", LANGS)
def test_versus_chart_annotator_note(app, lang):
    at = app
    at.radio(key="lang").set_value(lang).run()
    for i in (CLEAR_EX, DISPERSED_EX):
        at.button(key=f"ex_{i}").click().run()
        html = page_html(at)
        n = EXAMPLES[i]["n_votes"]
        assert escape(i18n.t(lang, "vs_n", n=n)) in html and escape(i18n.t(lang, "vs_few", n=n)) in html
    assert i18n.t("en", "vs_n", n=5) == "Based on 5 ArtEmis annotators"
    assert i18n.t("en", "vs_few", n=5) == "With few annotators, a share can look very certain (e.g. 100% = all 5 chose it)."


def test_versus_chart_axis_and_neutral_legend():
    for m in ("light", "dark"):
        fig = ui.versus_chart(m, core.EMOTIONS, core.EMOTIONS, np.full(9, 1 / 9), np.full(9, 1 / 9), "Model",
                              "ArtEmis viewers", i18n.t("en", "vs_axis"))
        assert fig.layout.xaxis.title.text == "Proportion" and i18n.t("zh", "vs_axis") == "比例"
        legend = [tr for tr in fig.data if tr.showlegend]
        assert [tr.name for tr in legend] == ["Model", "ArtEmis viewers"]
        for tr in legend:  # neutral swatches, not an emotion colour
            assert tr.marker.color == ui.LEGEND_NEUTRAL[m] and tr.marker.color not in ui.PALETTE[m].values()
        assert not legend[0].marker.pattern.shape and legend[1].marker.pattern.shape == "/"  # solid vs hatched
        data = [tr for tr in fig.data if not tr.showlegend]
        assert {tr.name: bool(tr.marker.pattern.shape) for tr in data} == {"ArtEmis viewers": True, "Model": False}


@pytest.mark.parametrize("lang", LANGS)
def test_showing_line_names_the_example(app, lang):
    at = app
    at.radio(key="lang").set_value(lang).run()
    at.button(key=f"ex_{CLEAR_EX}").click().run()
    ex = EXAMPLES[CLEAR_EX]
    artist, title, year = ui.artwork_parts(ex["artist"], ex["painting"], i18n.t(lang, "untitled"))
    want = i18n.t(lang, "showing_example", caption=ui.caption_text(artist, title, year))
    assert f'af-showing">{escape(want)}<' in page_html(at)


def test_versus_chart_legend_takes_no_bar_slot():
    """Legend-only traces share their series' offsetgroup: the grouped bars keep exactly two slots."""
    fig = ui.versus_chart("light", core.EMOTIONS, core.EMOTIONS, np.full(9, 1 / 9), np.full(9, 1 / 9), "M", "V", "P")
    assert {tr.offsetgroup for tr in fig.data} == {"model", "viewers"} and len(fig.data) == 4
