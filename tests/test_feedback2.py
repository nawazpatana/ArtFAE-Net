"""Second feedback round (Assoc. Prof. Shih-Yun Lu, 2026-11-20): emotion categories (Q1), colour palette (Q3),
descriptive aesthetic properties (Q2), attention explanation + grid (Q4), map comparison (Q5) and the
iconological reading (Q6), in EN and zh-TW.

  pytest tests/test_feedback2.py -v
Everything except the last block runs without torch (APP_MODE=gallery, precomputed examples). The last block
uploads a re-encoded example in cloud_live mode (live CPU inference; needs torch and the cached CLIP checkpoint,
otherwise it is skipped).
"""
import io
import json
import logging
import re
import subprocess
import sys
from html import escape
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
import core  # noqa: E402
import gallery  # noqa: E402
import i18n  # noqa: E402
import ui  # noqa: E402
import visual  # noqa: E402

EXAMPLES = core.load_examples()
LANGS = ["en", "zh"]
HEX = re.compile(r"^#[0-9A-F]{6}$")
EX = 5  # Modigliani, Portrait of Paul Alexandre: the most varied palette of the examples


def data(i):
    return Path(EXAMPLES[i]["path"]).read_bytes()


def html_of(at):
    return " ".join(m.value for m in at.markdown)


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


def open_example(at, lang, i=EX):
    if lang != "en":
        at.radio(key="lang").set_value(lang).run()
    at.button(key=f"ex_{i}").click().run()
    assert not at.exception, at.exception
    return at


# ================================================================ Q3 colour palette
def blocks_image():
    """Six solid colours with known areas (out of 100 columns): 30, 25, 20, 12, 8, 5."""
    colours = [(200, 30, 40), (20, 90, 200), (240, 240, 230), (10, 10, 10), (40, 160, 60), (250, 200, 0)]
    widths = [30, 25, 20, 12, 8, 5]
    arr = np.zeros((50, 100, 3), np.uint8)
    x = 0
    for c, w in zip(colours, widths):
        arr[:, x:x + w] = c
        x += w
    return Image.fromarray(arr), colours, widths


def test_palette_recovers_known_colours_and_shares():
    img, colours, widths = blocks_image()
    pal = visual.palette(img)
    assert len(pal) == 6
    assert [p["hex"] for p in pal] == [visual.hex_code(c) for c in colours]  # exact, sorted by share
    assert [p["percent"] for p in pal] == [float(w) for w in widths]
    assert all(HEX.match(p["hex"]) for p in pal)


@pytest.mark.parametrize("i", range(12))
def test_palette_hex_matches_swatch_colour_and_shares_sum_to_100(i):
    pal = visual.palette(core.load_upload(data(i)))
    assert len(pal) == 6
    shares = [p["share"] for p in pal]
    assert shares == sorted(shares, reverse=True) and abs(sum(shares) - 1) < 1e-12
    assert round(sum(p["percent"] for p in pal) * 10) == 1000  # shown values sum to exactly 100.0%
    for p in pal:
        assert HEX.match(p["hex"]) and p["hex"] == p["hex"].upper()
        assert visual.hex_to_rgb(p["hex"]) == tuple(p["rgb"])                       # hex = the swatch's RGB
        assert p["hex"] == visual.hex_code(visual.lab_to_srgb(p["lab"]))            # = the cluster centre
        lab_back = visual.srgb_to_lab(np.array(p["rgb"], float))
        assert np.linalg.norm(lab_back - np.array(p["lab"])) < 1.0                  # rounding only (ΔE < 1)
        html = ui.swatch(p["hex"], "x", "y")
        assert f'background:{p["hex"]}"' in html and f'data-hex="{p["hex"]}"' in html
    assert visual.palette(core.load_upload(data(i))) == pal  # deterministic


def test_colour_conversions_round_trip():
    rgb = np.random.default_rng(1).integers(0, 256, (500, 3))
    assert np.abs(visual.lab_to_srgb(visual.srgb_to_lab(rgb)) - rgb).max() < 1e-6
    assert visual.hex_code((255, 0, 16)) == "#FF0010" and visual.hex_to_rgb("#FF0010") == (255, 0, 16)
    assert visual.round_shares([1 / 3, 1 / 3, 1 / 3]) == [334, 333, 333]


def test_gallery_json_has_the_palette_and_it_matches_a_fresh_computation():
    d = core.load_json(gallery.FILE)
    assert d["format"] == gallery.FORMAT == "artfae-gallery/2"
    for i, ex in enumerate(EXAMPLES):
        live = visual.palette(core.load_upload(data(i)))
        stored = d["examples"][ex["file"]]["palette"]
        assert [(p["hex"], p["percent"]) for p in stored] == [(p["hex"], p["percent"]) for p in live]
        assert np.allclose([p["share"] for p in stored], [p["share"] for p in live], atol=1e-12)
        r = gallery.result(data(i), "r1")
        assert r["palette"] == stored and r["aesthetic"] == d["examples"][ex["file"]]["aesthetic"]


@pytest.mark.parametrize("lang", LANGS)
def test_palette_in_the_app_and_in_the_json_record(app, lang):
    at = open_example(app, lang)
    pal = gallery.result(data(EX), "r1")["palette"]
    codes = [c.value for c in at.code]
    assert codes[:6] == [p["hex"] for p in pal]  # st.code: Streamlit's copy button
    html = html_of(at)
    assert escape(i18n.t(lang, "palette_h")) in html
    caps = [c.value for c in at.caption]
    assert i18n.t(lang, "palette_note") in caps and i18n.t(lang, "palette_copy") in caps
    for p in pal:
        pct = f'{p["percent"]:.1f}%'
        assert f'background:{p["hex"]}"' in html and escape(i18n.t(lang, "palette_share", pct=pct)) in html
    assert len(at.get("download_button")) == 1  # its content is the record built here (AppTest cannot read it)
    rec = core.result_record(gallery.result(data(EX), "r1"), app_mode="gallery",
                             artwork_id=EXAMPLES[EX]["artwork_id"], palette=pal)
    cols = rec["palette"]["colours"]
    assert [c["hex"] for c in cols] == codes[:6] and round(sum(c["percent"] for c in cols) * 10) == 1000
    assert "k-means" in rec["palette"]["method"] and "CIELAB" in rec["palette"]["method"]
    assert len(json.dumps(rec)) < 8000


def test_json_record_without_palette_is_unchanged_otherwise():
    r = gallery.result(data(0), "r1")
    a = core.result_record(r, app_mode="gallery", artwork_id="x/y", timestamp="t")
    b = core.result_record(r, app_mode="gallery", artwork_id="x/y", timestamp="t", palette=r["palette"])
    assert "palette" not in a and {k: v for k, v in b.items() if k != "palette"} == a
    assert core.APP_VERSION == "1.3.0"


# ================================================================ Q2 descriptive aesthetic properties
def test_measures_are_in_range_deterministic_and_match_the_attributes():
    names = list(core.NAMES)
    for i in range(12):
        img = core.load_upload(data(i))
        m = visual.aesthetic_measures(img)
        assert list(m["values"]) == visual.MEASURE_NAMES
        assert all(0 <= v <= 1 for v in m["values"].values()), m
        assert m == visual.aesthetic_measures(img)
        a = dict(zip(names, gallery.result(data(i), "r1")["attrs"]))
        assert abs(m["values"]["edge_density"] - a["edge_density"]) < 1e-6
        assert abs(m["values"]["rms_contrast"] - a["luminance_std"]) < 1e-6
        assert abs(m["values"]["luminance_range"] - a["contrast_p95_p05"]) < 1e-6


def test_measures_on_constructed_images():
    grey = Image.fromarray(np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1)))
    m = visual.aesthetic_measures(grey)["values"]
    assert m["hue_spread"] == 0 and m["complementary"] == 0  # no colour, no harmony measure
    half = np.zeros((256, 256, 3), np.uint8); half[:, :128] = 255
    m = visual.aesthetic_measures(Image.fromarray(half))
    assert m["values"]["balance_lum_lr"] == 0 and m["sides"]["lum_lr"] == "left" and m["values"]["balance_lum_tb"] == 1
    flat = visual.aesthetic_measures(Image.new("RGB", (64, 64), (120, 30, 30)))["values"]
    assert flat["colour_entropy"] == 0 and flat["edge_density"] == 0 and flat["rms_contrast"] < 1e-6
    for c1, c2 in (((0, 0, 255), (255, 255, 0)), ((255, 0, 0), (0, 255, 255)), ((0, 200, 0), (200, 0, 200))):
        pair = np.zeros((256, 256, 3), np.uint8); pair[:, :128] = c1; pair[:, 128:] = c2  # exact complements
        m = visual.aesthetic_measures(Image.fromarray(pair))["values"]
        assert m["complementary"] > .95 and m["hue_spread"] > .95, (c1, c2, m)
    one = np.zeros((256, 256, 3), np.uint8); one[:, :128] = (255, 0, 0); one[:, 128:] = (200, 40, 40)  # one hue family
    m = visual.aesthetic_measures(Image.fromarray(one))["values"]
    assert m["complementary"] == 0 and m["hue_spread"] < .05


def test_percentile_table_covers_the_training_paintings():
    t = core.load_json("aesthetic_percentiles.json")
    assert t["names"] == visual.MEASURE_NAMES and t["n_train"] == 43012 and len(t["quantiles"]) == len(visual.MEASURE_NAMES)
    for q in t["quantiles"]:
        assert len(q) == 1001 and np.all(np.diff(q) >= 0)
    assert all(v < 1e-6 for v in t["check_vs_attributes_npy_max_abs_diff"].values())
    pct = core.percentiles([0.0] * 10, t)
    assert np.all((pct >= 0) & (pct <= 100))


def test_no_quality_score_anywhere():
    for lang in LANGS:
        assert i18n.t(lang, "aes_not_score") == {"en": "These describe aesthetic properties; they are not a quality score.",
                                                 "zh": "這些數值描述作品的美感屬性，並非品質評分。"}[lang]
        texts = [v for k, v in i18n.STRINGS[lang].items() if k not in ("aes_not_score", "about_changelog")]
        texts += list(i18n.AESTHETIC_NAMES[lang].values())
        bad = [t for t in texts if re.search(r"quality score|beauty score|aesthetic score|美感分數|品質評分|美學分數", str(t))]
        assert not bad, bad
    assert not any("score" in k or "quality" in k for k in visual.MEASURE_NAMES)


@pytest.mark.parametrize("lang", LANGS)
def test_aesthetic_properties_in_the_app(app, lang):
    at = open_example(app, lang)
    html = html_of(at)
    assert escape(i18n.t(lang, "aes_h")) in html and escape(i18n.t(lang, "aes_not_score")) in html
    t = core.load_json("aesthetic_percentiles.json")
    m = gallery.result(data(EX), "r1")["aesthetic"]
    pct = core.percentiles([m["values"][k] for k in visual.MEASURE_NAMES], t)
    for g in visual.GROUPS:
        assert escape(i18n.t(lang, f"aes_g_{g}")) in html and escape(i18n.t(lang, f"aes_x_{g}")) in html
    for k, p in zip(visual.MEASURE_NAMES, pct):
        assert escape(i18n.aesthetic(lang, k)) in html
        assert escape(i18n.t(lang, "percentile", p=int(round(p)))) in html
    side = i18n.side(lang, "lum", m["sides"]["lum_lr"])
    assert f'{m["values"]["balance_lum_lr"]:.2f} · {escape(side)}' in html


# ================================================================ Q4 attention explanation + grid
EXPLAIN_EN = ("Bright regions received higher attention weights from this emotion's query: the model compares one learned "
              "query per emotion with 49 regions (a 7 × 7 grid over the central square CLIP sees), and the weights sum to "
              "100%. The maps are coarse, were not validated as explanations, and the attention module contributed little "
              "to the final predictions in our experiments.")


def test_attention_explanation_text_and_facts():
    assert i18n.t("en", "attention_explain") == EXPLAIN_EN
    assert "49" in i18n.t("zh", "attention_explain") and "7 × 7" in i18n.t("zh", "attention_explain")
    for i in range(12):  # what the sentence claims about the maps holds for every stored map
        for mode in core.MODES:
            maps = gallery.result(data(i), mode)["maps"]
            assert maps.shape == (9, 7, 7) and core.POOL_GRID ** 2 == 49
            assert np.allclose(maps.reshape(9, 49).sum(1), 1, atol=1e-5) and (maps >= 0).all()


def test_grid_overlay_draws_only_the_grid_lines():
    base = Image.new("RGB", (640, 640), (10, 20, 30))
    g = np.asarray(visual.with_grid(base))
    lines = [min(round(i * 640 / 7), 639) for i in range(8)]
    for x in lines:
        assert (g[:, x] == 255).all() and (g[x, :] == 255).all()
    mask = np.ones((640, 640), bool)
    for x in lines:
        mask[:, max(0, x - 1):x + 2] = False; mask[max(0, x - 1):x + 2, :] = False
    assert (g[mask] == (10, 20, 30)).all()  # nothing else changes
    assert np.asarray(base).max() == 30  # the input is not modified


@pytest.mark.parametrize("lang", LANGS)
def test_attention_explanation_beside_the_maps_and_grid_toggle(app, lang):
    at = open_example(app, lang)
    html = html_of(at)
    assert html.count(escape(i18n.t(lang, "attention_explain"))) >= 2  # beside the main map and the comparison
    toggle = at.toggle(key="att_grid")
    assert toggle.label == i18n.t(lang, "attention_grid") and toggle.value is False
    n_images = len(at.get("imgs"))
    toggle.set_value(True).run()
    assert not at.exception and len(at.get("imgs")) == n_images


# ================================================================ Q5 map comparison
def test_similarity_labels_and_value():
    a = np.arange(49, dtype=float)
    assert visual.map_similarity(a, a) == (1.0, "very_similar")
    assert visual.map_similarity(a, -a)[1] == "different"
    rng = np.random.default_rng(0)
    for target, want in ((0.95, "very_similar"), (0.8, "similar"), (0.3, "different")):
        noise = rng.normal(size=49); noise -= noise.mean()
        noise -= noise @ (a - a.mean()) / ((a - a.mean()) @ (a - a.mean())) * (a - a.mean())
        noise *= np.std(a) / np.std(noise) * np.sqrt(1 / target ** 2 - 1)
        r, label = visual.map_similarity(a, a + noise)
        assert abs(r - target) < 1e-9 and label == want
    assert visual.map_similarity(np.full(49, 1 / 49), a)[1] == "different"  # constant map: no correlation
    maps = gallery.result(data(EX), "r1")["maps"]
    r, _ = visual.map_similarity(maps[0], maps[6])
    assert abs(r - np.corrcoef(maps[0].ravel(), maps[6].ravel())[0, 1]) < 1e-12


def test_difference_map_colours_and_scale():
    sq = Image.new("RGB", (70, 70), (128, 128, 128))
    a = np.zeros((7, 7)); b = np.zeros((7, 7))
    a[1, 1] = 0.3; b[5, 5] = 0.2
    img, pp = visual.difference_overlay(sq, a, b, size=70)
    px = np.asarray(img).astype(int)
    assert abs(pp - 30.0) < 1e-9  # max |A - B| in percentage points
    r_a, _, b_a = px[15, 15]; r_b, _, b_b = px[55, 55]
    assert r_a > b_a + 40 and b_b > r_b + 40  # A higher = orange, B higher = blue
    flipped, pp2 = visual.difference_overlay(sq, b, a, size=70)
    assert pp2 == pp and np.asarray(flipped)[55, 55, 0] > np.asarray(flipped)[55, 55, 2]
    same, pp0 = visual.difference_overlay(sq, a, a, size=70)
    assert pp0 == 0


@pytest.mark.parametrize("lang", LANGS)
def test_comparison_in_the_app(app, lang):
    at = open_example(app, lang)
    r = gallery.result(data(EX), "r1")
    ea, eb = at.selectbox(key=[s.key for s in at.selectbox if s.key.startswith("att_cmp_a")][0]).value, \
        at.selectbox(key=[s.key for s in at.selectbox if s.key.startswith("att_cmp_b")][0]).value
    rho, label = visual.map_similarity(r["maps"][core.EMOTIONS.index(ea)], r["maps"][core.EMOTIONS.index(eb)])
    html = html_of(at)
    name = i18n.t(lang, f"sim_{label}")
    assert escape(i18n.t(lang, "cmp_similarity", label="§", r=f"{rho:.2f}")).replace("§", f"<b>{escape(name)}</b>") in html
    assert escape(i18n.t(lang, "cmp_same_region")) in html
    assert escape(i18n.t(lang, "cmp_diff_label", a=i18n.emotion(lang, ea), b=i18n.emotion(lang, eb))) in html
    assert escape(i18n.t(lang, "cmp_diff_a", a=i18n.emotion(lang, ea))) in html
    assert i18n.t(lang, "cmp_sim_scale") in [c.value for c in at.caption]
    if lang == "en":
        assert i18n.t(lang, "cmp_same_region") == ("Emotion queries often attend to the same visually dominant region; "
                                                   "differences between emotions are usually small.")
    key_b = [s.key for s in at.selectbox if s.key.startswith("att_cmp_b")][0]
    at.selectbox(key=key_b).set_value(ea).run()  # the same emotion on both sides
    assert not at.exception and i18n.t(lang, "cmp_same_emotion") in [i.value for i in at.info]


def test_gallery_mode_has_everything_the_comparison_needs():
    d = core.load_json(gallery.FILE)
    for ex in EXAMPLES:
        for mode in core.MODES:
            assert len(d["examples"][ex["file"]][mode]["maps"]) == 9 * 49  # all nine maps, so any pair can be compared


# ================================================================ Q1 emotion categories
def test_emotion_category_text():
    neg = [e for e in core.EMOTIONS if core.SENTIMENT[e] == "negative"]
    pos = [e for e in core.EMOTIONS if core.SENTIMENT[e] == "positive"]
    assert neg == ["anger", "disgust", "fear", "sadness"] and pos == ["amusement", "awe", "contentment", "excitement"]
    en = i18n.t("en", "emo_cat_intro")
    for s in ("ArtEmis (Achlioptas et al., CVPR 2021)", "Ekman's basic emotions", "finer-grained forms of happiness",
              "one dominant emotion", "at least 5 annotators", "share of annotators"):
        assert s in en, s
    for lang in LANGS:
        assert set(i18n.EMOTION_DEFS[lang]) == set(core.EMOTIONS)
        assert all(len(v) < 120 and "\n" not in v for v in i18n.EMOTION_DEFS[lang].values())  # one line each


@pytest.mark.parametrize("lang", LANGS)
def test_emotion_categories_on_results_and_about(app, lang):
    at = open_example(app, lang)
    for page in ("analyse", "about"):
        if page == "about":
            at.radio(key="page").set_value("about").run()
        assert i18n.t(lang, "emo_cat_h") in [e.label for e in at.expander]
        html = html_of(at)
        assert escape(i18n.t(lang, "emo_cat_intro")) in html
        for e in core.EMOTIONS:
            assert f"<dt>{escape(i18n.emotion(lang, e))}</dt><dd>{escape(i18n.emotion_def(lang, e))}</dd>" in html
        assert i18n.t(lang, "emo_cat_cite") in [c.value for c in at.caption]


# ================================================================ Q6 iconology
@pytest.mark.parametrize("lang", LANGS)
def test_iconology_levels_for_a_gallery_example(app, lang):
    at = open_example(app, lang)
    html = html_of(at)
    for k in ("icon_h", "icon_l1_h", "icon_l1_x", "icon_l2_h", "icon_l2_x", "icon_l3_h", "icon_l3_x", "icon_l3_private",
              "icon_l2_note"):
        assert escape(i18n.t(lang, k)) in html, k
    assert escape(i18n.t(lang, "icon_l2_upload")) not in html
    r = gallery.result(data(EX), "r1")
    assert ui.palette_strip([(c["hex"], c["share"]) for c in r["palette"]]) in html  # level 1: the palette
    assert escape(i18n.description(lang, r["description"])) in html                  # level 1: visual attributes
    top = core.EMOTIONS[int(r["probs"].argmax())]
    assert f'{escape(i18n.emotion(lang, top))} {r["probs"].max() * 100:.1f}%' in html  # level 1: expressional qualities
    m = core.load_json("examples/metadata.json")[EXAMPLES[EX]["file"]]
    for label, value in (("icon_title", m["title"]), ("icon_artist", m["artist"]), ("icon_date", i18n.date(lang, m["date"])),
                         ("icon_style", i18n.style(lang, m["style"])), ("icon_genre", i18n.genre(lang, m["genre"]))):
        assert f"<dt>{escape(i18n.t(lang, label))}</dt><dd>{escape(value)}</dd>" in html
    assert (m["title"], m["artist"], m["date"], m["genre"]) == ("Portrait of Paul Alexandre", "Amedeo Modigliani", "1909", "portrait")
    box = at.text_area[0]
    assert box.label == i18n.t(lang, "icon_l3_box")
    if lang == "en":
        assert i18n.t(lang, "icon_l3_private") == "Not saved. Only you can see it."
        assert i18n.t(lang, "icon_l2_note") == ("Themes and symbols are identified by art-historical knowledge; the model "
                                                "does not recognise them.")


def test_example_metadata_is_complete_and_sourced():
    meta = core.load_json("examples/metadata.json")
    assert set(meta) == {e["file"] for e in EXAMPLES}
    for m in meta.values():
        assert all(m[k] for k in ("title", "artist", "date", "style", "genre"))
        assert m["source"].startswith("https://www.wikiart.org/")
        assert i18n.genre("zh", m["genre"]) != m["genre"] and i18n.style("zh", m["style"]) != m["style"]  # translated


def test_interpretation_is_never_stored_logged_or_downloaded(app, tmp_path, monkeypatch, caplog):
    secret = "MY-PRIVATE-READING-7731 a vanitas about mortality"
    before = {p for p in Path("/tmp").glob("*")} if Path("/tmp").exists() else set()
    caplog.set_level(logging.DEBUG)
    at = open_example(app, "en")
    at.text_area[0].input(secret).run()
    assert not at.exception and at.text_area[0].value == secret
    assert secret not in html_of(at)  # not echoed into the page
    for b in at.get("download_button"):
        assert secret not in str(b.proto)
    assert secret not in caplog.text
    new = [p for p in Path("/tmp").glob("*") if p not in before] if before else []
    for p in new:
        if p.is_file() and p.stat().st_size < 5_000_000:
            assert secret.encode() not in p.read_bytes()
    src = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    body = src[src.index("def iconology"):src.index("def aesthetics")]
    assert "LOG." not in body and "open(" not in body and "write" not in body


@pytest.mark.parametrize("lang", LANGS)
def test_about_cites_panofsky(app, lang):
    at = app
    if lang != "en":
        at.radio(key="lang").set_value(lang).run()
    at.radio(key="page").set_value("about").run()
    html = html_of(at)
    assert escape(i18n.t(lang, "about_panofsky_h")) in html
    assert "Studies in Iconology" in i18n.t(lang, "about_panofsky") and "1939" in i18n.t(lang, "about_panofsky")
    assert escape(i18n.t(lang, "about_panofsky")) in html


# ================================================================ torch-free
def test_visual_module_is_torch_free():
    code = "import sys; sys.path.insert(0, %r); import visual, gallery; " \
           "assert not [m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers')]; print('ok')" % str(APP)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr


def test_cloud_entry_reloads_the_visual_module():
    src = (ROOT / "streamlit_cloud/streamlit_app.py").read_text(encoding="utf-8")
    assert '"visual"' in src.split("OWN_MODULES")[1].split("\n")[0]


# ================================================================ an upload (live; cloud_live mode)
def _live_ready():
    try:
        import torch  # noqa: F401
        import inference as inf
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(inf.CLIP_MODEL, "pytorch_model.bin", revision=inf.CLIP_REVISION), str)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _live_ready(), reason="needs torch and the cached CLIP checkpoint (live inference)")
@pytest.mark.parametrize("lang", LANGS)
def test_upload_gets_a_live_palette_and_no_iconography(monkeypatch, lang):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    monkeypatch.setenv("APP_MODE", "cloud_live"); monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    st.cache_data.clear()
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=600); at.run()
    if lang != "en":
        at.radio(key="lang").set_value(lang).run()
    b = io.BytesIO(); Image.open(EXAMPLES[EX]["path"]).convert("RGB").save(b, "PNG")  # not byte-identical: live
    up = b.getvalue()
    assert not gallery.has(up)
    at.file_uploader(key="upload").set_value(("photo.png", up, "image/png")).run()
    assert not at.exception and not at.error, (at.exception, [e.value for e in at.error])
    want = visual.palette(core.load_upload(up))
    assert [c.value for c in at.code][:6] == [p["hex"] for p in want]
    html = html_of(at)
    assert escape(i18n.t(lang, "icon_l2_upload")) in html and escape(i18n.t(lang, "icon_l2_note")) not in html
    assert escape(i18n.t(lang, "aes_not_score")) in html
