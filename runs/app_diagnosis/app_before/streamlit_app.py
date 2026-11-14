"""ArtFAE-Net bilingual demo (English / 繁體中文).

run:  streamlit run app/streamlit_app.py
"""
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i18n  # noqa: E402
import inference as inf  # noqa: E402
import ui  # noqa: E402

EMOS = inf.EMOTIONS
GROUPS = {
    "group_luminance": ["luminance", "value", "dark", "bright"],
    "group_colour": ["red", "green", "blue", "saturation", "saturation_var", "warm", "cool", "hue_sin", "hue_cos"],
    "group_contrast": ["luminance_std", "contrast_p95_p05"],
    "group_edges": ["edge_density", "edge_tl", "edge_tr", "edge_bl", "edge_br"],
}
SHARES = {"warm", "cool", "dark", "bright", "edge_density", "edge_tl", "edge_tr", "edge_bl", "edge_br",
          *(f"hue_bin_{i}" for i in range(8))}

st.set_page_config(page_title=i18n.t("en", "app_title"), layout="wide", initial_sidebar_state="expanded")
ss = st.session_state
ss.setdefault("lang", "en")
ss.setdefault("page", "analyse")
ss.setdefault("source", None)


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False)
def engine():
    return inf.load_engine()


@st.cache_data(show_spinner=False)
def assets():
    return inf.load_json("attr_percentiles.json"), inf.load_examples(), inf.load_json("results.json")


@st.cache_data(show_spinner=False, max_entries=64)
def analyse_bytes(data: bytes) -> dict:
    return inf.predict(engine(), inf.load_image(io.BytesIO(data)))


@st.cache_data(show_spinner=False)
def thumbnail(path: str, size: int = 360) -> Image.Image:
    im = inf.load_image(path)
    im.thumbnail((size, size))
    return im


def file_bytes(path) -> bytes:
    return Path(path).read_bytes()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.radio(i18n.t(ss.lang, "language"), list(i18n.LANGS), format_func=i18n.LANGS.get, key="lang", horizontal=True)
    L = ss.lang
    st.radio(i18n.t(L, "page"), ["analyse", "compare", "about"], format_func=lambda k: i18n.t(L, f"page_{k}"),
             key="page")
    st.divider()
    st.caption(i18n.t(L, "sidebar_note"))

M = ui.mode(getattr(st.context.theme, "type", None))
st.markdown(ui.css(M), unsafe_allow_html=True)
PCT, EXAMPLES, RESULTS = assets()
tr = lambda key, **kw: i18n.t(L, key, **kw)  # noqa: E731
emo = lambda e: i18n.emotion(L, e)  # noqa: E731


def hero(title_key, sub_key=None):
    sub = f"<p>{ui.esc(tr(sub_key))}</p>" if sub_key else ""
    st.markdown(f'<div class="af-hero"><h1>{ui.esc(tr(title_key))}</h1>{sub}</div>', unsafe_allow_html=True)


def ensure_engine():
    if "engine_ready" not in ss:
        with st.spinner(tr("loading_models")):
            engine()
        ss.engine_ready = True


def run(data: bytes):
    ensure_engine()
    with st.spinner(tr("spinner")):
        return analyse_bytes(data)


def fmt_value(name, x):
    return f"{x * 100:.1f}%" if name in SHARES else f"{x:.3f}"


def pct_label(p):
    return tr("percentile", p=p)


# ---------------------------------------------------------------- analyse
def on_upload():
    ss.source = ("upload", None) if ss.get("upload") is not None else None


def pick_example(i):
    ss.source = ("example", i)


def gallery():
    with st.expander(tr("or_gallery"), expanded=ss.source is None):
        st.caption(tr("gallery_note"))
        for row in range(0, len(EXAMPLES), 6):
            cols = st.columns(6, gap="small")
            for c, i in zip(cols, range(row, min(row + 6, len(EXAMPLES)))):
                ex = EXAMPLES[i]
                with c:
                    st.image(thumbnail(ex["path"]), width="stretch")
                    st.markdown(f'<div class="af-thumb-cap">{ui.esc(ex["painting"])}</div>', unsafe_allow_html=True)
                    st.button(tr("use_example"), key=f"ex_{i}", on_click=pick_example, args=(i,), width="stretch")


def page_analyse():
    hero("app_title", "app_tagline")
    st.file_uploader(tr("upload"), type=["jpg", "jpeg", "png", "webp"], help=tr("upload_help"), key="upload",
                     on_change=on_upload)
    gallery()
    src = ss.source
    if src is None:
        st.info(tr("empty_state"))
        return
    example = None
    if src[0] == "upload":
        up = ss.get("upload")
        if up is None:
            st.info(tr("empty_state"))
            return
        data = up.getvalue()
    else:
        example = EXAMPLES[src[1]]
        data = file_bytes(example["path"])
    try:
        img = inf.load_image(io.BytesIO(data))
        r = run(data)
    except Exception:  # unreadable upload
        st.error(tr("error_image"))
        return
    key = hashlib.sha1(data).hexdigest()[:10]
    probs, top = r["probs"], r["top"]
    sent = inf.sentiment_totals(probs)

    left, right = st.columns([5, 7], gap="large")
    with left:
        st.image(img, width="stretch")
        if example:
            st.markdown(f'<div class="af-meta">{ui.esc(tr("artwork_meta", painting=example["painting"], artist=example["artist"], style=example["style"]))}</div>',
                        unsafe_allow_html=True)
        st.caption(tr("inference_time", seconds=r["seconds"], device=r["device"].upper()))
    with right:
        st.markdown(ui.result_card(M, top, emo(top), float(probs.max()), inf.SENTIMENT[top],
                                   i18n.sentiment(L, inf.SENTIMENT[top]), tr("top_emotion"), tr("probability"),
                                   tr("sentiment")), unsafe_allow_html=True)
        st.markdown(f'<div class="af-section">{ui.esc(tr("distribution"))}</div>', unsafe_allow_html=True)
        st.plotly_chart(ui.distribution_chart(M, EMOS, [emo(e) for e in EMOS], probs,
                                              [i18n.sentiment(L, inf.SENTIMENT[e]) for e in EMOS], tr("prob_axis")),
                        theme=None, config={"displayModeBar": False}, key=f"dist_{key}")
        also = sorted([(e, emo(e), float(p)) for e, p in zip(EMOS, r["presence"]) if p > .5 and e != top],
                      key=lambda x: -x[2])
        inner = ui.chips(M, also) if also else f'<span class="af-muted">{ui.esc(tr("also_feel_none"))}</span>'
        st.markdown(ui.card(inner + f'<div class="af-muted" style="margin-top:.6rem">{ui.esc(tr("also_feel_help"))}</div>',
                            tr("also_feel")), unsafe_allow_html=True)
        items = [(k, i18n.sentiment(L, k), sent[k]) for k in ("positive", "negative", "other")]
        st.markdown(ui.card(ui.stat_tiles(M, items), tr("sentiment_summary")), unsafe_allow_html=True)

    t_att, t_aes, t_det = st.tabs([tr("tab_attention"), tr("tab_aesthetics"), tr("tab_details")])
    with t_att:
        a, b = st.columns([7, 4], gap="large")
        with b:
            sel = st.selectbox(tr("attention_emotion"), EMOS, index=EMOS.index(top), format_func=emo,
                               key=f"att_e_{key}")
            alpha = st.slider(tr("attention_opacity"), 0.0, 1.0, 0.55, 0.05, key="att_alpha")
            st.caption(tr("attention_crop"))
        with a:
            st.image(inf.attention_overlay(img, r["maps"][EMOS.index(sel)], alpha), width="stretch")
            st.caption(tr("attention_caption"))
    with t_aes:
        aesthetics(r["attrs"], r["description"])
    with t_det:
        tops = ", ".join(f"s{s} {emo(EMOS[int(p.argmax())])}" for s, p in zip(inf.SEEDS, r["probs_per_seed"]))
        st.markdown(ui.card(f'<div class="af-desc">{ui.esc(i18n.description(L, r["description"]))}</div>',
                            tr("description")), unsafe_allow_html=True)
        st.caption(tr("per_seed", items=tops))
        if example:
            ann = np.array([example["annotators"][e] for e in EMOS])
            st.markdown(f'<div class="af-section">{ui.esc(tr("annotators"))}</div>', unsafe_allow_html=True)
            st.caption(tr("annotators_help"))
            st.plotly_chart(ui.distribution_chart(M, EMOS, [emo(e) for e in EMOS], ann,
                                                  [i18n.sentiment(L, inf.SENTIMENT[e]) for e in EMOS], tr("prob_axis"),
                                                  height=320), theme=None, config={"displayModeBar": False},
                            key=f"ann_{key}")


def aesthetics(attrs, description):
    names = list(inf.NAMES)
    pct = inf.percentiles(attrs, PCT)
    val = dict(zip(names, attrs)); pc = dict(zip(names, pct))
    st.caption(tr("aesthetics_intro", n=PCT["n_train"]))
    rows = lambda keys: ui.attr_rows([(i18n.attribute(L, k), fmt_value(k, val[k]), pct_label(pc[k]), pc[k]) for k in keys])  # noqa: E731
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(ui.card(rows(GROUPS["group_luminance"]), tr("group_luminance")), unsafe_allow_html=True)
        st.markdown(ui.card(rows(GROUPS["group_contrast"]), tr("group_contrast")), unsafe_allow_html=True)
    with c2:
        st.markdown(ui.card(rows(GROUPS["group_colour"]), tr("group_colour")), unsafe_allow_html=True)
    with c3:
        e = GROUPS["group_edges"]
        st.markdown(ui.card(ui.quadrants(M, [val[k] for k in e[1:]]) + rows(e), tr("group_edges")),
                    unsafe_allow_html=True)
    hb = [f"hue_bin_{i}" for i in range(8)]
    h1, h2 = st.columns([5, 7], gap="medium")
    with h1:
        st.markdown(f'<div class="af-section">{ui.esc(tr("group_hue"))}</div>', unsafe_allow_html=True)
        st.caption(tr("hue_help"))
        st.plotly_chart(ui.hue_wheel(M, [float(val[k]) for k in hb], [i18n.attribute(L, k) for k in hb]),
                        theme=None, config={"displayModeBar": False}, key="hue")
    with h2:
        st.markdown(ui.card(rows(hb)), unsafe_allow_html=True)
    st.markdown(ui.card(f'<div class="af-desc">{ui.esc(i18n.description(L, description))}</div>', tr("description")),
                unsafe_allow_html=True)


# ---------------------------------------------------------------- compare
def page_compare():
    hero("compare_title", "compare_intro")
    ups = st.file_uploader(tr("compare_upload"), type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                           key="cmp_upload")
    picks = st.multiselect(tr("compare_examples"), list(range(len(EXAMPLES))), max_selections=4,
                           format_func=lambda i: EXAMPLES[i]["painting"], key="cmp_examples")
    items = [(u.name, u.getvalue()) for u in ups or []] + [(EXAMPLES[i]["painting"], file_bytes(EXAMPLES[i]["path"]))
                                                           for i in picks]
    if len(items) > 4:
        st.caption(tr("compare_max"))
        items = items[:4]
    if len(items) < 2:
        st.info(tr("compare_need"))
        return
    results = []
    for name, data in items:
        try:
            results.append((name, data, run(data)))
        except Exception:
            st.error(f'{tr("error_image")} ({name})')
    cols = st.columns(len(results), gap="medium")
    for j, (c, (name, data, r)) in enumerate(zip(cols, results)):
        with c:
            st.image(thumbnail_bytes(data), width="stretch")
            st.markdown(f'<div class="af-meta">{j + 1}. {ui.esc(name)}</div>', unsafe_allow_html=True)
            top = r["top"]
            st.markdown(ui.result_card(M, top, emo(top), float(r["probs"].max()), inf.SENTIMENT[top],
                                       i18n.sentiment(L, inf.SENTIMENT[top]), tr("top_emotion"), tr("probability"),
                                       tr("sentiment"), compact=True), unsafe_allow_html=True)
            st.plotly_chart(ui.distribution_chart(M, EMOS, [emo(e) for e in EMOS], r["probs"],
                                                  [i18n.sentiment(L, inf.SENTIMENT[e]) for e in EMOS], tr("prob_axis"),
                                                  height=320), theme=None, config={"displayModeBar": False},
                            key=f"cmp_{j}")
    st.markdown(f'<div class="af-section">{ui.esc(tr("compare_chart"))}</div>', unsafe_allow_html=True)
    st.plotly_chart(ui.compare_chart(M, EMOS, [emo(e) for e in EMOS],
                                     [(f"{j + 1}. {n}", r["probs"]) for j, (n, _, r) in enumerate(results)]),
                    theme=None, config={"displayModeBar": False}, key="cmp_all")


@st.cache_data(show_spinner=False, max_entries=32)
def thumbnail_bytes(data: bytes, size: int = 480) -> Image.Image:
    im = inf.center_square(inf.load_image(io.BytesIO(data)))
    im.thumbnail((size, size))
    return im


# ---------------------------------------------------------------- about
def page_about():
    hero("about_title")
    R = {r["key"]: r for r in RESULTS["rows"]}
    st.markdown(ui.card(ui.esc(tr("about_method")).replace("\n\n", "<br><br>"), tr("about_method_h")), unsafe_allow_html=True)
    st.markdown(tr("about_ensemble", ens=RESULTS["ensemble_macro_f1_primary"]))
    st.markdown(f"### {tr('about_results_h')}")
    head = "".join(f"<th>{ui.esc(tr(c))}</th>" for c in ("col_model", "col_macro_f1", "col_acc", "col_ap", "col_neg"))
    body = ""
    for k in ("baseline", "b32", "l14_r1", "primary"):
        r = R[k]
        cells = "".join(f"<td>{r[m][0]:.3f} ± {r[m][1]:.3f}</td>" for m in ("macro_f1", "acc", "anger_fear_ap", "neg_f1"))
        style = ' style="font-weight:600"' if k == "primary" else ""
        body += f"<tr{style}><td>{ui.esc(tr('row_' + k))}</td>{cells}</tr>"
    table = (f'<table class="af-table" style="width:100%;border-collapse:collapse"><thead><tr>{head}</tr></thead>'
             f"<tbody>{body}</tbody></table>"
             "<style>.af-table th,.af-table td{text-align:left;padding:.55rem .6rem;border-bottom:1px solid var(--af-border);"
             "font-variant-numeric:tabular-nums}.af-table th{color:var(--af-muted);font-weight:600;font-size:.85rem}</style>")
    st.markdown(ui.card(table), unsafe_allow_html=True)
    st.caption(tr("about_results_note", n=RESULTS["n_test"]))
    b = RESULTS["primary_vs_baseline_macro_f1"]
    st.caption(tr("about_boot", d=b["diff"], lo=b["ci"][0], hi=b["ci"][1]))
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown(ui.card(ui.esc(tr("about_anger", n=RESULTS["anger_test_support"], f1=R["primary"]["anger_f1"][0],
                                      sd=R["primary"]["anger_f1"][1])), tr("about_anger_h")), unsafe_allow_html=True)
    with a2:
        st.markdown(ui.card(ui.esc(tr("about_subjective")), tr("about_subjective_h")), unsafe_allow_html=True)
    st.markdown(ui.card(ui.esc(tr("about_context")) + f'<div class="af-muted" style="margin-top:.6rem">{ui.esc(tr("about_privacy"))}</div>',
                        tr("about_context_h")), unsafe_allow_html=True)


{"analyse": page_analyse, "compare": page_compare, "about": page_about}[ss.page]()
with st.sidebar:
    if "engine_ready" in ss:
        st.caption(tr("device", device=engine().device.type.upper()))
