"""ArtFAE-Net bilingual demo (English / 繁體中文).

run:  streamlit run app/streamlit_app.py                        (full: live inference, needs torch + weights)
      APP_MODE=gallery streamlit run app/streamlit_app.py       (gallery: the 12 precomputed examples, no torch)
      APP_MODE=cloud_live streamlit run app/streamlit_app.py    (Streamlit Cloud: precomputed examples + live
                                                                 CPU inference for uploads, CLIP vision tower only)

APP_MODE comes from the environment, else from Streamlit secrets, else "full". In gallery mode torch,
transformers and CLIP are never imported and uploads are disabled. In gallery and cloud_live the examples'
results come from app/assets/precomputed/gallery.json (app/precompute_gallery.py). Uploads (full, cloud_live)
are decoded in memory (core.load_upload); they are never written to disk, and file names are not logged.
"""
import hashlib
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402  (torch-free; puts the repository's src/ on sys.path)
import i18n  # noqa: E402
import ui  # noqa: E402

APP_MODE = core.app_mode(st.secrets)
GALLERY = APP_MODE == "gallery"          # examples only, torch-free
CLOUD_LIVE = APP_MODE == "cloud_live"    # examples precomputed, uploads live on CPU
PRECOMPUTED = GALLERY or CLOUD_LIVE      # the examples' results come from gallery.json
UPLOADS = not GALLERY

# Errors behind the friendly "error_image" / "error_model" messages go to stderr with their traceback (the Streamlit
# Cloud log), never to the page. One handler, even though Streamlit re-executes this script on every interaction.
LOG = logging.getLogger("artfae.app")
if not LOG.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    LOG.addHandler(_h)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
if GALLERY:
    import core as inf  # noqa: E402  (same helper names as inference, without torch)
else:
    import inference as inf  # noqa: E402
if PRECOMPUTED:
    import gallery as precomputed  # noqa: E402  (not "gallery": that name is the gallery section below)

EMOS = inf.EMOTIONS
GROUPS = {
    "group_luminance": ["luminance", "value", "dark", "bright"],
    "group_colour": ["red", "green", "blue", "saturation", "saturation_var", "warm", "cool", "hue_sin", "hue_cos"],
    "group_contrast": ["luminance_std", "contrast_p95_p05"],
    "group_edges": ["edge_density", "edge_tl", "edge_tr", "edge_bl", "edge_br"],
}
SHARES = {"warm", "cool", "dark", "bright", "edge_density", "edge_tl", "edge_tr", "edge_bl", "edge_br",
          *(f"hue_bin_{i}" for i in range(8))}

st.set_page_config(page_title=i18n.t("en", "app_title"), layout="wide", initial_sidebar_state="auto")
ss = st.session_state
ss.setdefault("lang", "en")
ss.setdefault("page", "analyse")
ss.setdefault("source", None)
ss.setdefault("model", inf.MODES[0])  # r1 = default
ss.setdefault("prior", False)  # optional prior correction, R1 only, off by default
L = ss.lang  # the header pill (key "lang") updates this before the rerun starts
tr = lambda key, **kw: i18n.t(L, key, **kw)  # noqa: E731
emo = lambda e: i18n.emotion(L, e)  # noqa: E731
M = ui.mode(getattr(st.context.theme, "type", None))


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False)
def engine():
    if CLOUD_LIVE:  # ~1.5 GB: CLIP vision tower (fp32, CPU) + the 10 heads; shared by all sessions
        return inf.load_engine(device="cpu", vision_only=True)
    return inf.load_engine()


@st.cache_data(show_spinner=False)
def assets():
    return inf.load_json("attr_percentiles.json"), inf.load_examples(), inf.load_json("results.json")


@st.cache_data(show_spinner=False, max_entries=64)
def analyse_bytes(data: bytes, mode: str, prior: bool, app_mode: str = APP_MODE) -> dict:  # app_mode: cache key
    if PRECOMPUTED and (GALLERY or precomputed.has(data)):
        return precomputed.result(data, mode=mode, prior_correction=prior)
    return inf.predict(engine(), core.load_upload(data), mode=mode, prior_correction=prior)


def needs_engine(data: bytes) -> bool:
    return not (PRECOMPUTED and (GALLERY or precomputed.has(data)))


@st.cache_data(show_spinner=False)
def thumb_uri(path: str) -> str:
    return ui.img_data_uri(inf.load_image(path), max_side=520, quality=82)


@st.cache_data(show_spinner=False, max_entries=16)
def artwork_uri(data: bytes) -> str:
    return ui.img_data_uri(core.load_upload(data), max_side=1200)


def file_bytes(path) -> bytes:
    return Path(path).read_bytes()


PCT, EXAMPLES, RESULTS = assets()


def parts(ex):
    return ui.artwork_parts(ex["artist"], ex["painting"], tr("untitled"))


def caption(ex):
    return ui.caption_text(*parts(ex))


UPLOAD_TYPES = core.upload_types() if UPLOADS else []  # + HEIC/HEIF when pillow-heif is installed
FORMATS = ("、" if L == "zh" else ", ").join(["JPG", "PNG", "WebP"] + (["HEIC"] if "heic" in UPLOAD_TYPES else []))
st.markdown(ui.css(M, tr("drop_title"), tr("drop_sub", formats=FORMATS), tr("browse")), unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar: navigation + model
with st.sidebar:
    with st.container(key="nav_box"):
        st.radio(tr("page"), ["analyse", "compare", "about"], format_func=lambda k: tr(f"page_{k}"), key="page")
    with st.container(key="model_box"):
        st.radio(tr("model"), list(inf.MODES), format_func=lambda k: tr(f"model_{k}_plain"), key="model",
                 help="\n\n".join(f"**{tr(f'model_{k}_plain')}**: {tr(f'model_{k}_help')}" for k in inf.MODES))
    st.caption(tr(f"model_{ss.model}_help"))
    if ss.model == "r1":
        with st.expander(tr("advanced")):
            st.checkbox(tr("prior_label"), key="prior", help=tr("prior_help"))
            st.caption(tr("prior_plain"))
    st.divider()
    st.caption(tr("sidebar_note"))

# ---------------------------------------------------------------- header: name, tagline, language pill
with st.container(key="af_header"):
    h1, h2 = st.columns([10, 2], vertical_alignment="center")
    with h1:
        st.markdown(f'<div class="af-brand"><h1>{ui.esc(tr("app_title"))}</h1><p>{ui.esc(tr("app_tagline"))}</p></div>',
                    unsafe_allow_html=True)
    with h2:
        with st.container(key="lang_box"):
            st.radio(tr("language"), list(i18n.LANGS), format_func=i18n.LANG_SHORT.get, key="lang", horizontal=True,
                     label_visibility="collapsed")
    st.markdown('<div class="af-rule"></div>', unsafe_allow_html=True)


def section(title, lead=None):
    st.markdown(f'<div class="af-h2">{ui.esc(title)}</div>' + (f'<p class="af-lead">{ui.esc(lead)}</p>' if lead else ""),
                unsafe_allow_html=True)


def ensure_engine():
    if "engine_ready" not in ss:
        with st.spinner(tr("loading_models")):
            engine()
        ss.engine_ready = True


def run(data: bytes):
    args = (data, ss.model, ss.model == "r1" and bool(ss.prior), APP_MODE)
    if not needs_engine(data):  # precomputed example: nothing to load
        return analyse_bytes(*args)
    ensure_engine()
    with st.spinner(tr("spinner_cloud" if CLOUD_LIVE else "spinner")):
        return analyse_bytes(*args)


def analyse_checked(data: bytes, what: str):
    """Decode (in memory), then run the model; failures are logged with traceback and reported separately.
    what names the source for the log: an example's file name, or "upload" (an upload's file name is never logged).

    -> ("ok", img, r) | ("too_large", None, None): over 10 MB or 50 megapixels (error_too_large)
                      | ("image", None, None): not decodable as an image (error_image)
                      | ("model", None, None): weights missing, model loading or inference failed (error_model)
    """
    try:
        img = core.load_upload(data)
    except core.UploadTooLarge:
        LOG.warning("upload rejected as too large: source=%s bytes=%d app_mode=%s", what, len(data), APP_MODE)
        return "too_large", None, None
    except Exception:
        LOG.exception("image decoding failed: source=%s bytes=%d app_mode=%s cwd=%s", what, len(data), APP_MODE,
                      Path.cwd())
        return "image", None, None
    stage = "inference" if not needs_engine(data) or "engine_ready" in ss else "model loading"
    try:
        r = run(data)
    except Exception:
        LOG.exception("%s failed: source=%s app_mode=%s model=%s prior=%s cwd=%s", stage, what, APP_MODE,
                      ss.model, ss.prior, Path.cwd())
        return "model", None, None
    return "ok", img, r


def fmt_value(name, x):
    return f"{x * 100:.1f}%" if name in SHARES else f"{x:.3f}"


def pct_label(p):
    return tr("percentile", p=p)


# ---------------------------------------------------------------- analyse
def on_upload():
    ss.source = ("upload", None) if ss.get("upload") is not None else None
    ss.scroll_results = ss.source is not None


def pick_example(i):
    ss.source = ("example", i)
    ss.scroll_results = True


SCROLL_JS = """<script>setTimeout(() => {
  const d = window.parent.document, el = d.querySelector('.st-key-results');
  const calm = window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (el) el.scrollIntoView({behavior: calm ? 'auto' : 'smooth', block: 'start'});
}, 120);</script>"""


def scroll_to_results():
    """After a new pick, bring the results (rendered at the top) into view: the click happens in the gallery below."""
    if ss.pop("scroll_results", False):
        with st.container(key="af_scroll"):
            st.iframe(SCROLL_JS, height=1)  # same-origin iframe, hidden by .st-key-af_scroll (replaces the deprecated components.html)


ERRORS = {"image": "error_image", "too_large": "error_too_large", "model": "error_model"}
KEY_ATTRS = ["luminance", "saturation", "warm", "cool"]  # shown first in Visual Attributes; the rest in an expander


def join_list(items):
    return ("、" if L == "zh" else ", ").join(items)


def scope_html():
    return (f'<div class="af-note">{ui.esc(tr("scope_note", emotions=join_list([emo(e) for e in EMOS])))}'
            f'<div class="af-muted" style="margin-top:.45rem">{ui.esc(tr("upload_privacy"))}</div></div>')


def upload_bar():
    """Top of the Analyse page: the one upload control ("Upload or change artwork") with the scope notice beside it.
    Gallery mode has no uploads: its note and the scope notice instead."""
    with st.container(key="upload_bar"):
        if GALLERY:
            with st.container(key="gallery_mode_note"):
                st.info(tr("gallery_mode_note"))
            st.markdown(scope_html().replace(f'<div class="af-muted" style="margin-top:.45rem">'
                                             f'{ui.esc(tr("upload_privacy"))}</div>', ""), unsafe_allow_html=True)
            return
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            with st.container(key="dropzone_top"):
                st.file_uploader(tr("upload_change"), type=UPLOAD_TYPES, help=tr("upload_help", formats=FORMATS),
                                 key="upload", on_change=on_upload)
        with c2:
            st.markdown(scope_html(), unsafe_allow_html=True)


def gallery():
    section(tr("gallery_title"), tr("gallery_note"))
    present = [e for e in EMOS if any(ex["annotator_top"] == e for ex in EXAMPLES)]
    with st.container(key="gallery_filter"):
        pick = st.pills(tr("gallery_filter"), ["all", *present], default="all", key="gallery_pick",
                        format_func=lambda k: tr("filter_all") if k == "all" else emo(k))
    shown = [i for i, ex in enumerate(EXAMPLES) if pick in (None, "all") or ex["annotator_top"] == pick]
    current = ss.source[1] if ss.source and ss.source[0] == "example" else None
    with st.container(key="gallery"):
        for i in shown:
            ex = EXAMPLES[i]
            artist, title, year = parts(ex)
            cap = ui.caption_text(artist, title, year)
            top = ex["annotator_top"]
            with st.container(key=f"card_{i}"):
                st.markdown(ui.gallery_card(thumb_uri(ex["path"]), cap, artist, title, year, ui.PALETTE[M][top],
                                            tr("card_viewers", emotion=emo(top), pct=f'{ex["annotators"][top] * 100:.0f}%'),
                                            tr("card_selected") if i == current else None), unsafe_allow_html=True)
                st.button(tr("card_label", caption=cap), key=f"ex_{i}", on_click=pick_example, args=(i,),
                          width="stretch")  # stretched over the whole card (ui.css)


def page_analyse():
    upload_bar()
    src = ss.source
    if src is not None and src[0] == "upload" and ss.get("upload") is None:
        src = ss.source = None
    if src is None:  # empty state: the collection under the upload bar
        gallery()
        return
    example = None
    if src[0] == "upload":
        data = ss.upload.getvalue()
    else:
        example = EXAMPLES[src[1]]
        data = file_bytes(example["path"])
    status, img, r = analyse_checked(data, example["file"] if example else "upload")
    if status != "ok":  # see ERRORS; the details are in the log
        st.error(tr(ERRORS[status]))
        gallery()
        return
    results(data, img, r, example)
    gallery()


def card_html(r, compact=False):
    """The result card: the highest predicted proportion, or the dispersed message when the top two are within
    core.DISPERSED_MAX_GAP; then the top emotion's category and the summed proportions per sentiment."""
    probs, top = r["probs"], r["top"]
    (a, pa), (b, pb) = core.top_two(probs)
    dispersed = None
    if core.is_dispersed(probs):
        dispersed = (tr("dispersed_msg", a=emo(a), b=emo(b)), [(a, emo(a), pa), (b, emo(b), pb)], tr("dispersed_note"))
    sent = inf.sentiment_totals(probs)
    return ui.result_card(M, eyebrow=tr("dispersed_eyebrow" if dispersed else "top_emotion"), top_key=top,
                          top_label=emo(top), top_prob=float(probs.max()),
                          category_text=tr("top_category", cat=i18n.sentiment(L, inf.SENTIMENT[top])),
                          totals_label=tr("total_predicted"),
                          totals=[(k, i18n.sentiment(L, k), sent[k]) for k in ("positive", "negative", "other")],
                          dispersed=dispersed, compact=compact, colon=tr("colon"))


def settings_html(r):
    prior = tr("on") if r["prior_correction"] else tr("off")
    text = tr("settings_line", model=tr("model_" + r["mode"]), prior=prior, version=core.APP_VERSION)
    return f'<div class="af-settings">{ui.esc(text)}</div>'


def download_record(r, data, example, key):
    """'Download result (JSON)': the image's SHA-256 (uploads) or the artwork ID (gallery examples), never the image."""
    rec = core.result_record(r, app_mode=APP_MODE, artwork_id=example["artwork_id"] if example else None,
                             image_sha256=None if example else hashlib.sha256(data).hexdigest())
    ident = example["artwork_id"].split("/")[-1] if example else rec["input"]["sha256"][:12]
    st.download_button(tr("download_json"), data=json.dumps(rec, indent=1, ensure_ascii=False),
                       file_name=f"artfae_{r['mode']}_{ident}.json", mime="application/json", help=tr("download_help"),
                       key=f"dl_{key}", on_click="ignore", icon=":material/download:")


def people_card(probs, labels):
    counts = ui.people_counts(probs)
    order = sorted([i for i in range(len(EMOS)) if counts[i]], key=lambda i: (-counts[i], -probs[i]))
    legend = [(EMOS[i], (f"<b>{counts[i]}</b>&nbsp;{ui.esc(labels[i])} · {probs[i] * 100:.1f}%" if L == "en" else
                         f"{ui.esc(labels[i])}&nbsp;<b>{counts[i]}</b>&nbsp;人・{probs[i] * 100:.1f}%")) for i in order]
    zero = ui.zero_people(probs)
    zero_html = ""
    if zero:
        items = " · ".join(f'<span><span class="af-dot" style="--c:{ui.PALETTE[M][EMOS[i]]};display:inline-block" '
                           f'aria-hidden="true"></span> {ui.esc(labels[i])} {probs[i] * 100:.1f}%</span>' for i in zero)
        zero_html = f'<div class="af-zero">{ui.esc(tr("zero_people", items="")).rstrip()} {items}</div>'
    return ui.card(ui.people_block(M, EMOS, labels, counts, legend) + zero_html
                   + f'<div class="af-muted" style="margin-top:.7rem">{ui.esc(tr("ten_people_note"))}</div>',
                   tr("ten_people"))


def results(data, img, r, example):
    key = hashlib.sha1(data).hexdigest()[:10]
    probs, top = r["probs"], r["top"]
    labels = [emo(e) for e in EMOS]
    sentiments = [i18n.sentiment(L, inf.SENTIMENT[e]) for e in EMOS]
    with st.container(key="results"):
        left, right = st.columns([5, 7], gap="large")
        with left:
            with st.container(key="artwork"):
                if example:
                    artist, title, year = parts(example)
                    st.markdown(ui.framed_artwork(artwork_uri(data), caption(example), artist, title, year,
                                                  example["style"]), unsafe_allow_html=True)
                else:
                    st.markdown(ui.framed_artwork(artwork_uri(data), tr("alt_upload")), unsafe_allow_html=True)
        with right:  # the card and the full distribution, next to the artwork and above everything else
            st.markdown(card_html(r), unsafe_allow_html=True)
            st.markdown(settings_html(r), unsafe_allow_html=True)
            st.markdown(f'<div class="af-section" style="margin-top:.4rem">{ui.esc(tr("distribution"))}</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(ui.distribution_chart(M, EMOS, labels, probs, sentiments, tr("prob_axis"), height=340),
                            theme=None, config={"displayModeBar": False}, key=f"dist_{key}")
            download_record(r, data, example, key)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(people_card(probs, labels), unsafe_allow_html=True)
        with c2:
            if r["presence"] is not None:  # R3 only; R1 has no presence head
                also = sorted([(e, emo(e), float(p)) for e, p in zip(EMOS, r["presence"]) if p > .5 and e != top],
                              key=lambda x: -x[2])
                inner = ui.chips(M, also) if also else f'<span class="af-muted">{ui.esc(tr("also_feel_none"))}</span>'
                st.markdown(ui.card(f'<div class="af-muted" style="margin-bottom:.6rem">{ui.esc(tr("also_feel_r3"))}</div>'
                                    + inner + f'<div class="af-muted" style="margin-top:.6rem">{ui.esc(tr("also_feel_help"))}</div>',
                                    tr("also_feel")), unsafe_allow_html=True)
            if example:  # validation artworks carry the real annotator votes
                ann = np.array([example["annotators"][e] for e in EMOS])
                st.markdown(f'<div class="af-section">{ui.esc(tr("vs_title"))}</div>', unsafe_allow_html=True)
                st.caption(tr("vs_note"))
                st.plotly_chart(ui.versus_chart(M, EMOS, labels, probs, ann, tr("vs_model"), tr("vs_people"),
                                                tr("prob_axis")), theme=None, config={"displayModeBar": False},
                                key=f"vs_{key}")
        details_tabs(img, r, key)
    scroll_to_results()


def attention_view(img, r, emotion, alpha, original=False):
    if original:
        return inf.center_square(img).convert("RGB").resize((640, 640), Image.Resampling.LANCZOS)
    return inf.attention_overlay(img, r["maps"][EMOS.index(emotion)], alpha)


def attn_label(text, emotion=None):
    dot = (f'<span class="af-dot" style="--c:{ui.PALETTE[M][emotion]}" aria-hidden="true"></span>' if emotion else "")
    return f'<div class="af-attn-label">{dot}{ui.esc(text)}</div>'


def details_tabs(img, r, key):
    top = r["top"]
    t_att, t_aes, t_det = st.tabs([tr("tab_attention"), tr("tab_aesthetics"), tr("tab_details")])
    with t_att:
        a, b = st.columns([7, 5], gap="large")
        with b:  # the caveat first, beside the heatmap
            st.markdown(f'<div class="af-note" style="--af-accent:#B4583A"><b>{ui.esc(tr("attention_caption"))}</b>'
                        f'{ui.esc(tr("attention_note"))}</div>', unsafe_allow_html=True)
            sel = st.selectbox(tr("attention_emotion"), EMOS, index=EMOS.index(top), format_func=emo,
                               key=f"att_e_{key}")
            original = st.toggle(tr("attention_original"), key="att_original")
            alpha = st.slider(tr("attention_opacity"), 0.0, 1.0, 0.55, 0.05, key="att_alpha", disabled=original)
            st.caption(tr("attention_crop"))
        with a:
            st.markdown(attn_label(tr("attention_original_label")) if original else
                        attn_label(tr("attention_map_label", emotion=emo(sel)), sel), unsafe_allow_html=True)
            st.image(attention_view(img, r, sel, alpha, original), width="stretch",
                     alt=tr("attention_original_label") if original else tr("alt_attention", emotion=emo(sel)))
            st.caption(tr("attention_caption"))
        section(tr("attention_compare_h"))
        second = core.top_two(r["probs"])[1][0]
        p1, p2 = st.columns(2, gap="large")
        for col, k, label, default in ((p1, "a", "attention_compare_a", top), (p2, "b", "attention_compare_b", second)):
            with col:
                e = st.selectbox(tr(label), EMOS, index=EMOS.index(default), format_func=emo, key=f"att_cmp_{k}_{key}")
                st.markdown(attn_label(tr("attention_map_label", emotion=emo(e)), e), unsafe_allow_html=True)
                st.image(attention_view(img, r, e, alpha), width="stretch", alt=tr("alt_attention", emotion=emo(e)))
    with t_aes:
        aesthetics(r["attrs"], r["description"])
    with t_det:
        tops = ", ".join(f"s{s} {emo(EMOS[int(p.argmax())])}" for s, p in zip(inf.SEEDS, r["probs_per_seed"]))
        st.caption(tr("model_used_prior" if r["prior_correction"] else "model_used", name=tr(f"model_{r['mode']}")))
        st.caption(tr("precomputed_time" if r.get("precomputed") else "inference_time", seconds=r["seconds"],
                      device=r["device"].upper()))
        st.caption(tr("per_seed", items=tops))
        st.markdown(ui.card(f'<div class="af-desc">{ui.esc(i18n.description(L, r["description"]))}</div>',
                            tr("description")), unsafe_allow_html=True)


def aesthetics(attrs, description):
    names = list(inf.NAMES)
    pct = inf.percentiles(attrs, PCT)
    val = dict(zip(names, attrs)); pc = dict(zip(names, pct))
    n = PCT["n_train"]
    st.caption(tr("aesthetics_intro", n=n))
    key_rows = [(i18n.attribute(L, k), fmt_value(k, val[k]), tr(f"pct_{k}", p=int(round(pc[k])), n=n), pc[k])
                for k in KEY_ATTRS]
    st.markdown(ui.card(ui.key_attr_rows(key_rows), tr("attrs_key_h")), unsafe_allow_html=True)
    rows = lambda keys: ui.attr_rows([(i18n.attribute(L, k), fmt_value(k, val[k]),  # noqa: E731
                                       pct_label(int(round(pc[k]))), pc[k]) for k in keys])
    with st.expander(tr("attrs_more")):
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
    section(tr("compare_title"), tr("compare_intro"))
    if GALLERY:  # examples only
        st.info(tr("gallery_mode_note"))
        ups = []
    else:
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            ups = st.file_uploader(tr("compare_upload"), type=UPLOAD_TYPES, accept_multiple_files=True,
                                   help=tr("upload_help", formats=FORMATS), key="cmp_upload")
        with c2:
            st.markdown(scope_html(), unsafe_allow_html=True)
    picks = st.multiselect(tr("compare_examples"), list(range(len(EXAMPLES))), max_selections=4,
                           format_func=lambda i: caption(EXAMPLES[i]), key="cmp_examples")
    items = [(u.name, u.getvalue(), "upload", None) for u in ups or []] + \
        [(caption(EXAMPLES[i]), file_bytes(EXAMPLES[i]["path"]), EXAMPLES[i]["file"], EXAMPLES[i]) for i in picks]
    if len(items) > 4:
        st.caption(tr("compare_max"))
        items = items[:4]
    if len(items) < 2:
        st.info(tr("compare_need"))
        return
    results_ = []
    for name, data, what, example in items:
        status, _, r = analyse_checked(data, f"compare {what}")
        if status == "model":  # the same for every item: say it once and stop
            st.error(tr("error_model"))
            return
        if status != "ok":
            st.error(f'{tr(ERRORS[status])} ({name})')
            continue
        results_.append((name, data, r, example))
    labels = [emo(e) for e in EMOS]
    sentiments = [i18n.sentiment(L, inf.SENTIMENT[e]) for e in EMOS]
    with st.container(key=f"cmp_grid_{len(results_)}"):  # 2-4 columns; 2 x 2 on narrow screens, 1 on phones (ui.css)
        for j, (name, data, r, example) in enumerate(results_):
            with st.container(key=f"cmp_item_{j}"):
                st.image(thumbnail_bytes(data), width="stretch", alt=name)
                st.markdown(f'<div class="af-meta">{j + 1}. {ui.esc(name)}</div>', unsafe_allow_html=True)
                st.markdown(card_html(r, compact=True), unsafe_allow_html=True)
                st.markdown(settings_html(r), unsafe_allow_html=True)
                st.plotly_chart(ui.distribution_chart(M, EMOS, labels, r["probs"], sentiments, tr("prob_axis"),
                                                      height=320, font_size=13),
                                theme=None, config={"displayModeBar": False}, key=f"cmp_{j}")
                download_record(r, data, example, f"cmp{j}_{hashlib.sha1(data).hexdigest()[:8]}")
    st.markdown(f'<div class="af-section">{ui.esc(tr("compare_chart"))}</div>', unsafe_allow_html=True)
    st.plotly_chart(ui.compare_chart(M, EMOS, labels, [(f"{j + 1}. {n}", r["probs"]) for j, (n, _, r, _) in
                                                       enumerate(results_)], tr("prob_axis")),
                    theme=None, config={"displayModeBar": False}, key="cmp_all")


@st.cache_data(show_spinner=False, max_entries=32)
def thumbnail_bytes(data: bytes, size: int = 480) -> Image.Image:
    im = inf.center_square(core.load_upload(data))
    im.thumbnail((size, size))
    return im


# ---------------------------------------------------------------- about (restyled in stage 2)
def page_about():
    section(tr("about_title"))
    R = {r["key"]: r for r in RESULTS["rows"]}
    st.markdown(ui.card(ui.esc(tr("about_method")).replace("\n\n", "<br><br>"), tr("about_method_h")), unsafe_allow_html=True)
    st.markdown(tr("about_models"))
    H = RESULTS["holdout"]
    st.markdown(f"### {tr('about_holdout_h')}")
    hcols = ("softce", "macro_f1", "anger_fear_ap", "neg_f1")
    head = "".join(f"<th>{ui.esc(tr(c))}</th>" for c in ("col_model", "col_softce", "col_macro_f1", "col_ap", "col_neg"))
    body = "".join(f"<tr><td>{ui.esc(tr('row_h_' + k))}</td>"
                   + "".join(f"<td>{H['rows'][k][m][0]:.4f} ± {H['rows'][k][m][1]:.4f}</td>" for m in hcols) + "</tr>"
                   for k in inf.MODES)
    st.markdown(ui.card(f'<table class="af-table" style="width:100%;border-collapse:collapse"><thead><tr>{head}</tr>'
                        f"</thead><tbody>{body}</tbody></table>"), unsafe_allow_html=True)
    st.caption(tr("about_holdout_note", n=H["n"]))
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
             "font-variant-numeric:tabular-nums}.af-table th{color:var(--af-muted);font-weight:600;font-size:1rem}</style>")
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
    st.markdown(ui.card(ui.esc(tr("about_changelog")).replace("\n\n", "<br><br>"), tr("about_changelog_h")),
                unsafe_allow_html=True)
    st.markdown(ui.card(ui.esc(tr("about_context")) + f'<div class="af-muted" style="margin-top:.6rem">{ui.esc(tr("about_privacy"))}</div>',
                        tr("about_context_h")), unsafe_allow_html=True)


{"analyse": page_analyse, "compare": page_compare, "about": page_about}[ss.page]()
with st.sidebar:
    if "engine_ready" in ss:
        st.caption(tr("device", device=engine().device.type.upper()))
