"""Look and feel: gallery-style CSS (light "paper" / dark "charcoal" variables), emotion palette, Plotly charts and
HTML components.

Every visible string is passed in by the caller (from i18n); nothing here is user-facing text.
"""
import base64
import colorsys
import html
import io
import re

import numpy as np
import plotly.graph_objects as go

FONT = "Inter, 'Noto Sans TC', 'Microsoft JhengHei', 'PingFang TC', sans-serif"
SERIF = "'Playfair Display', 'Noto Serif TC', 'PMingLiU', serif"
FONTS_CSS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
             "&family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@500;700"
             "&family=Playfair+Display:wght@500;600;700&display=swap")

# warm for positive, cool / dark for negative, grey for something_else; lighter variants for dark mode
PALETTE = {
    "light": {"amusement": "#D9A23A", "awe": "#B4583A", "contentment": "#C99A64", "excitement": "#E07650",
              "anger": "#7E2440", "disgust": "#556338", "fear": "#36477A", "sadness": "#4C78A8",
              "something_else": "#8F8B85"},
    "dark": {"amusement": "#F0C060", "awe": "#E08560", "contentment": "#E3BC8A", "excitement": "#F49A74",
             "anger": "#D0607E", "disgust": "#9DB06C", "fear": "#8E9FDB", "sadness": "#7FAAD8",
             "something_else": "#AAA59E"},
}
SENTIMENT_COLOURS = {"light": {"positive": "#C98A3E", "negative": "#43557F", "other": "#8F8B85"},
                     "dark": {"positive": "#E6AE68", "negative": "#8EA3D8", "other": "#AAA59E"}}
# muted (explanatory) text is >= 7:1 against bg, card and track in each mode (WCAG AA needs 4.5:1; tested)
THEME = {
    "light": {"bg": "#F7F3EC", "card": "#FFFDF9", "text": "#2B2A28", "muted": "#4A443D", "border": "#E3D9CA",
              "accent": "#8C5A3C", "frame": "#3B3128", "mat": "#FBF8F2", "line": "#C9BBA5",
              "shadow": "0 1px 2px rgba(60,45,30,.06), 0 10px 30px rgba(60,45,30,.10)",
              "lift": "0 2px 4px rgba(60,45,30,.08), 0 18px 40px rgba(60,45,30,.16)",
              "grid": "#E9E2D6", "track": "#EFE8DC"},
    "dark": {"bg": "#191816", "card": "#242220", "text": "#EEE8DE", "muted": "#CFC6B9", "border": "#3A3631",
             "accent": "#D9A36A", "frame": "#0E0D0C", "mat": "#211F1D", "line": "#5A5249",
             "shadow": "0 1px 2px rgba(0,0,0,.45), 0 10px 30px rgba(0,0,0,.40)",
             "lift": "0 2px 4px rgba(0,0,0,.5), 0 18px 44px rgba(0,0,0,.55)",
             "grid": "#34302B", "track": "#2F2C28"},
}


def mode(theme_type) -> str:
    return "dark" if theme_type == "dark" else "light"


def css_str(s) -> str:
    """A Python string as a CSS string literal (for localised ::before / ::after content)."""
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def css(m: str, drop_title: str = "", drop_sub: str = "", browse: str = "") -> str:
    v = THEME[m]
    return f"""<style>
@import url('{FONTS_CSS}');
:root {{ --af-bg:{v['bg']}; --af-card:{v['card']}; --af-text:{v['text']}; --af-muted:{v['muted']};
        --af-border:{v['border']}; --af-shadow:{v['shadow']}; --af-lift:{v['lift']}; --af-track:{v['track']};
        --af-accent:{v['accent']}; --af-frame:{v['frame']}; --af-mat:{v['mat']}; --af-line:{v['line']}; }}
html, body, .stApp, .stMarkdown, input, textarea, button, [data-testid="stWidgetLabel"] {{ font-family: {FONT}; }}
html {{ font-size: 16px; }}
.stApp :is(p, span, label, li, div, button, input, textarea, td, th, summary, figcaption):not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{ font-family: {FONT}; }}
.stApp :is(h1, h2, h3, h4, .af-display, .af-top, .af-section, .af-h2, .af-cap .t, .af-figcap .t, .af-desc),
.stApp :is(h1, h2, h3, h4) * {{ font-family: {SERIF} !important; }}
.stApp {{ background: var(--af-bg); color: var(--af-text); }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: "Material Symbols Rounded" !important; }}
h1, h2, h3, h4, .af-display {{ font-family: {SERIF} !important; letter-spacing: .005em; color: var(--af-text); }}
.block-container, [data-testid="stMainBlockContainer"] {{ max-width: 1200px; padding: 3.6rem 2rem 5rem 2rem; }}
p, li, .stMarkdown p {{ font-size: 1rem; line-height: 1.6; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ font-size: 1.05rem; line-height: 1.55;
    color: var(--af-muted); }}
/* Streamlit draws captions at opacity .6, which would blend the muted colour below WCAG AA: draw them opaque */
[data-testid="stCaptionContainer"] {{ opacity: 1 !important; }}
[data-testid="stWidgetLabel"] p {{ font-size: 1rem; font-weight: 500; }}
*:focus-visible {{ outline: 3px solid var(--af-accent) !important; outline-offset: 3px; }}

/* ---------- header ---------- */
.st-key-af_header {{ margin-bottom: 1.4rem; }}
.st-key-af_header [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; align-items: center; gap: 1rem; }}
.st-key-af_header [data-testid="stColumn"]:last-child {{ flex: 0 0 auto !important; width: auto !important; min-width: 0; }}
.st-key-af_header [data-testid="stColumn"]:first-child {{ min-width: 0; }}
.af-brand {{ display: flex; align-items: baseline; gap: .9rem; flex-wrap: wrap; }}
.af-brand h1 {{ font-size: clamp(1.9rem, 4vw, 2.6rem); margin: 0; padding: 0; font-weight: 700; line-height: 1.1; }}
.af-brand p {{ margin: 0; color: var(--af-muted); font-size: 1.05rem; }}
.af-rule {{ height: 1px; background: var(--af-border); margin: .9rem 0 0 0; }}

/* ---------- radio groups restyled: language pill, model tiles, navigation ---------- */
.st-key-lang_box [data-testid="stWidgetLabel"] {{ display: none; }}
.st-key-lang_box [role="radiogroup"], .st-key-model_box [role="radiogroup"] {{ gap: 4px; }}
.st-key-lang_box [role="radiogroup"] {{ display: inline-flex; flex-wrap: nowrap; background: var(--af-track);
    border: 1px solid var(--af-border); border-radius: 999px; padding: 4px; }}
.st-key-lang_box label[data-testid="stRadioOption"], .st-key-model_box label[data-testid="stRadioOption"],
.st-key-nav_box label[data-testid="stRadioOption"] {{ margin: 0; cursor: pointer; transition: background .2s, box-shadow .2s, border-color .2s; }}
.st-key-lang_box label[data-testid="stRadioOption"] > div > div:not([data-testid]),
.st-key-model_box label[data-testid="stRadioOption"] > div > div:not([data-testid]),
.st-key-nav_box label[data-testid="stRadioOption"] > div > div:not([data-testid]) {{ display: none; }}
.st-key-lang_box label[data-testid="stRadioOption"] {{ padding: .38rem 1rem; border-radius: 999px; }}
.st-key-lang_box label[data-testid="stRadioOption"] p {{ font-weight: 600; font-size: 1rem; white-space: nowrap; }}
.st-key-lang_box label[data-testid="stRadioOption"][data-selected="true"] {{ background: var(--af-card); box-shadow: 0 1px 4px rgba(0,0,0,.15); }}
.st-key-lang_box label:has(input:focus-visible), .st-key-model_box label:has(input:focus-visible),
.st-key-nav_box label:has(input:focus-visible) {{ outline: 3px solid var(--af-accent); outline-offset: 2px; }}
.st-key-model_box [role="radiogroup"] {{ display: flex; flex-direction: column; background: var(--af-track);
    border: 1px solid var(--af-border); border-radius: 14px; padding: 4px; }}
.st-key-model_box label[data-testid="stRadioOption"] {{ padding: .6rem .8rem; border-radius: 10px; width: 100%; }}
.st-key-model_box label[data-testid="stRadioOption"] p {{ font-size: 1rem; font-weight: 500; }}
.st-key-model_box label[data-testid="stRadioOption"][data-selected="true"] {{ background: var(--af-card);
    box-shadow: inset 4px 0 0 var(--af-accent), 0 1px 4px rgba(0,0,0,.12); }}
.st-key-model_box label[data-testid="stRadioOption"][data-selected="true"] p {{ font-weight: 650; }}
.st-key-nav_box [role="radiogroup"] {{ gap: 2px; }}
.st-key-nav_box label[data-testid="stRadioOption"] {{ padding: .55rem .8rem; border-radius: 10px; width: 100%; }}
.st-key-nav_box label[data-testid="stRadioOption"] p {{ font-size: 1.05rem; }}
.st-key-nav_box label[data-testid="stRadioOption"]:hover {{ background: var(--af-track); }}
.st-key-nav_box label[data-testid="stRadioOption"][data-selected="true"] {{ background: var(--af-track);
    box-shadow: inset 4px 0 0 var(--af-accent); }}
.st-key-nav_box label[data-testid="stRadioOption"][data-selected="true"] p {{ font-weight: 650; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--af-border); }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ font-size: 1rem; line-height: 1.55; }}

/* ---------- drop zone as an empty picture frame ---------- */
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzone"] {{ position: relative; display: flex;
    flex-direction: column; align-items: center; justify-content: center; gap: 1rem; text-align: center;
    background: var(--af-mat); border: 14px solid var(--af-frame); border-radius: 6px; cursor: pointer;
    box-shadow: inset 0 0 0 1px var(--af-border), inset 0 0 0 10px var(--af-mat), inset 0 0 0 12px var(--af-line),
                var(--af-shadow); transition: box-shadow .25s, transform .25s; }}
.st-key-dropzone_hero [data-testid="stFileUploaderDropzone"] {{ min-height: clamp(260px, 42vh, 400px); padding: 2.5rem 1.5rem; }}
.st-key-dropzone_compact [data-testid="stFileUploaderDropzone"] {{ min-height: 150px; padding: 1.4rem; border-width: 10px; }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzone"]:hover {{ box-shadow: inset 0 0 0 1px var(--af-border),
    inset 0 0 0 10px var(--af-mat), inset 0 0 0 12px var(--af-accent), var(--af-lift); }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {{ display: flex; flex-direction: column;
    align-items: center; margin: 0; order: 1; }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzone"] > span {{ order: 2; }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div {{ display: none; }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzoneInstructions"]::before {{ content: {css_str(drop_title)};
    font-family: {SERIF}; font-size: clamp(1.5rem, 3vw, 2.1rem); font-weight: 600; color: var(--af-text); }}
.st-key-dropzone_compact [data-testid="stFileUploaderDropzoneInstructions"]::before {{ font-size: 1.35rem; }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzoneInstructions"]::after {{ content: {css_str(drop_sub)};
    margin-top: .45rem; font-size: 1rem; color: var(--af-muted); }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzone"] button {{ font-size: 0 !important; border-radius: 999px;
    padding: .55rem 1.3rem; background: var(--af-accent); border: none; color: transparent; }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzone"] button::after {{ content: {css_str(browse)};
    font-size: 1rem; font-weight: 600; color: var(--af-bg); }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzone"] button:hover {{ filter: brightness(1.08); }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderDropzone"] button > * {{ display: none; }}
[class*="st-key-dropzone"] [data-testid="stFileUploaderFile"] {{ font-size: 1rem; }}

/* ---------- gallery ---------- */
.af-h2 {{ font-family: {SERIF}; font-size: clamp(1.4rem, 2.6vw, 1.8rem); font-weight: 600; margin: 2.2rem 0 .2rem 0; }}
.af-lead {{ color: var(--af-muted); font-size: 1rem; margin: 0 0 .9rem 0; }}
.st-key-gallery {{ display: grid !important; grid-template-columns: repeat(auto-fill, minmax(178px, 1fr));
    gap: 1.4rem !important; margin-top: .6rem; align-items: stretch !important; grid-auto-rows: 1fr; }}
.st-key-gallery > div {{ width: auto !important; min-width: 0; display: flex; }}
.st-key-gallery > div > [class*="st-key-card_"] {{ flex: 1 1 auto; }}
[class*="st-key-card_"] {{ position: relative; background: var(--af-card); border: 1px solid var(--af-border);
    border-radius: 14px; box-shadow: var(--af-shadow); overflow: hidden; gap: 0 !important; height: 100%;
    transition: transform .22s ease, box-shadow .22s ease; }}
[class*="st-key-card_"]:hover {{ transform: translateY(-5px); box-shadow: var(--af-lift); }}
[class*="st-key-card_"]:has(button:focus-visible) {{ outline: 3px solid var(--af-accent); outline-offset: 3px; }}
[class*="st-key-card_"] .stElementContainer {{ margin: 0; }}
/* the whole card is the button: its (visually hidden) st.button is stretched over image, title and artist line.
   Streamlit sizes a button's element container to its content, which beats inset: 0, hence the explicit size. */
[class*="st-key-card_"] .stElementContainer:has([data-testid="stButton"]) {{ position: absolute; inset: 0; z-index: 3;
    width: 100% !important; height: 100% !important; margin: 0 !important; }}
[class*="st-key-card_"] [data-testid="stButton"], [class*="st-key-card_"] [data-testid="stButton"] button {{
    width: 100% !important; height: 100% !important; min-height: 0 !important; padding: 0 !important; }}
[class*="st-key-card_"] [data-testid="stButton"] button {{ background: transparent !important; border: none !important;
    color: transparent !important; box-shadow: none !important; cursor: pointer; }}
[class*="st-key-card_"] [data-testid="stButton"] button:focus-visible {{ outline: none !important; }}
[class*="st-key-card_"] [data-testid="stButton"] button p {{ position: absolute; width: 1px; height: 1px; overflow: hidden;
    clip: rect(0 0 0 0); white-space: nowrap; }}
.af-thumb {{ position: relative; aspect-ratio: 4 / 5; width: 100%; overflow: hidden; background: var(--af-track); }}
.af-thumb img {{ width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .4s ease; }}
[class*="st-key-card_"]:hover .af-thumb img {{ transform: scale(1.035); }}
.af-cap {{ padding: .75rem .9rem .9rem .9rem; }}
.af-cap .a {{ font-weight: 600; font-size: 1rem; line-height: 1.35; color: var(--af-text); }}
.af-cap .t {{ font-family: {SERIF}; font-style: italic; font-size: 1rem; line-height: 1.4; color: var(--af-text);
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
.af-tag {{ display: inline-flex; align-items: center; gap: .4rem; margin-top: .5rem; font-size: 1rem; color: var(--af-muted); }}
.af-dot {{ width: .7rem; height: .7rem; border-radius: 50%; background: var(--c); flex: 0 0 auto; }}
.af-selected {{ position: absolute; top: .6rem; left: .6rem; z-index: 2; background: var(--af-accent); color: var(--af-bg);
    font-size: .9rem; font-weight: 600; padding: .2rem .65rem; border-radius: 999px; }}
.st-key-gallery_filter [data-testid="stWidgetLabel"] p {{ color: var(--af-muted); }}
.st-key-gallery_filter button {{ font-size: 1rem !important; border-radius: 999px !important; }}

/* ---------- results ---------- */
.st-key-af_scroll {{ display: none; }}
.st-key-results {{ scroll-margin-top: 4.5rem; animation: af-fade .45s ease both; }}
@keyframes af-fade {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
.st-key-results > [data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
.stLayoutWrapper:has(> .st-key-artwork) {{ position: sticky; top: 1.2rem; align-self: flex-start; z-index: 1; }}
.af-frame {{ margin: 0; background: var(--af-frame); padding: 14px; border-radius: 6px; box-shadow: var(--af-shadow); }}
.af-frame .mat {{ background: var(--af-mat); padding: clamp(10px, 2.2vw, 26px); }}
.af-frame img {{ display: block; width: 100%; height: auto; max-height: 72vh; object-fit: contain; }}
.af-figcap {{ margin-top: .9rem; }}
.af-figcap .a {{ font-weight: 600; font-size: 1.05rem; }}
.af-figcap .t {{ font-family: {SERIF}; font-style: italic; font-size: 1.1rem; }}
.af-figcap .s {{ color: var(--af-muted); font-size: 1rem; }}
.af-card {{ background: var(--af-card); border: 1px solid var(--af-border); border-radius: 18px;
           box-shadow: var(--af-shadow); padding: 1.35rem 1.5rem; margin-bottom: 1rem; color: var(--af-text); }}
.af-result {{ border-left: 7px solid var(--af-accent); padding: 1.6rem 1.8rem; }}
.af-eyebrow {{ letter-spacing: .01em; font-size: 1.05rem; color: var(--af-muted); font-weight: 600; }}
.af-top {{ font-family: {SERIF}; font-size: clamp(2.2rem, 4.6vw, 3.2rem); line-height: 1.08; margin: .3rem 0 .25rem 0;
          color: var(--af-text); font-weight: 700; overflow-wrap: normal; word-break: keep-all; hyphens: none; }}
.af-top::before {{ content: ""; display: inline-block; width: .42em; height: .42em; border-radius: 50%;
          background: var(--af-accent); margin-right: .32em; vertical-align: .14em; }}
.af-compact .af-top {{ font-size: clamp(1.5rem, 2.4vw, 2.2rem); }}
.af-compact.af-result {{ padding: 1.2rem 1.3rem; }}
.af-prob {{ font-size: 1.4rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.af-prob span {{ color: var(--af-muted); font-size: 1rem; font-weight: 400; margin-left: .4rem; }}
.af-bar {{ height: 8px; border-radius: 99px; background: var(--af-track); margin: .9rem 0 1.1rem 0; overflow: hidden; }}
.af-bar > div {{ height: 100%; border-radius: 99px; background: var(--af-accent); }}
.af-pill {{ display: inline-block; padding: .28rem .85rem; border-radius: 99px; font-size: 1rem; font-weight: 600;
           border: 1.5px solid var(--af-pill); color: var(--af-text); }}
.af-pill::before {{ content: ""; display: inline-block; width: .6rem; height: .6rem; border-radius: 50%;
           background: var(--af-pill); margin-right: .45rem; }}
.af-chip {{ display: inline-block; margin: .2rem .35rem .2rem 0; padding: .32rem .85rem; border-radius: 99px;
           font-size: 1rem; background: var(--af-track); color: var(--af-text); border-left: 5px solid var(--c); }}
.af-chip b {{ font-weight: 600; margin-left: .35rem; color: var(--af-muted); }}
.af-section {{ font-family: {SERIF}; font-size: 1.25rem; margin: 0 0 .65rem 0; font-weight: 600; }}
.af-muted {{ color: var(--af-muted); font-size: 1.05rem; line-height: 1.55; }}
.af-stats {{ display: flex; gap: .7rem; flex-wrap: wrap; }}
.af-stat {{ flex: 1 1 0; min-width: 96px; padding: .6rem .85rem; border-radius: 12px; background: var(--af-track);
           border-top: 4px solid var(--c); }}
.af-stat .v {{ font-size: 1.35rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.af-stat .k {{ color: var(--af-muted); font-size: 1rem; }}
.af-people {{ display: flex; flex-wrap: wrap; gap: clamp(4px, 1vw, 10px); margin: .3rem 0 .9rem 0; }}
.af-people svg {{ width: clamp(26px, 6.4vw, 40px); height: auto; }}
.af-legend {{ display: flex; flex-wrap: wrap; gap: .4rem 1.1rem; }}
.af-legend span {{ display: inline-flex; align-items: center; gap: .45rem; font-size: 1rem; }}
.af-legend b {{ font-variant-numeric: tabular-nums; }}
.af-attr {{ display: flex; justify-content: space-between; align-items: baseline; padding: .45rem 0;
           border-bottom: 1px dashed var(--af-border); gap: .8rem; }}
.af-attr:last-child {{ border-bottom: none; }}
.af-attr .n {{ font-size: 1rem; }} .af-attr .v {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
.af-attr .p {{ color: var(--af-muted); font-size: 1rem; }}
.af-pbar {{ height: 4px; border-radius: 99px; background: var(--af-track); margin-top: .25rem; }}
.af-pbar > div {{ height: 100%; border-radius: 99px; background: var(--af-muted); opacity: .7; }}
.af-quad {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; width: 132px; margin: .6rem 0 .2rem 0; }}
.af-quad div {{ height: 46px; border-radius: 6px; display: flex; align-items: center; justify-content: center;
               font-size: .95rem; color: #fff; font-variant-numeric: tabular-nums; }}
.af-desc {{ font-family: {SERIF}; font-size: 1.15rem; line-height: 1.7; }}
.af-meta {{ color: var(--af-muted); font-size: 1rem; margin-top: .35rem; }}
[data-testid="stImage"] img {{ border-radius: 12px; }}
div[data-testid="stTabs"] button p, [data-testid="stTab"] p {{ font-size: 1.1rem !important; font-weight: 600; }}
div[data-testid="stTabs"] {{ margin-top: .8rem; }}
[data-testid="stExpander"] summary p {{ font-size: 1rem; font-weight: 600; }}

/* ---------- milestone 1: result card wording, notes, compare grid ---------- */
.af-card, .af-card * {{ overflow-wrap: normal; word-break: normal; hyphens: manual; }}
.af-disp-msg {{ font-family: {SERIF}; font-size: clamp(1.3rem, 2.4vw, 1.6rem); line-height: 1.35; margin: .4rem 0 .5rem 0;
    font-weight: 600; }}
.af-disp-vals {{ font-size: 1.15rem; font-variant-numeric: tabular-nums; margin-bottom: .5rem; }}
.af-disp-vals .sep {{ color: var(--af-muted); margin: 0 .3rem; }}
.af-dv, .af-tv {{ display: inline-flex; align-items: center; gap: .35rem; white-space: nowrap; }}
.af-disp-note {{ color: var(--af-muted); font-size: 1.05rem; line-height: 1.5; margin-bottom: .8rem; }}
.af-cat {{ font-size: 1.05rem; font-weight: 600; margin-top: .2rem; }}
.af-totals {{ font-size: 1.05rem; margin-top: .45rem; line-height: 1.8; font-variant-numeric: tabular-nums; }}
.af-totals .lab {{ color: var(--af-muted); font-weight: 600; margin-right: .2rem; }}
.af-compact .af-disp-msg {{ font-size: 1.15rem; }}
.af-compact .af-disp-vals, .af-compact .af-cat, .af-compact .af-totals {{ font-size: 1rem; }}
.af-zero {{ margin-top: .6rem; color: var(--af-muted); font-size: 1.05rem; line-height: 1.6; }}
.af-zero span {{ white-space: nowrap; }}
.af-kattr {{ padding: .35rem 0 .6rem 0; border-bottom: 1px dashed var(--af-border); }}
.af-kattr:last-child {{ border-bottom: none; }}
.af-kattr .af-attr {{ border-bottom: none; padding-bottom: .1rem; }}
.af-ksent {{ font-size: 1.05rem; color: var(--af-text); line-height: 1.5; }}
.af-note {{ background: var(--af-card); border: 1px solid var(--af-border); border-left: 6px solid var(--af-accent);
    border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .9rem; font-size: 1.05rem; line-height: 1.55; }}
.af-note b {{ display: block; font-size: 1.1rem; margin-bottom: .25rem; }}
.af-attn-label {{ font-weight: 650; font-size: 1.1rem; margin: 0 0 .45rem 0; display: flex; align-items: center; gap: .45rem; }}
.st-key-upload_bar {{ margin-bottom: 1.4rem; }}
.st-key-dropzone_top [data-testid="stFileUploaderDropzone"] {{ min-height: 150px; padding: 1.2rem; border-width: 10px; }}
.st-key-dropzone_top [data-testid="stFileUploaderDropzoneInstructions"]::before {{ font-size: 1.35rem; }}
.st-key-dropzone_top [data-testid="stWidgetLabel"] p {{ font-size: 1.1rem; font-weight: 650; }}
.af-settings {{ color: var(--af-muted); font-size: 1rem; margin: -.4rem 0 .9rem 0; line-height: 1.5; }}
[class*="st-key-cmp_grid"] {{ display: grid !important; gap: 1.3rem !important; align-items: start; }}
.st-key-cmp_grid_2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.st-key-cmp_grid_3, .st-key-cmp_grid_4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}  /* 2 x 2 */
@media (min-width: 1300px) {{ .st-key-cmp_grid_3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
@media (min-width: 1600px) {{ .st-key-cmp_grid_4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }} }}
[class*="st-key-cmp_grid"] > div {{ width: auto !important; min-width: 0; }}
[class*="st-key-cmp_item_"] {{ min-width: 0; }}
[class*="st-key-cmp_item_"] [data-testid="stImage"] img {{ max-height: 300px; object-fit: cover; }}
.af-compact .af-top {{ font-size: clamp(1.45rem, 2.2vw, 1.9rem); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.af-compact .af-totals .lab {{ display: block; }}
.af-compact .af-totals .af-tv {{ display: flex; }}
.af-compact .af-totals .sep {{ display: none; }}
@media (max-width: 560px) {{ [class*="st-key-cmp_grid"] {{ grid-template-columns: minmax(0, 1fr) !important; }} }}

@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
@media (max-width: 640px) {{
  .block-container, [data-testid="stMainBlockContainer"] {{ padding: 3.4rem 16px 4rem 16px; }}
  .st-key-gallery {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem !important; }}
  .af-cap {{ padding: .6rem .7rem .75rem .7rem; }}
  .stLayoutWrapper:has(> .st-key-artwork) {{ position: static; }}
  .af-frame {{ padding: 9px; }}
  .af-frame img {{ max-height: 52vh; }}
  .af-result {{ padding: 1.2rem 1.25rem; }}
  .af-brand p {{ font-size: 1rem; }}
  .af-disp-vals .sep {{ display: none; }}
  .af-disp-vals .af-dv {{ display: flex; }}
  .st-key-lang_box label[data-testid="stRadioOption"] {{ padding: .32rem .75rem; }}
}}
</style>"""


def esc(s) -> str:
    return html.escape(str(s))


def img_data_uri(im, max_side: int = 1100, quality: int = 86) -> str:
    """PIL image -> JPEG data URI (downscaled), for <img> tags with alt text and CSS-controlled framing."""
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- artwork captions from manifest fields
_SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "of", "on", "or", "over", "the", "to",
          "with", "en", "et"}
_PARTICLES = {"de", "du", "des", "la", "le", "van", "von", "der", "den", "di", "da", "del", "y", "ter"}


def _cap(w: str) -> str:
    return w[:1].upper() + w[1:]


def artwork_parts(artist: str, painting: str, untitled: str):
    """('Aldemir Martins', 'White cat 2001') -> ('Aldemir Martins', 'White Cat', '2001').
    WikiArt slug rules: a trailing 4-digit year is split off; a trailing 1-2 digit variant index is dropped;
    'not detected NNN' (no title on WikiArt) becomes `untitled`."""
    words = [w for w in re.split(r"\s+", painting.strip()) if w]
    year = ""
    if words and re.fullmatch(r"1\d{3}|20\d{2}", words[-1]):
        year = words.pop()
    if words and re.fullmatch(r"\d{1,2}", words[-1]):
        words.pop()
    if not words or " ".join(words[:2]).lower() == "not detected":
        title = untitled
    else:
        title = " ".join(_cap(w) if i == 0 or w.lower() not in _SMALL else w.lower() for i, w in enumerate(words))
    a = [w for w in artist.split() if w]
    name = " ".join(_cap(w) if i == 0 or w.lower() not in _PARTICLES else w.lower() for i, w in enumerate(a))
    return name, title, year


def caption_text(artist: str, title: str, year: str) -> str:
    return f"{artist} — {title}" + (f" ({year})" if year else "")


def gallery_card(src: str, alt: str, artist: str, title: str, year: str, tag_colour: str, tag_label: str,
                 selected_label: str | None = None) -> str:
    sel = f'<span class="af-selected">{esc(selected_label)}</span>' if selected_label else ""
    t = esc(title) + (f" ({esc(year)})" if year else "")
    return (f'<div class="af-thumb">{sel}<img src="{src}" alt="{esc(alt)}" loading="lazy"></div>'
            f'<div class="af-cap"><div class="a">{esc(artist)}</div><div class="t">{t}</div>'
            f'<div class="af-tag"><span class="af-dot" style="--c:{tag_colour}" aria-hidden="true"></span>'
            f'{esc(tag_label)}</div></div>')


def framed_artwork(src: str, alt: str, artist: str | None = None, title: str | None = None, year: str = "",
                   sub: str | None = None) -> str:
    cap = ""
    if artist or title or sub:
        t = (f'<div class="t">{esc(title)}' + (f" ({esc(year)})" if year else "") + "</div>") if title else ""
        a = f'<div class="a">{esc(artist)}</div>' if artist else ""
        s = f'<div class="s">{esc(sub)}</div>' if sub else ""
        cap = f'<figcaption class="af-figcap">{a}{t}{s}</figcaption>'
    return (f'<figure style="margin:0"><div class="af-frame"><div class="mat"><img src="{src}" alt="{esc(alt)}"></div>'
            f"</div>{cap}</figure>")


def result_card(m, *, eyebrow, top_key, top_label, top_prob, category_text, totals_label, totals,
                dispersed=None, compact=False, colon=": ") -> str:
    """The main result card. Every colour comes with its label and value.

    Normal: eyebrow ("Highest predicted proportion"), the top emotion and its %, a bar.
    dispersed = (message, [(key, label, prob), (key, label, prob)], note): replaces the single top emotion when the top
    two predicted proportions are within core.DISPERSED_MAX_GAP.
    category_text: "Category of the top emotion: Positive"; totals: [(sentiment_key, label, value)] after totals_label.
    """
    c = PALETTE[m][top_key]
    if dispersed is None:
        body = (f'<div class="af-top">{esc(top_label)}</div>'
                f'<div class="af-prob">{top_prob * 100:.1f}%</div>'
                f'<div class="af-bar"><div style="width:{top_prob * 100:.1f}%"></div></div>')
    else:
        msg, pair, note = dispersed
        vals = " <span class=\"sep\" aria-hidden=\"true\">|</span> ".join(
            f'<span class="af-dv"><span class="af-dot" style="--c:{PALETTE[m][k]}" aria-hidden="true"></span>'
            f'{esc(lab)}{esc(colon)}<b>{p * 100:.1f}%</b></span>' for k, lab, p in pair)
        body = (f'<div class="af-disp-msg">{esc(msg)}</div><div class="af-disp-vals">{vals}</div>'
                f'<div class="af-disp-note">{esc(note)}</div>')
    tot = '<span class="sep" aria-hidden="true"> · </span>'.join(f'<span class="af-tv"><span class="af-dot" style="--c:{SENTIMENT_COLOURS[m][k]}" aria-hidden="true">'
                     f'</span>{esc(lab)} <b>{v * 100:.1f}%</b></span>' for k, lab, v in totals)
    cls = "af-card af-result" + (" af-compact" if compact else "") + (" af-dispersed" if dispersed else "")
    return (f'<div class="{cls}" style="--af-accent:{c}"><div class="af-eyebrow">{esc(eyebrow)}</div>{body}'
            f'<div class="af-cat">{esc(category_text)}</div>'
            f'<div class="af-totals"><span class="lab">{esc(totals_label)}{esc(colon.rstrip())}</span> {tot}</div></div>')


# ---------------------------------------------------------------- "if 10 people saw this"
def people_counts(probs, n: int = 10) -> np.ndarray:
    """Largest-remainder (Hamilton) rounding of n * probs to whole people; ties broken by the larger probability."""
    p = np.asarray(probs, np.float64)
    q = n * p / p.sum()
    c = np.floor(q).astype(int)
    rem = q - c
    for i in sorted(range(len(p)), key=lambda i: (-rem[i], -p[i]))[: n - int(c.sum())]:
        c[i] += 1
    return c


_PERSON = ('<svg viewBox="0 0 24 34" role="img" aria-label="{label}"><title>{label}</title>'
           '<circle cx="12" cy="7" r="5.6" fill="{c}"/><path d="M2.5 33v-8.2a9.5 9.5 0 0 1 19 0V33z" fill="{c}"/></svg>')


def people_block(m, keys, labels, counts, legend_items) -> str:
    """keys/labels/counts per emotion; icons grouped by emotion (largest first), each with its emotion as text
    label; legend_items: [(emotion_key, text)] for emotions with at least one person."""
    order = [list(keys).index(k) for k, _ in legend_items]  # same order as the legend (the caller sorts it)
    icons = "".join(_PERSON.format(c=PALETTE[m][keys[i]], label=esc(labels[i])) for i in order for _ in range(counts[i]))
    legend = "".join(f'<span><span class="af-dot" style="--c:{PALETTE[m][k]}" aria-hidden="true"></span>{t}</span>'
                     for k, t in legend_items)
    return f'<div class="af-people">{icons}</div><div class="af-legend">{legend}</div>'


def zero_people(probs, n: int = 10) -> list:
    """Indices of the emotions that round to 0 of n people (people_counts), largest probability first."""
    c = people_counts(probs, n)
    p = np.asarray(probs, np.float64)
    return sorted([i for i in range(len(p)) if c[i] == 0], key=lambda i: -p[i])


def chips(m, items) -> str:
    """items: [(emotion_key, label, prob)]"""
    return "".join(f'<span class="af-chip" style="--c:{PALETTE[m][k]}">{esc(lab)}<b>{p * 100:.0f}%</b></span>'
                   for k, lab, p in items)


def stat_tiles(m, items) -> str:
    """items: [(sentiment_key, label, value)]"""
    return '<div class="af-stats">' + "".join(
        f'<div class="af-stat" style="--c:{SENTIMENT_COLOURS[m][k]}"><div class="v">{v * 100:.1f}%</div>'
        f'<div class="k">{esc(lab)}</div></div>' for k, lab, v in items) + "</div>"


def card(inner: str, title: str | None = None) -> str:
    head = f'<div class="af-section">{esc(title)}</div>' if title else ""
    return f'<div class="af-card">{head}{inner}</div>'


def attr_rows(rows) -> str:
    """rows: [(label, value_str, percentile_label, percentile)]"""
    return "".join(f'<div><div class="af-attr"><span class="n">{esc(n)}</span><span><span class="v">{esc(v)}</span>'
                   f'&nbsp;&nbsp;<span class="p">{esc(pl)}</span></span></div>'
                   f'<div class="af-pbar"><div style="width:{p:.0f}%"></div></div></div>'
                   for n, v, pl, p in rows)


def key_attr_rows(rows) -> str:
    """rows: [(label, value_str, sentence, percentile)]: the plain-language comparison on its own line."""
    return "".join(f'<div class="af-kattr"><div class="af-attr"><span class="n">{esc(n)}</span><span class="v">{esc(v)}</span>'
                   f'</div><div class="af-ksent">{esc(s)}</div><div class="af-pbar"><div style="width:{p:.0f}%"></div>'
                   f'</div></div>' for n, v, s, p in rows)


def quadrants(m, values) -> str:
    """values: edge densities tl, tr, bl, br in [0, 1]."""
    top = max(max(values), 1e-6)
    base = np.array([0x36, 0x47, 0x7A]) if m == "light" else np.array([0x8E, 0x9F, 0xDB])
    cells = []
    for v in values:
        a = .25 + .75 * v / top
        cells.append(f'<div style="background:rgba({base[0]},{base[1]},{base[2]},{a:.2f})">{v * 100:.1f}%</div>')
    return '<div class="af-quad">' + "".join(cells) + "</div>"


# ---------------------------------------------------------------- charts
def _layout(fig, m, height, **kw):
    v = THEME[m]
    fig.update_layout(height=height, margin=dict(l=10, r=24, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=v["text"], size=15),
                      hoverlabel=dict(font_family=FONT, font_size=15), showlegend=False, **kw)
    return fig


def distribution_chart(m, keys, labels, probs, sentiments, axis_title, height=360, font_size=15):
    """Horizontal bars, sorted (largest on top), coloured by emotion within sentiment families."""
    order = np.argsort(probs)
    fig = go.Figure(go.Bar(
        x=[probs[i] for i in order], y=[labels[i] for i in order], orientation="h",
        marker=dict(color=[PALETTE[m][keys[i]] for i in order], line=dict(width=0)),
        text=[f"{probs[i] * 100:.1f}%" for i in order], textposition="outside", cliponaxis=False,
        customdata=[sentiments[i] for i in order],
        hovertemplate="%{y} · %{customdata}<br>%{x:.1%}<extra></extra>"))
    _layout(fig, m, height, bargap=.32)
    fig.update_layout(font=dict(size=font_size))
    fig.update_xaxes(title=dict(text=axis_title, standoff=8), tickformat=".0%", gridcolor=THEME[m]["grid"],
                     zeroline=False, range=[0, max(probs) * 1.3], automargin=True, nticks=5)
    fig.update_yaxes(showgrid=False, automargin=True, ticksuffix="  ")
    return fig


def versus_chart(m, keys, labels, pred, actual, name_pred, name_actual, axis_title, height=420):
    """Predicted vs. real annotator vote shares: two bars per emotion (model solid, viewers hatched), sorted by the
    viewers' share; a legend names both series, and every bar carries its value as text."""
    order = np.argsort(np.asarray(actual) + 1e-6 * np.asarray(pred))
    y = [labels[i] for i in order]
    cols = [PALETTE[m][keys[i]] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=name_actual, y=y, x=[actual[i] for i in order], orientation="h",
                         marker=dict(color=cols, opacity=.5, pattern=dict(shape="/", fgcolor=THEME[m]["text"],
                                                                          fgopacity=.35, size=7)),
                         text=[f"{actual[i] * 100:.0f}%" for i in order], textposition="outside", cliponaxis=False,
                         hovertemplate=f"{name_actual}<br>%{{y}}: %{{x:.0%}}<extra></extra>"))
    fig.add_trace(go.Bar(name=name_pred, y=y, x=[pred[i] for i in order], orientation="h",
                         marker=dict(color=cols, line=dict(width=0)),
                         text=[f"{pred[i] * 100:.0f}%" for i in order], textposition="outside", cliponaxis=False,
                         hovertemplate=f"{name_pred}<br>%{{y}}: %{{x:.1%}}<extra></extra>"))
    _layout(fig, m, height, barmode="group", bargap=.28, bargroupgap=.08, uniformtext=dict(minsize=13, mode="show"))
    fig.update_traces(textfont=dict(size=14))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="reversed",
                                                   font=dict(size=15)), margin=dict(l=10, r=24, t=40, b=10))
    top = max(max(pred), max(actual))
    fig.update_xaxes(title=dict(text=axis_title, standoff=8), tickformat=".0%", gridcolor=THEME[m]["grid"],
                     zeroline=False, range=[0, top * 1.25], automargin=True)
    fig.update_yaxes(showgrid=False, automargin=True, ticksuffix="  ")
    return fig


def sentiment_bar(m, items, height=70):
    """items: [(sentiment_key, label, value)] as one stacked 100% bar."""
    fig = go.Figure()
    for k, lab, v in items:
        fig.add_trace(go.Bar(x=[v], y=[""], orientation="h", name=lab, marker_color=SENTIMENT_COLOURS[m][k],
                             text=f"{lab} {v * 100:.0f}%" if v > .08 else "", textposition="inside",
                             insidetextanchor="middle", hovertemplate=f"{lab}: %{{x:.1%}}<extra></extra>"))
    _layout(fig, m, height, barmode="stack")
    fig.update_traces(textfont=dict(color="#FFFFFF", family=FONT))
    fig.update_xaxes(visible=False, range=[0, 1]); fig.update_yaxes(visible=False)
    return fig


def hue_wheel(m, hist, labels, height=300):
    """8-bin saturation-weighted hue histogram as a polar bar 'wheel', each wedge in its own hue."""
    centres = [(i + .5) * 45 for i in range(8)]
    cols = ["#%02x%02x%02x" % tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h / 360, .62, .88)) for h in centres]
    fig = go.Figure(go.Barpolar(r=hist, theta=centres, width=[44] * 8, marker_color=cols,
                                marker_line=dict(color=THEME[m]["card"], width=1.5), customdata=labels,
                                hovertemplate="%{customdata}: %{r:.1%}<extra></extra>"))
    _layout(fig, m, height)
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(showticklabels=False, ticks="", showline=False, gridcolor=THEME[m]["grid"]),
                                 angularaxis=dict(direction="counterclockwise", rotation=0, gridcolor=THEME[m]["grid"],
                                                  tickvals=[0, 90, 180, 270], ticktext=["0°", "90°", "180°", "270°"])))
    return fig


SERIES = {"light": ["#8C5A3C", "#36477A", "#556338", "#B4583A"], "dark": ["#E0A86E", "#8E9FDB", "#9DB06C", "#F49A74"]}


def compare_chart(m, keys, labels, series, axis_title="", height=None):
    """series: [(name, probs)] -> horizontal grouped bars: one row per emotion (the label on the axis), one colour per
    image (named in the legend), every bar with its value."""
    fig = go.Figure()
    order = list(range(len(keys)))[::-1]  # first emotion on top
    for j, (name, p) in enumerate(series):
        fig.add_trace(go.Bar(y=[labels[i] for i in order], x=[p[i] for i in order], name=name, orientation="h",
                             marker=dict(color=SERIES[m][j % 4], line=dict(width=0)),
                             text=[f"{p[i] * 100:.0f}%" for i in order], textposition="outside", cliponaxis=False,
                             hovertemplate=f"{name}<br>%{{y}}: %{{x:.1%}}<extra></extra>"))
    height = height or 120 + 26 * len(keys) * max(len(series), 1)
    _layout(fig, m, height, barmode="group", bargap=.22, bargroupgap=.06)
    fig.update_traces(textfont=dict(size=13))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, traceorder="normal",
                                                   font=dict(size=14)), margin=dict(l=10, r=30, t=60, b=10))
    top = max(float(max(p)) for _, p in series)
    fig.update_xaxes(title=dict(text=axis_title, standoff=8), tickformat=".0%", gridcolor=THEME[m]["grid"],
                     zeroline=False, range=[0, top * 1.22], automargin=True)
    fig.update_yaxes(showgrid=False, automargin=True, ticksuffix="  ")
    return fig
