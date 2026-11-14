"""Look and feel: CSS (light / dark variables), emotion palette, Plotly charts and HTML components.

Every visible string is passed in by the caller (from i18n); nothing here is user-facing text.
"""
import colorsys
import html

import numpy as np
import plotly.graph_objects as go

FONT = "Noto Sans TC, Microsoft JhengHei, PingFang TC, sans-serif"
SERIF = "'Playfair Display', 'Noto Serif TC', 'PMingLiU', serif"
FONTS_CSS = ("https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700"
             "&family=Noto+Serif+TC:wght@500;700&family=Playfair+Display:wght@500;700&display=swap")

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
THEME = {
    "light": {"bg": "#F7F3EC", "card": "#FFFDF9", "text": "#2B2A28", "muted": "#6F6A62", "border": "#E6DED1",
              "shadow": "0 1px 2px rgba(60,45,30,.06), 0 8px 24px rgba(60,45,30,.08)", "grid": "#E9E2D6",
              "track": "#EFE8DC"},
    "dark": {"bg": "#191816", "card": "#24221F", "text": "#ECE6DC", "muted": "#A79F93", "border": "#37332E",
             "shadow": "0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35)", "grid": "#34302B",
             "track": "#2F2C28"},
}


def mode(theme_type) -> str:
    return "dark" if theme_type == "dark" else "light"


def css(m: str) -> str:
    v = THEME[m]
    return f"""<style>
@import url('{FONTS_CSS}');
:root {{ --af-bg:{v['bg']}; --af-card:{v['card']}; --af-text:{v['text']}; --af-muted:{v['muted']};
        --af-border:{v['border']}; --af-shadow:{v['shadow']}; --af-track:{v['track']}; }}
html, body, .stMarkdown, input, textarea {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: "Material Symbols Rounded" !important; }}
h1, h2, h3, h4, .af-display {{ font-family: {SERIF} !important; letter-spacing: .01em; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }}
.af-hero h1 {{ font-size: 2.3rem; margin: 0 0 .1rem 0; font-weight: 700; }}
.af-hero p {{ color: var(--af-muted); margin: 0 0 1.4rem 0; font-size: 1.05rem; }}
.af-card {{ background: var(--af-card); border: 1px solid var(--af-border); border-radius: 18px;
           box-shadow: var(--af-shadow); padding: 1.4rem 1.6rem; margin-bottom: 1.1rem; color: var(--af-text); }}
.af-result {{ border-left: 7px solid var(--af-accent); padding: 1.7rem 1.9rem; }}
.af-eyebrow {{ text-transform: uppercase; letter-spacing: .14em; font-size: .74rem; color: var(--af-muted);
              font-weight: 600; }}
.af-top {{ font-family: {SERIF}; font-size: 3rem; line-height: 1.1; margin: .3rem 0 .2rem 0;
          color: var(--af-accent); font-weight: 700; }}
.af-compact .af-top {{ font-size: clamp(1.5rem, 2.4vw, 2.2rem); }}
.af-compact.af-result {{ padding: 1.2rem 1.3rem; }}
.af-top {{ overflow-wrap: normal; word-break: keep-all; hyphens: none; }}
.af-prob {{ font-size: 1.35rem; font-weight: 500; }}
.af-prob span {{ color: var(--af-muted); font-size: .95rem; font-weight: 400; margin-left: .35rem; }}
.af-bar {{ height: 8px; border-radius: 99px; background: var(--af-track); margin: .9rem 0 1.1rem 0; overflow: hidden; }}
.af-bar > div {{ height: 100%; border-radius: 99px; background: var(--af-accent); }}
.af-pill {{ display: inline-block; padding: .28rem .8rem; border-radius: 99px; font-size: .88rem; font-weight: 500;
           border: 1px solid var(--af-pill); color: var(--af-pill); }}
.af-chip {{ display: inline-block; margin: .2rem .35rem .2rem 0; padding: .32rem .8rem; border-radius: 99px;
           font-size: .9rem; background: var(--af-track); color: var(--af-text); border-left: 4px solid var(--c); }}
.af-chip b {{ font-weight: 600; margin-left: .35rem; color: var(--af-muted); }}
.af-section {{ font-family: {SERIF}; font-size: 1.2rem; margin: 0 0 .6rem 0; font-weight: 600; }}
.af-muted {{ color: var(--af-muted); font-size: .88rem; }}
.af-stats {{ display: flex; gap: .8rem; flex-wrap: wrap; }}
.af-stat {{ flex: 1 1 0; min-width: 110px; padding: .7rem .9rem; border-radius: 14px; background: var(--af-track);
           border-top: 4px solid var(--c); }}
.af-stat .v {{ font-size: 1.5rem; font-weight: 600; }} .af-stat .k {{ color: var(--af-muted); font-size: .85rem; }}
.af-attr {{ display: flex; justify-content: space-between; align-items: baseline; padding: .42rem 0;
           border-bottom: 1px dashed var(--af-border); gap: .8rem; }}
.af-attr:last-child {{ border-bottom: none; }}
.af-attr .n {{ font-size: .92rem; }} .af-attr .v {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
.af-attr .p {{ color: var(--af-muted); font-size: .8rem; white-space: nowrap; }}
.af-pbar {{ height: 4px; border-radius: 99px; background: var(--af-track); margin-top: .25rem; }}
.af-pbar > div {{ height: 100%; border-radius: 99px; background: var(--af-muted); opacity: .7; }}
.af-quad {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; width: 120px; margin: .6rem 0 .2rem 0; }}
.af-quad div {{ height: 44px; border-radius: 6px; display: flex; align-items: center; justify-content: center;
               font-size: .78rem; color: #fff; font-variant-numeric: tabular-nums; }}
.af-desc {{ font-family: {SERIF}; font-size: 1.15rem; line-height: 1.7; }}
.af-meta {{ color: var(--af-muted); font-size: .85rem; margin-top: .35rem; }}
.af-thumb-cap {{ font-size: .78rem; color: var(--af-muted); line-height: 1.25; min-height: 2.1em; }}
[data-testid="stImage"] img {{ border-radius: 12px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--af-border); }}
div[data-testid="stTabs"] button p {{ font-size: 1rem; }}
@media (max-width: 1100px) {{ .af-top {{ font-size: 2.4rem; }} }}
</style>"""


def esc(s) -> str:
    return html.escape(str(s))


def result_card(m, emotion_key, emotion_label, prob, sentiment_key, sentiment_label, eyebrow, prob_word,
                sentiment_word, compact=False) -> str:
    c = PALETTE[m][emotion_key]
    sc = SENTIMENT_COLOURS[m][sentiment_key]
    return (f'<div class="af-card af-result{" af-compact" if compact else ""}" style="--af-accent:{c}">'
            f'<div class="af-eyebrow">{esc(eyebrow)}</div>'
            f'<div class="af-top">{esc(emotion_label)}</div>'
            f'<div class="af-prob">{prob * 100:.1f}%<span>{esc(prob_word)}</span></div>'
            f'<div class="af-bar"><div style="width:{prob * 100:.1f}%"></div></div>'
            f'<span class="af-eyebrow">{esc(sentiment_word)}</span>&nbsp; '
            f'<span class="af-pill" style="--af-pill:{sc}">{esc(sentiment_label)}</span></div>')


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


def quadrants(m, values) -> str:
    """values: edge densities tl, tr, bl, br in [0, 1]."""
    top = max(max(values), 1e-6)
    base = np.array([0x36, 0x47, 0x7A]) if m == "light" else np.array([0x8E, 0x9F, 0xDB])
    cells = []
    for v in values:
        a = .25 + .75 * v / top
        cells.append(f'<div style="background:rgba({base[0]},{base[1]},{base[2]},{a:.2f})">{v * 100:.1f}%</div>')
    return '<div class="af-quad">' + "".join(cells) + "</div>"


def _layout(fig, m, height, **kw):
    v = THEME[m]
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=v["text"], size=14),
                      hoverlabel=dict(font_family=FONT), showlegend=False, **kw)
    return fig


def distribution_chart(m, keys, labels, probs, sentiments, axis_title, height=360):
    """Horizontal bars, sorted (largest on top), coloured by emotion within sentiment families."""
    order = np.argsort(probs)
    fig = go.Figure(go.Bar(
        x=[probs[i] for i in order], y=[labels[i] for i in order], orientation="h",
        marker=dict(color=[PALETTE[m][keys[i]] for i in order], line=dict(width=0)),
        text=[f"{probs[i] * 100:.1f}%" for i in order], textposition="outside", cliponaxis=False,
        customdata=[sentiments[i] for i in order],
        hovertemplate="%{y} · %{customdata}<br>%{x:.1%}<extra></extra>"))
    _layout(fig, m, height, bargap=.32)
    fig.update_xaxes(title=dict(text=axis_title, standoff=8), tickformat=".0%", gridcolor=THEME[m]["grid"],
                     zeroline=False, range=[0, max(probs) * 1.22], automargin=True)
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


def compare_chart(m, keys, labels, series, height=420):
    """series: [(name, probs)] -> grouped bars per emotion, one shade per image."""
    shades = [1.0, .72, .5, .34]
    fig = go.Figure()
    for j, (name, p) in enumerate(series):
        fig.add_trace(go.Bar(x=labels, y=p, name=name, marker=dict(color=[PALETTE[m][k] for k in keys],
                                                                   opacity=shades[j], line=dict(width=0)),
                             text=[str(j + 1)] * len(keys), textposition="outside",
                             hovertemplate=f"{name}<br>%{{x}}: %{{y:.1%}}<extra></extra>"))
    _layout(fig, m, height, barmode="group", bargap=.25)
    fig.update_yaxes(tickformat=".0%", gridcolor=THEME[m]["grid"], zeroline=False, automargin=True)
    fig.update_xaxes(showgrid=False, automargin=True)
    return fig
