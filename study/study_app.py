"""ArtFAE-Net human-evaluation study (English / 繁體中文). Separate from the demo; reuses its look (app/ui.py) and
emotion names (app/i18n.py), unchanged.

Flow: consent -> participant code (write it down; needed to withdraw) -> instructions + 1 practice trial -> 20 trials + 2 attention checks -> questionnaire -> thank-you.
Each trial: (a) BLIND emotion choice, strength, optional reason; (b) REVEAL R1 "10 viewers" + bars, plausibility;
(c) A/B R1 vs R3 unlabelled, random sides; (d) optional attention overlay rating. Response times per step.

The model outputs are read only through model_outputs(), which the blind step never calls (tests enforce this),
so nothing about the model's answer reaches the browser before the participant submits.

run:  streamlit run study/study_app.py
env:  STUDY_MODE=pilot|production, DATASET_REPO, HF_TOKEN ... (see storage.deployment);  STUDY_DEV=1 enables ?shot=<step> for screenshots only.
"""
import json
import os
import re
import secrets
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

STUDY = Path(__file__).resolve().parent
for d in (STUDY / "vendor", STUDY.parent / "app"):  # Space bundle copies app/ui.py and app/i18n.py into vendor/
    if (d / "ui.py").exists():
        sys.path.insert(0, str(d))
        break
sys.path.insert(0, str(STUDY))
# Streamlit re-runs this script on every interaction but keeps imported modules in sys.modules. After an edit on a
# running server the old storage module stayed loaded ("module 'storage' has no attribute 'new_participant_code'").
# So: if any study module is not the file next to this script, or that file changed since it was loaded, drop all
# of them (they import each other) and import afresh.
_LOCAL = ("common", "assignment", "frozen", "storage", "study_i18n")


def _stale(name):
    m, f = sys.modules.get(name), STUDY / f"{name}.py"
    if m is None:
        return False
    return Path(getattr(m, "__file__", "") or "").resolve() != f or getattr(m, "_af_mtime", None) != f.stat().st_mtime_ns


if any(_stale(n) for n in _LOCAL):
    for n in _LOCAL:
        sys.modules.pop(n, None)
import assignment  # noqa: E402
import frozen  # noqa: E402  (the frozen stimulus set, stimuli_FROZEN.txt)
import i18n as app_i18n  # noqa: E402  (the demo's strings: emotion and language names)
import storage as store  # noqa: E402
import study_i18n as si  # noqa: E402
import ui  # noqa: E402  (the demo's look)
from common import EMOTIONS, SENTIMENT  # noqa: E402

for _n in _LOCAL:
    sys.modules[_n]._af_mtime = (STUDY / f"{_n}.py").stat().st_mtime_ns

APP_VERSION = "study-1.0"
DEV = os.environ.get("STUDY_DEV") == "1"
SCALE = [1, 2, 3, 4, 5]

st.set_page_config(page_title="ArtFAE-Net study", layout="wide", initial_sidebar_state="collapsed")
ss = st.session_state

# Refuse to run with an unsafe or incomplete deployment setting (STUDY_MODE, CONSENT_APPROVED, DATASET_REPO, HF_TOKEN).
CFG = store.deployment()
MODE = CFG["mode"]
if CFG["problems"]:
    st.error(f"{si.t('en', 'config_error')}\n\n{si.t('zh', 'config_error')}")
    st.code("\n".join(CFG["problems"]))
    st.stop()

# Refuse to run unless every stimulus file, precomputed output and the assignment sets match stimuli_FROZEN.txt.
# Checked at the start of every session, before anything is shown, stored or assigned.
if not ss.get("_frozen_ok"):
    _problems = frozen.verify(STUDY)
    if _problems:
        st.error(f"{si.t('en', 'frozen_error')}\n\n{si.t('zh', 'frozen_error')}")
        st.code("\n".join(_problems[:20]) + (f"\n... and {len(_problems) - 20} more" if len(_problems) > 20 else ""))
        st.stop()
    ss._frozen_ok = True
ss.setdefault("lang", "en")
ss.setdefault("phase", "consent")
L = ss.lang
tr = lambda key, **kw: si.t(L, key, **kw)  # noqa: E731
emo = lambda e: app_i18n.emotion(L, e)  # noqa: E731
M = ui.mode(getattr(st.context.theme, "type", None))


# ---------------------------------------------------------------- data (stimuli: metadata only; outputs: lazy)
@st.cache_resource(show_spinner=False)
def _backend(storage_version):  # a new backend object whenever storage.py changes
    return store.from_env(CFG)


def backend():
    return _backend(store._af_mtime)


@st.cache_data(show_spinner=False)
def stimuli():
    return pd.read_csv(STUDY / "stimuli.csv", keep_default_na=False).set_index("stimulus_id")


@lru_cache(maxsize=1)
def _all_outputs():
    return json.loads((STUDY / "precomputed/outputs.json").read_text())["stimuli"]


def model_outputs(sid: str) -> dict:
    """The ONLY way the app reads model outputs. Logged per session so tests can prove the blind step never asks."""
    ss.setdefault("_outputs_read", []).append((ss.get("phase"), ss.get("step"), sid))
    return _all_outputs()[sid]


@st.cache_data(show_spinner=False)
def image_uri(rel: str) -> str:
    return ui.img_data_uri(Image.open(STUDY / rel), max_side=1100)


@st.cache_data(show_spinner=False)
def sets():
    s = stimuli()
    return assignment.make_sets([{"stimulus_id": i, "stratum": r.stratum} for i, r in s[s.role == "stimulus"].iterrows()])


def consent_text() -> str:
    return re.sub(r"<!--.*?-->", "", (STUDY / f"consent_{L}.md").read_text(encoding="utf-8"), flags=re.S).strip()


# ---------------------------------------------------------------- page chrome
EXTRA_CSS = """<style>
.st-key-af_header {{ margin-bottom: 1rem; }}
.af-progress {{ display:flex; align-items:center; gap:1rem; margin: .2rem 0 1.2rem 0; color: var(--af-muted); font-size: 1rem; }}
.af-progress .track {{ flex:1; height:6px; background: var(--af-track); border-radius:99px; overflow:hidden; }}
.af-progress .track > div {{ height:100%; background: var(--af-accent); border-radius:99px; }}
.af-badge {{ display:inline-block; background: var(--af-accent); color: var(--af-bg); font-weight:600; border-radius:99px;
            padding:.15rem .7rem; font-size:.95rem; margin-bottom:.6rem; }}
.af-check {{ border-left: 6px solid var(--af-accent); background: var(--af-card); padding: .9rem 1.1rem; border-radius: 12px;
            margin-bottom: 1rem; font-weight: 600; font-size: 1.05rem; }}
.st-key-emotion_box [role="radiogroup"] {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:.5rem; }}
.st-key-emotion_box label[data-testid="stRadioOption"] {{ border:1px solid var(--af-border); background: var(--af-card);
            border-radius: 12px; padding:.6rem .8rem; margin:0; min-height:3rem; }}
.st-key-emotion_box label[data-testid="stRadioOption"][data-selected="true"] {{ border-color: var(--af-accent);
            box-shadow: inset 0 0 0 2px var(--af-accent); }}
.st-key-emotion_box label[data-testid="stRadioOption"] p {{ font-size: 1.05rem; }}
[class*="st-key-scale_"] [role="radiogroup"] {{ gap: .4rem; flex-wrap: wrap; }}
[class*="st-key-scale_"] label[data-testid="stRadioOption"] {{ border:1px solid var(--af-border); background: var(--af-card);
            border-radius: 999px; padding:.35rem .9rem; margin:0; }}
[class*="st-key-scale_"] label[data-testid="stRadioOption"][data-selected="true"] {{ border-color: var(--af-accent);
            box-shadow: inset 0 0 0 2px var(--af-accent); }}
[class*="st-key-scale_"] label[data-testid="stRadioOption"] p, .stRadio label p, .stCheckbox label p {{ font-size: 1rem; }}
.af-pilot {{ background: #8a5a00; color: #fff; font-weight: 700; font-size: 1.05rem; text-align: center;
            padding: .6rem 1rem; border-radius: 10px; margin: 0 0 .8rem 0; letter-spacing: .01em; }}
.af-pcode {{ font-size: 2.4rem; font-weight: 700; letter-spacing: .12em; text-align: center; margin: .4rem 0 1rem 0;
            font-family: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace; }}
.af-ab-label {{ font-family: 'Playfair Display','Noto Serif TC',serif; font-size: 1.4rem; font-weight: 600; margin-bottom: .2rem; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {{ display:none; }}
@media (max-width: 640px) {{ .st-key-emotion_box [role="radiogroup"] {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
</style>""".replace("{{", "{").replace("}}", "}")

st.markdown(ui.css(M), unsafe_allow_html=True)
st.markdown(EXTRA_CSS, unsafe_allow_html=True)
if MODE == "pilot":
    st.markdown(f'<div class="af-pilot" role="status">{ui.esc(tr("pilot_banner"))}</div>', unsafe_allow_html=True)
with st.container(key="af_header"):
    h1, h2 = st.columns([10, 2], vertical_alignment="center")
    with h1:
        st.markdown(f'<div class="af-brand"><h1>{ui.esc(tr("study_title"))}</h1><p>{ui.esc(tr("study_tagline"))}</p></div>',
                    unsafe_allow_html=True)
    with h2:
        with st.container(key="lang_box"):
            st.radio(tr("language"), list(app_i18n.LANGS), format_func=app_i18n.LANG_SHORT.get, key="lang",
                     horizontal=True, label_visibility="collapsed")
    st.markdown('<div class="af-rule"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def timer(key: str) -> None:
    """Start the response-time clock the first time a step is shown."""
    if ss.get("_t_key") != key:
        ss._t_key, ss._t0 = key, time.monotonic()


def elapsed_ms() -> int:
    return int(round((time.monotonic() - ss._t0) * 1000))


def save(fn, row) -> bool:
    try:
        fn(row)
        return True
    except Exception:  # noqa: BLE001
        st.error(tr("error_storage"))
        return False


def ab_chart(probs, xmax, key):
    """Horizontal bars in the fixed ArtEmis order, identical axis range for A and B, values printed on the bars."""
    import plotly.graph_objects as go
    labels = [emo(e) for e in EMOTIONS]
    fig = go.Figure(go.Bar(x=list(probs)[::-1], y=labels[::-1], orientation="h",
                           marker=dict(color=[ui.PALETTE[M][e] for e in EMOTIONS][::-1], line=dict(width=0)),
                           text=[f"{p * 100:.1f}%" for p in probs][::-1], textposition="outside", cliponaxis=False,
                           hoverinfo="skip"))
    ui._layout(fig, M, 340, bargap=.3)
    fig.update_xaxes(tickformat=".0%", gridcolor=ui.THEME[M]["grid"], zeroline=False, range=[0, xmax], automargin=True)
    fig.update_yaxes(showgrid=False, automargin=True, ticksuffix="  ")
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False, "staticPlot": True}, key=key)


def scale_fmt(lo_key, hi_key):
    return lambda v: tr(lo_key) if v == 1 else tr(hi_key) if v == 5 else str(v)


def progress(trial):
    if trial["kind"] == "practice":
        st.markdown(f'<span class="af-badge">{ui.esc(tr("practice_badge"))}</span>', unsafe_allow_html=True)
        return
    i, n = ss.ti + 1, len(ss.plan)
    st.markdown(f'<div class="af-progress"><span>{ui.esc(tr("progress", i=i, n=n))}</span>'
                f'<div class="track"><div style="width:{100 * ss.ti / len(ss.plan):.0f}%"></div></div></div>',
                unsafe_allow_html=True)


def artwork(trial):
    rec = stimuli().loc[trial["stimulus_id"]]
    st.markdown(ui.framed_artwork(image_uri(rec.image), tr("alt_artwork")), unsafe_allow_html=True)


def current_trial():
    return assignment.practice_trial() if ss.phase == "practice" else ss.plan[ss.ti]


def trial_index():
    return 0 if ss.phase == "practice" else ss.ti + 1


# ---------------------------------------------------------------- trial steps
def step_blind(trial):
    """(a) artwork + questions only. Must not call model_outputs()."""
    timer(f"{trial_index()}:blind")
    left, right = st.columns([6, 5], gap="large")
    with left:
        artwork(trial)
    with right:
        if trial["kind"] == "check":
            st.markdown(f'<div class="af-check">{ui.esc(tr("check_instruction", emotion=emo(trial["expect_emotion"]), strength=trial["expect_strength"]))}</div>',
                        unsafe_allow_html=True)
        with st.form(f"blind_{trial_index()}", border=False):
            with st.container(key="emotion_box"):
                e = st.radio(tr("blind_q"), EMOTIONS, index=None, format_func=emo, key=f"b_emo_{trial_index()}")
            with st.container(key=f"scale_strength_{trial_index()}"):
                s = st.radio(tr("blind_strength"), SCALE, index=None, horizontal=True,
                             format_func=scale_fmt("strength_1", "strength_5"), key=f"b_str_{trial_index()}")
            reason = st.text_area(tr("blind_reason"), placeholder=tr("blind_reason_ph"), max_chars=600,
                                  key=f"b_why_{trial_index()}")
            ok = st.form_submit_button(tr("submit"), type="primary")
            skip_art = skip_button("blind")
        if skip_art:
            skip_trial(trial, "blind")
            st.rerun()
        if ok:
            if e is None or s is None:
                st.error(tr("need_emotion") if e is None else tr("need_strength"))
                return
            ss.cur = {"blind_emotion": e, "blind_strength": int(s), "blind_reason": (reason or "").strip() or None,
                      "rt_blind_ms": elapsed_ms()}
            if trial["kind"] == "check":
                ss.cur |= {"check_expect_emotion": trial["expect_emotion"], "check_expect_strength": trial["expect_strength"],
                           "check_pass": bool(e == trial["expect_emotion"] and int(s) == trial["expect_strength"])}
                finish_trial(trial)
            else:
                if not save_progress(trial):
                    return
                ss.step = "reveal"
            st.rerun()


def step_reveal(trial):
    """(b) the R1 estimate: 10 viewers + all nine bars; plausibility 1-5."""
    timer(f"{trial_index()}:reveal")
    out = model_outputs(trial["stimulus_id"])
    probs, counts = np.array(out["r1"]), np.array(out["r1_counts"])
    labels = [emo(e) for e in EMOTIONS]
    left, right = st.columns([5, 6], gap="large")
    with left:
        artwork(trial)
    with right:
        st.markdown(f'<div class="af-h2" style="margin-top:0">{ui.esc(tr("reveal_h"))}</div>'
                    f'<p class="af-lead">{ui.esc(tr("reveal_intro"))}</p>', unsafe_allow_html=True)
        order = sorted([i for i in range(9) if counts[i]], key=lambda i: -counts[i])
        legend = [(EMOTIONS[i], f"<b>{counts[i]}</b>&nbsp;{ui.esc(labels[i])}" if L == "en" else
                   f"{ui.esc(labels[i])}&nbsp;<b>{counts[i]}</b>&nbsp;人") for i in order]
        st.markdown(ui.card(ui.people_block(M, EMOTIONS, labels, counts, legend), tr("ten_people")), unsafe_allow_html=True)
        st.markdown(f'<div class="af-section">{ui.esc(tr("distribution"))}</div>', unsafe_allow_html=True)
        st.plotly_chart(ui.distribution_chart(M, EMOTIONS, labels, probs, [app_i18n.sentiment(L, SENTIMENT[e]) for e in EMOTIONS],
                                              "", height=320), theme=None, config={"displayModeBar": False, "staticPlot": True},
                        key=f"rev_{trial_index()}")
        with st.form(f"reveal_{trial_index()}", border=False):
            with st.container(key=f"scale_plaus_{trial_index()}"):
                r = st.radio(tr("plaus_q"), SCALE, index=None, horizontal=True, format_func=scale_fmt("plaus_1", "plaus_5"),
                             key=f"r_pl_{trial_index()}")
            ok = st.form_submit_button(tr("next"), type="primary")
            skip_art = skip_button("reveal")
        if skip_art:
            skip_trial(trial, "reveal")
            st.rerun()
        if ok:
            if r is None:
                st.error(tr("need_rating"))
                return
            ss.cur |= {"plausibility": int(r), "rt_reveal_ms": elapsed_ms()}
            if not save_progress(trial):
                return
            ss.step = "ab"
            st.rerun()


def step_ab(trial):
    """(c) R1 vs R3 side by side, unlabelled; which side is which was randomised at assignment and is recorded."""
    timer(f"{trial_index()}:ab")
    out = model_outputs(trial["stimulus_id"])
    left_model = trial["ab_left"]
    right_model = "r3" if left_model == "r1" else "r1"
    xmax = 1.22 * max(max(out["r1"]), max(out["r3"]))
    st.markdown(f'<div class="af-h2" style="margin-top:0">{ui.esc(tr("ab_h"))}</div>'
                f'<p class="af-lead">{ui.esc(tr("ab_intro"))}</p>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    for col, name, model in ((a, "ab_a", left_model), (b, "ab_b", right_model)):
        with col:
            st.markdown(f'<div class="af-ab-label">{ui.esc(tr(name))}</div>', unsafe_allow_html=True)
            ab_chart(out[model], xmax, key=f"ab_{name}_{trial_index()}")
    with st.form(f"ab_{trial_index()}", border=False):
        with st.container(key=f"scale_ab_{trial_index()}"):
            c = st.radio(tr("ab_q"), ["A", "B", "same"], index=None, horizontal=True,
                         format_func={"A": tr("ab_opt_a"), "B": tr("ab_opt_b"), "same": tr("ab_opt_same")}.get,
                         key=f"ab_c_{trial_index()}")
        ok = st.form_submit_button(tr("next"), type="primary")
        skip_art = skip_button("ab")
    if skip_art:
        skip_trial(trial, "ab")
        st.rerun()
    if ok:
        if c is None:
            st.error(tr("need_choice"))
            return
        ss.cur |= {"ab_left_model": left_model, "ab_choice": c,
                   "ab_choice_model": {"A": left_model, "B": right_model, "same": "same"}[c], "rt_ab_ms": elapsed_ms()}
        if not save_progress(trial):
            return
        ss.step = "attn"
        st.rerun()


def step_attention(trial):
    """(d) optional: the R1 attention overlay for R1's top emotion; rating 1-5 or skip."""
    timer(f"{trial_index()}:attn")
    out = model_outputs(trial["stimulus_id"])
    left, right = st.columns([6, 5], gap="large")
    with left:
        st.markdown(ui.framed_artwork(image_uri(out["attention_image"]), tr("alt_attention", emotion=emo(out["attention_emotion"]))),
                    unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="af-h2" style="margin-top:0">{ui.esc(tr("att_h"))}</div>'
                    f'<p class="af-lead">{ui.esc(tr("att_intro", emotion=emo(out["attention_emotion"])))}</p>',
                    unsafe_allow_html=True)
        with st.form(f"att_{trial_index()}", border=False):
            with st.container(key=f"scale_att_{trial_index()}"):
                r = st.radio(tr("att_q"), SCALE, index=None, horizontal=True, format_func=scale_fmt("att_1", "att_5"),
                             key=f"a_r_{trial_index()}")
            c1, c2, c3 = st.columns(3)
            ok = c1.form_submit_button(tr("next"), type="primary")
            skip = c2.form_submit_button(tr("att_skip"))
            with c3:
                skip_art = skip_button("attn")
        if skip_art:
            skip_trial(trial, "attn")
            st.rerun()
        if ok or skip:
            if ok and r is None:
                st.error(tr("need_rating"))
                return
            ss.cur |= {"attention_rating": None if skip else int(r), "attention_skipped": bool(skip),
                       "rt_attention_ms": elapsed_ms()}
            finish_trial(trial)
            st.rerun()


def trial_row(trial, status):
    return {"participant_id": ss.pid, "trial_index": trial_index(), "kind": trial["kind"],
            "stimulus_id": trial["stimulus_id"], "lang": L, "created_at": store.now(), "study_mode": MODE,
            "phase": MODE, "skipped": False, "skipped_step": None, **ss.cur, "status": status}


def save_progress(trial) -> bool:
    """Store the steps submitted so far (status incomplete), so they are kept if the page is closed mid-trial."""
    return save(backend().add_trial, trial_row(trial, "incomplete"))


RT_KEY = {"blind": "rt_blind_ms", "reveal": "rt_reveal_ms", "ab": "rt_ab_ms", "attn": "rt_attention_ms"}


def skip_button(step):
    return st.form_submit_button(tr("skip_artwork"), key=f"skip_{step}_{trial_index()}")


def skip_trial(trial, step):
    """'Skip this artwork': record where, keep earlier submitted steps, move on. Analysis treats it as missing."""
    ss.cur |= {"skipped": True, "skipped_step": step, RT_KEY[step]: elapsed_ms()}
    if trial["kind"] == "check":
        ss.cur |= {"check_expect_emotion": trial["expect_emotion"], "check_expect_strength": trial["expect_strength"],
                   "check_pass": None}
    finish_trial(trial, status="skipped")


def finish_trial(trial, status="complete"):
    if not save(backend().add_trial, trial_row(trial, status)):
        return
    ss.cur, ss.step = {}, "blind"
    if ss.phase == "practice":
        ss.phase = "practice_done"
    elif ss.ti + 1 < len(ss.plan):
        ss.ti += 1
    else:
        ss.phase = "questionnaire"


STEPS = {"blind": step_blind, "reveal": step_reveal, "ab": step_ab, "attn": step_attention}


# ---------------------------------------------------------------- pages
def page_consent():
    st.markdown(f'<div class="af-h2" style="margin-top:0">{ui.esc(tr("consent_h"))}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(consent_text())
    pilot = MODE == "pilot"
    agree = st.checkbox(tr("consent_check_pilot" if pilot else "consent_check"), key="consent_ok")  # required
    optin = st.checkbox(tr("consent_optin"), value=False, key="consent_optin") if pilot else False  # optional
    if st.button(tr("consent_continue"), type="primary", key="consent_go"):
        if not agree:
            st.error(tr("consent_required"))
            return
        b = backend()
        ss.pid = str(uuid.uuid4())
        ss.pcode = store.new_participant_code(b.participant_codes())
        idx = b.next_arrival_index()
        k, plan = assignment.assign(idx, ss.pid, sets())
        ss.plan, ss.ti, ss.step, ss.cur = plan, 0, "blind", {}
        if save(b.upsert_participant, {"participant_id": ss.pid, "participant_code": ss.pcode, "arrival_index": idx,
                                        "set_index": k, "plan": json.dumps(plan), "lang": L,
                                        "consented_at": store.now(), "app_version": APP_VERSION, "study_mode": MODE,
                                        "consent_pilot": True if pilot else None, "future_use_optin": bool(optin),
                                        "consent_version": CFG["consent_version"], "phase": MODE,
                                        "status": "incomplete"}):
            ss.phase = "pcode"
            st.rerun()


def page_pcode():
    """Shown once, right after consent: the code the participant needs to withdraw their answers."""
    st.markdown(f'<div class="af-h2" style="margin-top:0">{ui.esc(tr("pcode_h"))}</div>'
                f'<p class="af-lead">{ui.esc(tr("pcode_label"))}</p>'
                f'<div class="af-card af-pcode" role="status">{ui.esc(ss.pcode)}</div>'
                f'<p class="af-lead">{ui.esc(tr("pcode_body"))}</p>', unsafe_allow_html=True)
    if st.button(tr("pcode_continue"), type="primary", key="pcode_go"):
        ss.phase = "instructions"
        st.rerun()


def page_instructions():
    st.markdown(f'<div class="af-h2" style="margin-top:0">{ui.esc(tr("instr_h"))}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(tr("instr_body"))
    if st.button(tr("instr_start_practice"), type="primary", key="go_practice"):
        ss.phase, ss.step, ss.cur = "practice", "blind", {}
        st.rerun()


def page_practice_done():
    st.markdown(f'<div class="af-h2" style="margin-top:0">{ui.esc(tr("practice_done_h"))}</div>'
                f'<p class="af-lead">{ui.esc(tr("practice_done_body"))}</p>', unsafe_allow_html=True)
    if st.button(tr("start_study"), type="primary", key="go_study"):
        ss.phase, ss.ti, ss.step = "trial", 0, "blind"
        save(backend().upsert_participant, {"participant_id": ss.pid, "started_at": store.now()})
        st.rerun()


def page_trial():
    trial = current_trial()
    progress(trial)
    STEPS[ss.step](trial)


def page_questionnaire():
    st.markdown(f'<div class="af-h2" style="margin-top:0">{ui.esc(tr("q_h"))}</div>'
                f'<p class="af-lead">{ui.esc(tr("q_intro"))}</p>', unsafe_allow_html=True)
    with st.form("questionnaire", border=False):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            age = st.selectbox(tr("q_age"), si.AGE, index=None, format_func=tr, key="q_age", placeholder="—")
            gender = st.selectbox(tr("q_gender"), si.GENDER, index=None, format_func=tr, key="q_gender", placeholder="—")
            with st.container(key="scale_art"):
                art = st.radio(tr("q_art"), si.ART, index=None, format_func=tr, key="q_art")
        with c2:
            country = st.text_input(tr("q_country"), max_chars=80, key="q_country")
            native = st.selectbox(tr("q_native"), si.NATIVE, index=None, format_func=tr, key="q_native", placeholder="—")
            native_other = st.text_input(tr("q_native_other"), max_chars=60, key="q_native_other")
        st.markdown(f'<div class="af-h2">{ui.esc(tr("q_sus_h"))}</div><p class="af-lead">{ui.esc(tr("q_sus_intro"))}</p>',
                    unsafe_allow_html=True)
        sus = []
        for i, item in enumerate(si.SUS[L], 1):
            with st.container(key=f"scale_sus_{i}"):
                sus.append(st.radio(f"{i}. {item}", SCALE, index=None, horizontal=True,
                                    format_func=scale_fmt("sus_scale_1", "sus_scale_5"), key=f"sus_{i}"))
        comment = st.text_area(tr("q_comment"), max_chars=2000, key="q_comment")
        ok = st.form_submit_button(tr("q_submit"), type="primary")
    if ok:
        missing = [tr(k) for k, v in (("q_age", age), ("q_art", art), ("q_country", (country or "").strip()),
                                      ("q_native", native)) if not v]
        missing += [f"SUS {i}" for i, v in enumerate(sus, 1) if v is None]
        if missing:
            st.error(tr("q_missing", items=", ".join(missing)))
            return
        trials = [t for t in backend().trials() if t["participant_id"] == ss.pid and t["kind"] == "check"]
        code = "AF-" + secrets.token_hex(4).upper()
        row = {"participant_id": ss.pid, "completed_at": store.now(), "completion_code": code, "lang": L, "status": "complete",
               "age_band": age, "gender": gender, "art_experience": art, "country_culture": country.strip(),
               "native_language": native if native != "lang_other" else f"lang_other:{(native_other or '').strip()}",
               **{f"sus_{i}": int(v) for i, v in enumerate(sus, 1)}, "comment": (comment or "").strip() or None,
               "checks_passed": sum(bool(t["check_pass"]) for t in trials)}
        if save(backend().upsert_participant, row):
            ss.code, ss.phase = code, "thanks"
            st.rerun()


def page_thanks():
    st.markdown(f'<div class="af-h2" style="margin-top:0">{ui.esc(tr("thanks_h"))}</div>'
                f'<p class="af-lead">{ui.esc(tr("thanks_body"))}</p>'
                f'<div class="af-card" style="font-size:2rem;font-weight:700;letter-spacing:.08em;text-align:center">'
                f'{ui.esc(ss.code)}</div><p class="af-lead">{ui.esc(tr("pcode_reminder", code=ss.pcode))}</p>'
                f'<p class="af-lead">{ui.esc(tr("thanks_close"))}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------- dev-only screenshot jumps (STUDY_DEV=1)
if DEV and "shot" in st.query_params and not ss.get("_shot"):
    ss._shot = True
    shot = st.query_params["shot"]
    ss.pid, ss.cur, ss.code, ss.pcode = "dev-" + uuid.uuid4().hex[:8], {}, "AF-DEV00000", store.new_participant_code()
    _, ss.plan = assignment.assign(0, ss.pid, sets())
    ss.ti = next(i for i, t in enumerate(ss.plan) if t["kind"] == ("check" if shot == "check" else "stimulus"))
    ss.phase = {"consent": "consent", "pcode": "pcode", "instructions": "instructions", "practice": "practice", "questionnaire": "questionnaire",
                "thanks": "thanks"}.get(shot, "trial")
    ss.step = {"reveal": "reveal", "ab": "ab", "attn": "attn"}.get(shot, "blind")
    if ss.step != "blind":
        ss.cur = {"blind_emotion": "awe", "blind_strength": 3, "plausibility": 3}

{"consent": page_consent, "pcode": page_pcode, "instructions": page_instructions, "practice": page_trial,
 "practice_done": page_practice_done, "trial": page_trial, "questionnaire": page_questionnaire,
 "thanks": page_thanks}[ss.phase]()
