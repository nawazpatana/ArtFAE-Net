"""Drive the real study app headlessly (streamlit.testing AppTest) as a synthetic participant.

Used by tests/test_study.py and pilot_sim.py. Answers are random but plausible; `fail_checks` makes the
participant fail the attention checks (to exercise the exclusion rule).
"""
import random
from pathlib import Path

import study_i18n as si
from common import EMOTIONS

APP = Path(__file__).resolve().parent / "study_app.py"


def _click(at, label):
    btn = [b for b in at.button if b.label == label]
    assert btn, f"no button {label!r}; have {[b.label for b in at.button]}"
    btn[0].click().run()
    assert not at.exception, at.exception


def new_app(timeout=120):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, at.exception
    return at


def run_participant(at, lang="en", seed=0, fail_checks=False, on_step=None, max_steps=400, optin=False, skip_rate=0.0):
    """Complete the whole study. on_step(at, phase, step, trial) is called on every screen before answering.
    optin ticks the optional future-use box; skip_rate is the chance of pressing "Skip this artwork" on a step."""
    rng = random.Random(seed)
    t = lambda k: si.t(lang, k)  # noqa: E731
    if lang != at.session_state["lang"]:
        at.radio(key="lang").set_value(lang).run()
    for _ in range(max_steps):
        ss = at.session_state
        phase = ss["phase"]
        step = ss["step"] if "step" in ss else None
        trial = None
        if phase in ("practice", "trial"):
            trial = {"kind": "practice", "stimulus_id": "P01"} if phase == "practice" else ss["plan"][ss["ti"]]
        if on_step:
            on_step(at, phase, step, trial)
        if phase == "consent":
            at.checkbox(key="consent_ok").check()
            if optin:
                at.checkbox(key="consent_optin").check()
            at.button(key="consent_go").click().run()
        elif phase == "pcode":
            at.button(key="pcode_go").click().run()
        elif phase == "instructions":
            at.button(key="go_practice").click().run()
        elif phase == "practice_done":
            at.button(key="go_study").click().run()
        elif phase in ("practice", "trial"):
            idx = 0 if phase == "practice" else ss["ti"] + 1
            if skip_rate and rng.random() < skip_rate:
                at.button(key=f"skip_{step}_{idx}").click().run()
                assert not at.exception, at.exception
                continue
            if step == "blind":
                if trial["kind"] == "check" and not fail_checks:
                    e, s = trial["expect_emotion"], trial["expect_strength"]
                else:
                    e, s = rng.choice(EMOTIONS), rng.randint(1, 5)
                at.radio(key=f"b_emo_{idx}").set_value(e)
                at.radio(key=f"b_str_{idx}").set_value(s)
                if rng.random() < .3:
                    at.text_area(key=f"b_why_{idx}").input("synthetic reason")
                _click(at, t("submit"))
            elif step == "reveal":
                at.radio(key=f"r_pl_{idx}").set_value(rng.randint(1, 5))
                _click(at, t("next"))
            elif step == "ab":
                at.radio(key=f"ab_c_{idx}").set_value(rng.choice(["A", "B", "B", "same"]))
                _click(at, t("next"))
            elif step == "attn":
                if rng.random() < .2:
                    _click(at, t("att_skip"))
                else:
                    at.radio(key=f"a_r_{idx}").set_value(rng.randint(1, 5))
                    _click(at, t("next"))
        elif phase == "questionnaire":
            at.selectbox(key="q_age").set_value(rng.choice(si.AGE))
            if rng.random() < .7:
                at.selectbox(key="q_gender").set_value(rng.choice(si.GENDER))
            at.radio(key="q_art").set_value(rng.choice(si.ART))
            at.text_input(key="q_country").input(rng.choice(["Taiwan", "Pakistan", "Japan", "Germany"]))
            at.selectbox(key="q_native").set_value(rng.choice(si.NATIVE[:4]))
            for i in range(1, 11):
                at.radio(key=f"sus_{i}").set_value(rng.randint(1, 5))
            _click(at, t("q_submit"))
        elif phase == "thanks":
            if on_step:
                on_step(at, phase, None, None)
            return at.session_state["pid"], at.session_state["code"]
        assert not at.exception, at.exception
    raise RuntimeError("participant did not finish")
