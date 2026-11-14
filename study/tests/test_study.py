"""Study-mode tests: the blind step never exposes model outputs; assignment is balanced; storage round-trips;
both languages render the whole flow. Run with the demo's venv:  artfae_demo/.venv/bin/python -m pytest study/tests
"""
import collections
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(ROOT / "app"))
import assignment  # noqa: E402
import storage as store  # noqa: E402
import study_i18n as si  # noqa: E402
from common import EMOTIONS, N_TRIALS, people_counts  # noqa: E402

STIM = pd.read_csv(STUDY / "stimuli.csv", keep_default_na=False)
OUT = json.loads((STUDY / "precomputed/outputs.json").read_text())


@pytest.fixture
def db(tmp_path, monkeypatch):
    import streamlit as st
    monkeypatch.setenv("STUDY_STORAGE", "sqlite")
    monkeypatch.setenv("STUDY_DB", str(tmp_path / "t.sqlite"))
    st.cache_resource.clear()
    yield tmp_path / "t.sqlite"
    st.cache_resource.clear()


# ---------------------------------------------------------------- data integrity
def test_label_order_and_names():
    sys.path.insert(0, str(ROOT))
    from artfae_balance import EMOTIONS as E
    import i18n
    assert EMOTIONS == list(E) == OUT["meta"]["emotions"]
    for lang in i18n.LANGS:
        assert set(i18n.EMOTION_NAMES[lang]) == set(EMOTIONS)


def test_i18n_parity():
    assert set(si.STRINGS["en"]) == set(si.STRINGS["zh"])
    assert len(si.SUS["en"]) == len(si.SUS["zh"]) == 10


def test_stimuli_and_outputs():
    st_ = STIM[STIM.role == "stimulus"]
    assert len(st_) == 60 and st_.stimulus_id.is_unique and st_.artwork_id.is_unique
    assert {"anger/high", "fear/high", "disgust/high", "tied", "dark", "minimalist"} <= set(st_.stratum)
    assert not {c for c in STIM.columns if c.startswith(("p_", "n_"))}  # vote shares only in private/
    for sid in [*st_.stimulus_id, "P01"]:
        o = OUT["stimuli"][sid]
        assert abs(sum(o["r1"]) - 1) < 1e-4 and abs(sum(o["r3"]) - 1) < 1e-4
        assert o["r1_counts"] == people_counts(o["r1"]).tolist() and sum(o["r1_counts"]) == 10
        assert (STUDY / o["attention_image"]).exists()
    assert OUT["meta"]["max_abs_diff_vs_saved_holdout_logits"] < 1e-3


def test_people_counts_largest_remainder():
    assert people_counts([.26, .24, .17, .09, .09, .05, .05, .03, .02]).tolist() == [3, 2, 2, 1, 1, 1, 0, 0, 0]
    assert people_counts([1, 0, 0, 0, 0, 0, 0, 0, 0]).sum() == 10


# ---------------------------------------------------------------- assignment
def test_assignment_balanced():
    sets = assignment.make_sets(STIM[STIM.role == "stimulus"].to_dict("records"))
    assert sorted(sum(sets, [])) == sorted(STIM[STIM.role == "stimulus"].stimulus_id)
    seen, left = collections.Counter(), collections.Counter()
    first_positions = collections.Counter()
    for k in range(300):
        _, plan = assignment.assign(k, f"p{k}", sets)
        stim = [t for t in plan if t["kind"] == "stimulus"]
        checks = [i for i, t in enumerate(plan) if t["kind"] == "check"]
        assert len(stim) == N_TRIALS and len({t["stimulus_id"] for t in stim}) == N_TRIALS
        assert len(checks) == 2 and 4 <= checks[0] <= 8 and checks[1] >= 13
        seen.update(t["stimulus_id"] for t in stim)
        left.update(t["ab_left"] for t in stim)
        first_positions[stim[0]["stimulus_id"]] += 1
    assert set(seen.values()) == {100}  # every stimulus seen exactly 300 * 20 / 60 times
    assert abs(left["r1"] / sum(left.values()) - .5) < .03
    assert len(first_positions) > 40  # orders are shuffled per participant
    # each set covers the strata about equally
    strata = dict(zip(STIM.stimulus_id, STIM.stratum.str.split("/").str[0]))
    for s in sets:
        c = collections.Counter(strata[i] for i in s)
        assert max(c.values()) <= 4


def test_assignment_deterministic():
    sets = assignment.make_sets(STIM[STIM.role == "stimulus"].to_dict("records"))
    assert assignment.assign(7, "abc", sets) == assignment.assign(7, "abc", sets)


# ---------------------------------------------------------------- storage
def _rows():
    p = {"participant_id": "u1", "arrival_index": 0, "set_index": 0, "plan": "[]", "lang": "zh", "consented_at": "t0"}
    tr = {f: None for f in store.TRIAL_FIELDS} | {"participant_id": "u1", "trial_index": 1, "kind": "stimulus",
                                                   "stimulus_id": "S01", "blind_emotion": "awe", "blind_strength": 4,
                                                   "blind_reason": "光線 “quoted”", "plausibility": 3, "ab_choice": "A"}
    return p, tr


@pytest.mark.parametrize("kind", ["sqlite", "hf_local"])
def test_storage_round_trip(tmp_path, kind):
    b = store.SQLiteStorage(tmp_path / "s.sqlite") if kind == "sqlite" else \
        store.HFDatasetStorage(tmp_path / "hf", repo_id="x/y", scheduler=None)
    p, tr = _rows()
    assert b.next_arrival_index() == 0
    b.upsert_participant(p)
    b.upsert_participant({"participant_id": "u1", "completed_at": "t9", "sus_1": 5})
    b.add_trial(tr)
    b.add_trial(tr | {"plausibility": 4})  # a re-save of the same trial replaces it
    assert b.next_arrival_index() == 1
    (pp,), (tt,) = b.participants(), b.trials()
    assert pp["lang"] == "zh" and pp["completed_at"] == "t9" and pp["sus_1"] == 5 and pp["consented_at"] == "t0"
    assert tt["blind_reason"] == "光線 “quoted”" and tt["plausibility"] == 4 and tt["blind_strength"] == 4
    assert set(tt) == set(store.TRIAL_FIELDS)
    if kind == "hf_local":
        assert all(f.suffix == ".jsonl" for f in (tmp_path / "hf" / "data").iterdir())


def test_no_identifying_fields():
    bad = {"name", "email", "ip", "ip_address", "user_agent", "address", "phone"}
    assert not bad & set(store.TRIAL_FIELDS) and not bad & set(store.PARTICIPANT_FIELDS)


# ---------------------------------------------------------------- app: blind step and both languages
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_full_flow_and_blind_step(db, lang):
    import sim
    blind_screens = []

    def check(at, phase, step, trial):
        if phase in ("practice", "trial") and step == "blind":
            reads = list(at.session_state["_outputs_read"]) if "_outputs_read" in at.session_state else []
            assert not [r for r in reads if r[1] == "blind"], reads
            # nothing model-derived on the page: no charts, no people icons, no overlay
            assert not at.get("plotly_chart")
            html = " ".join(m.value for m in at.markdown)
            assert 'class="af-people' not in html and 'class="af-card af-result' not in html and "<svg" not in html
            if trial["kind"] != "check":
                o = OUT["stimuli"][trial["stimulus_id"]]
                for p in o["r1"] + o["r3"]:
                    assert f"{p * 100:.1f}%" not in html
            blind_screens.append(trial["stimulus_id"])
        assert any(si.t(at.session_state["lang"], "pilot_banner") in m.value for m in at.markdown), (phase, step)
        if phase == "trial" and step in ("reveal", "ab", "attn"):
            assert at.session_state["_outputs_read"][-1][2] == trial["stimulus_id"]

    at = sim.new_app()
    assert "_outputs_read" not in at.session_state  # nothing read before consent
    pid, code = sim.run_participant(at, lang=lang, seed=1, on_step=check)
    assert len(blind_screens) == 1 + N_TRIALS + 2
    assert si.t(lang, "thanks_h") in " ".join(m.value for m in at.markdown)
    b = store.SQLiteStorage(db)
    (p,) = b.participants()
    assert p["participant_id"] == pid and p["completion_code"] == code and p["checks_passed"] == 2
    assert p["participant_code"] == at.session_state["pcode"] and p["participant_code"] != code
    thanks = " ".join(m.value for m in at.markdown)
    assert code in thanks and p["participant_code"] in thanks  # completion code, plus the participant-code reminder
    trials = b.trials()
    assert len(trials) == 1 + N_TRIALS + 2 and {t["lang"] for t in trials} == {lang}
    main = [t for t in trials if t["kind"] == "stimulus"]
    assert all(t["rt_blind_ms"] is not None and t["rt_reveal_ms"] is not None and t["rt_ab_ms"] is not None for t in main)
    assert all(t["ab_choice_model"] in ("r1", "r3", "same") and t["ab_left_model"] in ("r1", "r3") for t in main)
    plan = json.loads(p["plan"])
    assert [t["stimulus_id"] for t in sorted(trials, key=lambda t: t["trial_index"])][1:] == [x["stimulus_id"] for x in plan]


def test_consent_required(db):
    import sim
    at = sim.new_app()
    at.button(key="consent_go").click().run()
    assert at.session_state["phase"] == "consent" and at.error
    assert not store.SQLiteStorage(db).participants()


def test_blind_requires_answer(db):
    import sim
    at = sim.new_app()
    at.checkbox(key="consent_ok").check(); at.button(key="consent_go").click().run()
    at.button(key="pcode_go").click().run()
    at.button(key="go_practice").click().run()
    sim._click(at, si.t("en", "submit"))
    assert at.session_state["step"] == "blind" and at.error
    assert "_outputs_read" not in at.session_state


def test_image_rights():
    """Every shown image (60 stimuli, practice, 2 checks) passes all three rules, with recorded sources:
    artist died before 1956, work created before 1931, a 2D work (not sculpture, relief, architecture, ceramic...)."""
    sys.path.insert(0, str(STUDY))
    import artist_rights as ar
    s = STIM
    assert len(s) == 63 and s.artist_slug.is_unique and s.artwork_id.is_unique
    assert (s.rights_ok.astype(str) == "True").all() and (s.is_2d.astype(str) == "True").all()
    assert (s.artist_death_year.astype(int) < ar.DEATH_BEFORE).all()
    assert (s.created_year.astype(int) < ar.CREATED_BEFORE).all()
    assert s.death_source.str.startswith(("WikiArt https://www.wikiart.org/en/", "Wikidata https://www.wikidata.org/")).all()
    assert s.form_source.str.startswith("2D").all() and (s.created_source != "").all()
    assert not set(s.artwork_id) & (set(ar.EXCLUDE) | set(ar.MANUAL_3D))
    # re-derive every rule from the cached sources (no network)
    ac, wc = ar.Cache(ar.ARTIST_CACHE), ar.Cache(ar.WORK_CACHE)
    for _, r in s.iterrows():
        w = wc.data[r.artwork_id]
        assert not {g.strip() for g in (w.get("genre") or "").lower().split(",")} & ar.GENRES_3D, r.stimulus_id
        assert not ar.MEDIA_3D.search(w.get("media") or ""), r.stimulus_id
        _, death, _ = ar.death_info(ac.data[r.artist_slug])
        created, _ = ar.created_info(w, int(r.year) if r.year else None)
        assert death < ar.DEATH_BEFORE and created < ar.CREATED_BEFORE, r.stimulus_id
    rep = pd.read_csv(STUDY / "rights_replacements.csv", keep_default_na=False)
    assert (rep.reason != "").all()
    replaced = set(s.loc[s.replaced_from != "", "stimulus_id"])
    assert replaced == set(rep.stimulus_id)
    assert not set(rep.old_artwork_id) & set(s.artwork_id)  # nothing that failed is shown


# ---------------------------------------------------------------- frozen stimulus set
import shutil  # noqa: E402

import frozen  # noqa: E402


def test_frozen_manifest_matches():
    assert frozen.verify(STUDY) == []
    m = json.loads((STUDY / frozen.MANIFEST).read_text())
    files = m["files"]
    assert sum(k.startswith("stimuli/images/") for k in files) == 63 == len(STIM)
    assert {"stimuli.csv", "precomputed/outputs.json", "precomputed/maps.npz"} <= set(files)
    assert sum(k.startswith("precomputed/attention/") for k in files) == len(OUT["stimuli"])
    assert m["assignment_sets"] == assignment.make_sets(STIM[STIM.role == "stimulus"].to_dict("records"))
    assert len(m["frozen_on"]) == 10 and m["frozen_on"].startswith("20")


def _study_copy(tmp_path):
    """A self-contained copy of the study (as in the Space bundle), so it can be tampered with."""
    dst = tmp_path / "study"
    shutil.copytree(STUDY, dst, ignore=shutil.ignore_patterns("private", "space", "tests", "__pycache__"))
    (dst / "vendor").mkdir()
    for f in ("ui.py", "i18n.py"):
        shutil.copy2(ROOT / "app" / f, dst / "vendor" / f)
    return dst


def test_frozen_detects_every_kind_of_change(tmp_path):
    d = _study_copy(tmp_path)
    assert frozen.verify(d) == []
    img = d / "stimuli/images/S01.jpg"
    img.write_bytes(img.read_bytes() + b"\0")
    (d / "precomputed/outputs.json").write_text((d / "precomputed/outputs.json").read_text().replace("0.", "0.0", 1))
    (d / "precomputed/attention/S02.jpg").unlink()
    shutil.copy2(d / "stimuli/images/S03.jpg", d / "stimuli/images/S99.jpg")
    p = frozen.verify(d)
    assert "changed: stimuli/images/S01.jpg" in p and "changed: precomputed/outputs.json" in p
    assert "missing: precomputed/attention/S02.jpg" in p and "not in the frozen set: stimuli/images/S99.jpg" in p
    d2 = _study_copy(tmp_path / "b")
    csv_ = d2 / "stimuli.csv"
    csv_.write_text(csv_.read_text().replace("anger/high", "anger/low", 1))
    assert frozen.verify(d2) == ["changed: stimuli.csv"]


@pytest.mark.parametrize("tamper", [False, True])
def test_app_refuses_to_start_when_tampered(tmp_path, monkeypatch, tamper):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    d = _study_copy(tmp_path)
    if tamper:
        img = d / "stimuli/images/S10.jpg"
        b = bytearray(img.read_bytes()); b[len(b) // 2] ^= 1; img.write_bytes(bytes(b))
    db = tmp_path / "t.sqlite"
    monkeypatch.setenv("STUDY_STORAGE", "sqlite")
    monkeypatch.setenv("STUDY_DB", str(db))
    st.cache_resource.clear()
    at = AppTest.from_file(str(d / "study_app.py"), default_timeout=120).run()
    assert not at.exception
    if tamper:
        assert at.error and si.t("en", "frozen_error") in at.error[0].value and si.t("zh", "frozen_error") in at.error[0].value
        assert "changed: stimuli/images/S10.jpg" in at.code[0].value
        assert not at.checkbox and not at.button  # no consent, nothing can be started
        assert not db.exists()  # nothing stored
    else:
        assert not at.error and at.checkbox(key="consent_ok") is not None
    st.cache_resource.clear()


# ---------------------------------------------------------------- STUDY_MODE (pilot / production)
def _app(monkeypatch, tmp_path, **env):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    for k in ("STUDY_MODE", "CONSENT_APPROVED", "CONSENT_VERSION", "STUDY_STORAGE", "DATASET_REPO", "HF_TOKEN", "SPACE_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("STUDY_DB", str(tmp_path / "m.sqlite"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    st.cache_resource.clear()
    at = AppTest.from_file(str(STUDY / "study_app.py"), default_timeout=120).run()
    assert not at.exception
    return at


def _html(at):
    return " ".join(m.value for m in at.markdown)


def test_default_is_pilot_with_banner(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path)
    assert si.t("en", "pilot_banner") in _html(at) and at.checkbox(key="consent_ok") is not None
    at.radio(key="lang").set_value("zh").run()
    assert si.t("zh", "pilot_banner") in _html(at)
    assert store.deployment()["mode"] == "pilot"


@pytest.mark.parametrize("env", [{"STUDY_MODE": "production"}, {"STUDY_MODE": "production", "CONSENT_APPROVED": "yes"},
                                 {"STUDY_MODE": "live"}])
def test_production_needs_consent_approval(monkeypatch, tmp_path, env):
    at = _app(monkeypatch, tmp_path, **env)
    assert at.error and si.t("en", "config_error") in at.error[0].value
    assert not at.checkbox and not at.button and not (tmp_path / "m.sqlite").exists()


def test_production_when_approved(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, STUDY_MODE="production", CONSENT_APPROVED="1", CONSENT_VERSION="final-1.0")
    assert "consent_optin" not in [c.key for c in at.checkbox]  # the pilot opt-in box is pilot-only
    assert not at.error and si.t("en", "pilot_banner") not in _html(at)
    at.checkbox(key="consent_ok").check()
    at.button(key="consent_go").click().run()
    (p,) = store.SQLiteStorage(tmp_path / "m.sqlite").participants()
    assert p["study_mode"] == "production" and p["participant_code"] == store.normalize_code(p["participant_code"])


@pytest.mark.parametrize("mode,repo,token,ok", [
    ("pilot", "nawazishpatana/artfae-study-pilot", "x", True),
    ("pilot", "nawazishpatana/artfae-study-responses", "x", False),  # pilot must never write to the responses repo
    ("production", "nawazishpatana/artfae-study-pilot", "x", False),
    ("production", "nawazishpatana/artfae-study-responses", "x", True),
    ("pilot", "", "x", False),
    ("pilot", "nawazishpatana/artfae-study-pilot", "", False),  # no HF_TOKEN secret
])
def test_dataset_repo_must_match_mode(monkeypatch, mode, repo, token, ok):
    for k in ("STUDY_MODE", "CONSENT_APPROVED", "DATASET_REPO", "HF_TOKEN", "STUDY_STORAGE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SPACE_ID", "nawazishpatana/artfae-study")  # as on the Space: hf storage by default
    monkeypatch.setenv("STUDY_MODE", mode)
    monkeypatch.setenv("CONSENT_APPROVED", "1")
    monkeypatch.setenv("CONSENT_VERSION", "final-1.0")
    if repo:
        monkeypatch.setenv("DATASET_REPO", repo)
    if token:
        monkeypatch.setenv("HF_TOKEN", token)
    cfg = store.deployment()
    assert cfg["storage"] == "hf" and (not cfg["problems"]) == ok, cfg


def test_app_refuses_wrong_repo_before_any_network(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, SPACE_ID="nawazishpatana/artfae-study", HF_TOKEN="dummy",
              DATASET_REPO="nawazishpatana/artfae-study-responses")
    assert at.error and "DATASET_REPO must be nawazishpatana/artfae-study-pilot" in at.code[0].value


# ---------------------------------------------------------------- consent wording, participant code, withdrawal
import re  # noqa: E402

import withdraw  # noqa: E402

CODE_RE = re.compile(r"^[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{4}-[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{4}$")


def test_consent_checkbox_confirms_18_plus():
    assert si.t("en", "consent_check") == "I am 18 or older, I have read this information, and I agree to take part."
    assert "18" in si.t("zh", "consent_check")
    for lang in ("en", "zh"):  # the final templates' checkbox line is the app's checkbox wording
        tpl = (STUDY / f"consent_{lang}_FINAL_TEMPLATE.md").read_text(encoding="utf-8")
        assert f"☐ {si.t(lang, 'consent_check')}" in tpl


def bad_bold(text):
    """**spans** that CommonMark will NOT render as bold: an opening ** between a letter and punctuation, or a
    closing ** between punctuation and a letter (e.g. "**標籤：**文字", "您**[待填]**"). CJK has no spaces, so it bites."""
    import unicodedata
    punct = lambda ch: unicodedata.category(ch)[0] in "PS"  # noqa: E731
    letter = lambda ch: not ch.isspace() and not punct(ch)  # noqa: E731
    bad = []
    for line in text.splitlines():
        for m in re.finditer(r"\*\*(.+?)\*\*", line):
            before = line[m.start() - 1] if m.start() else " "
            after = line[m.end()] if m.end() < len(line) else " "
            inner = m.group(1)
            if (letter(before) and punct(inner[0])) or (punct(inner[-1]) and letter(after)):
                bad.append(m.group(0))
    return bad


def test_consent_files():
    en, zh = ((STUDY / f"consent_{l}.md").read_text(encoding="utf-8") for l in ("en", "zh"))
    assert en.startswith("**Pilot test") and "NTCU" in en and "NTUE" not in en + zh
    assert "國立臺中教育大學" in zh and "數位內容科技學系" in zh and "臺北" not in zh
    # the independent-contact placeholder stays unfilled in both languages until a real office is named
    assert "**[appropriate independent ethics office and contact details]**" in en
    assert "**[待填：適當的獨立倫理審查單位及聯絡方式]**" in zh
    assert not bad_bold(en)
    for f in ("consent_zh.md", "consent_zh_FINAL_TEMPLATE.md"):
        t = (STUDY / f).read_text(encoding="utf-8")
        assert t.startswith("<!-- Translation to be checked by a native speaker before use. -->")
        assert "[待填：呂教授中文姓名]" in t and "[待填：實驗室中文名稱]" in t
        assert not bad_bold(t), (f, bad_bold(t))
    sys.path.insert(0, str(STUDY / "space"))
    import build
    assert not [f for f in build.FILES if "TEMPLATE" in f]  # templates never ship to the Space


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("optin", [False, True])
def test_consent_to_code_to_instructions(db, lang, optin):
    """consent (two checkboxes) -> participant-code screen -> instructions, EN and zh-TW."""
    import sim
    at = sim.new_app()
    at.radio(key="lang").set_value(lang).run()
    assert at.checkbox(key="consent_ok").label == si.t(lang, "consent_check_pilot")
    box = at.checkbox(key="consent_optin")
    assert box.label == si.t(lang, "consent_optin") and box.value is False  # optional, unticked by default
    # the optional box alone does not start the study
    box.check(); at.button(key="consent_go").click().run()
    assert at.session_state["phase"] == "consent" and at.error and not store.SQLiteStorage(db).participants()
    if not optin:
        at.checkbox(key="consent_optin").uncheck()
    at.checkbox(key="consent_ok").check(); at.button(key="consent_go").click().run()
    assert not at.exception and at.session_state["phase"] == "pcode"
    code = at.session_state["pcode"]
    assert CODE_RE.match(code)
    html = " ".join(m.value for m in at.markdown)
    assert code in html and si.t(lang, "pcode_label") in html and si.t(lang, "pcode_body") in html
    assert "_outputs_read" not in at.session_state and not at.get("plotly_chart")
    (p,) = store.SQLiteStorage(db).participants()
    assert p["participant_code"] == code and p["consent_pilot"] == 1 and p["future_use_optin"] == int(optin)
    assert p["consent_version"] == "pilot-1.0-2026-11-14" and p["phase"] == "pilot" and p["status"] == "incomplete"
    at.button(key="pcode_go").click().run()
    assert not at.exception and at.session_state["phase"] == "instructions"


def test_stale_storage_module_is_reloaded(db):
    """The crash: a running server kept an old `storage` module without new_participant_code."""
    import types
    import sim
    saved = sys.modules.get("storage")
    sys.modules["storage"] = types.ModuleType("storage")  # an old/foreign module under the same name
    try:
        at = sim.new_app()
        at.checkbox(key="consent_ok").check(); at.button(key="consent_go").click().run()
        assert not at.exception and at.session_state["phase"] == "pcode"
        assert Path(sys.modules["storage"].__file__).resolve() == STUDY / "storage.py"
    finally:
        if saved is not None:
            sys.modules["storage"] = saved


def test_codes_unique_and_normalised():
    taken = {store.new_participant_code() for _ in range(2000)}
    assert len(taken) == 2000 and all(CODE_RE.match(c) for c in taken)
    assert store.new_participant_code(taken) not in taken
    assert store.normalize_code(" k7qm 3htr ") == store.normalize_code("K7QM-3HTR") == "K7QM-3HTR"
    for bad in ("K7QM-3HT", "K7QM-3HTO", "", "AF-1234ABCD"):
        with pytest.raises(ValueError):
            store.normalize_code(bad)


def test_sqlite_adds_new_columns_to_an_old_database(tmp_path):
    import sqlite3
    old = [f for f in store.PARTICIPANT_FIELDS if f != "participant_code"]
    with sqlite3.connect(tmp_path / "old.sqlite") as c:
        c.execute(f"CREATE TABLE participants ({', '.join(f + (' PRIMARY KEY' if f == 'participant_id' else '') for f in old)})")
        c.execute("INSERT INTO participants (participant_id, lang) VALUES ('u0', 'en')")
    b = store.SQLiteStorage(tmp_path / "old.sqlite")
    b.upsert_participant({"participant_id": "u1", "participant_code": "K7QM-3HTR"})
    assert {p["participant_id"]: p["participant_code"] for p in b.participants()} == {"u0": None, "u1": "K7QM-3HTR"}


def _two_participants(b):
    for pid, code in (("u1", "AAAA-2222"), ("u2", "BBBB-3333")):
        b.upsert_participant({"participant_id": pid, "participant_code": code, "lang": "en", "consented_at": "t0"})
        b.upsert_participant({"participant_id": pid, "completed_at": "t9"})  # a second participants record
        for i in range(3):
            b.add_trial({"participant_id": pid, "trial_index": i, "kind": "stimulus", "stimulus_id": f"S0{i + 1}"})


@pytest.mark.parametrize("kind", ["sqlite", "jsonl"])
def test_withdraw_cli_deletes_only_that_participant(tmp_path, kind, capsys):
    path = tmp_path / ("w.sqlite" if kind == "sqlite" else "hf")
    b = store.SQLiteStorage(path) if kind == "sqlite" else store.HFDatasetStorage(path, scheduler=None)
    _two_participants(b)
    log = tmp_path / "withdrawals.log"
    args = [f"--{kind}", str(path), "--log", str(log)]
    assert withdraw.main(["aaaa 2222", "--dry-run", *args]) == 0 and not log.exists()  # dry run: nothing changes
    assert len(b.trials()) == 6
    assert withdraw.main(["aaaa 2222", *args]) == 0
    n_p = 1 if kind == "sqlite" else 2  # JSONL: the consent record and the later update are separate lines
    assert f"deleted {n_p} participant row(s) and 3 trial row(s)" in capsys.readouterr().out
    assert {p["participant_id"] for p in b.participants()} == {"u2"}
    assert {t["participant_id"] for t in b.trials()} == {"u2"} and len(b.trials()) == 3
    raw = path.read_bytes() if kind == "sqlite" else b"".join(p.read_bytes() for p in (path / "data").iterdir())
    assert b"u1" not in raw and b"AAAA-2222" not in raw  # gone from the file itself (VACUUM / rewrite)
    # the log holds the code and the date, and nothing else
    (line,) = log.read_text().splitlines()
    date, code = line.split("\t")
    assert code == "AAAA-2222" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
    # unknown or already-withdrawn code: exit 1, nothing logged
    assert withdraw.main(["AAAA-2222", *args]) == 1 and len(log.read_text().splitlines()) == 1
    assert withdraw.main(["not-a-code", *args]) == 2


def test_withdraw_uses_the_active_backend(tmp_path, monkeypatch):
    for k in ("STUDY_MODE", "CONSENT_APPROVED", "DATASET_REPO", "HF_TOKEN", "SPACE_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("STUDY_STORAGE", "sqlite")
    monkeypatch.setenv("STUDY_DB", str(tmp_path / "active.sqlite"))
    _two_participants(store.from_env())
    assert withdraw.main(["BBBB-3333", "--log", str(tmp_path / "l.log")]) == 0
    assert {p["participant_id"] for p in store.SQLiteStorage(tmp_path / "active.sqlite").participants()} == {"u1"}


class FakeHfApi:
    """Stands in for HfApi: a dataset repo held in memory."""
    def __init__(self, files, tmp):
        self.files, self.tmp, self.commits, self.squashed = dict(files), tmp, [], 0

    def list_repo_files(self, repo_id, repo_type):
        return list(self.files) + ["README.md"]

    def hf_hub_download(self, repo_id, filename, repo_type, token=None, cache_dir=None):
        assert cache_dir and Path(cache_dir).exists()  # a temporary dir, not the shared cache
        p = Path(cache_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.files[filename], encoding="utf-8")
        return str(p)

    def create_commit(self, repo_id, operations, *, commit_message, repo_type, token=None):
        assert "u1" not in commit_message and "AAAA" not in commit_message
        for op in operations:
            if type(op).__name__ == "CommitOperationDelete":
                del self.files[op.path_in_repo]
            else:
                self.files[op.path_in_repo] = op.path_or_fileobj.decode("utf-8")
        self.commits.append(operations)

    def super_squash_history(self, repo_id, *, commit_message=None, repo_type=None, token=None):
        self.squashed += 1


def test_withdraw_hf_dataset(tmp_path):
    local = store.HFDatasetStorage(tmp_path / "a", scheduler=None)
    _two_participants(local)
    files = {f"data/{p.name}": p.read_text(encoding="utf-8") for p in (tmp_path / "a" / "data").iterdir()}
    only_u1 = '{"participant_id": "u1", "trial_index": 9}\n'
    files["data/trials-otherproc.jsonl"] = only_u1  # a file that becomes empty is deleted
    api = FakeHfApi(files, tmp_path)
    dry = withdraw.withdraw_hf("x/y", "AAAA-2222", dry_run=True, api=api)
    assert dry["participants"] == 2 and dry["trials"] == 4 and not api.commits and not api.squashed
    res = withdraw.withdraw_hf("x/y", "AAAA-2222", api=api)
    assert res["found"] and res["participants"] == 2 and res["trials"] == 4 and res["files"] == 3
    assert len(api.commits) == 1 and api.squashed == 1
    assert "data/trials-otherproc.jsonl" not in api.files
    assert not any('"u1"' in t for t in api.files.values()) and any('"u2"' in t for t in api.files.values())
    assert not withdraw.withdraw_hf("x/y", "AAAA-2222", api=api)["found"]  # idempotent
    assert len(api.commits) == 1


# ---------------------------------------------------------------- skips, incomplete sessions, pilot data handling
import analysis  # noqa: E402
import export  # noqa: E402
import purge_pilot  # noqa: E402


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_skip_button_on_every_step(db, lang):
    import sim
    at = sim.new_app()
    seen = set()

    def look(at, phase, step, trial):
        if phase in ("practice", "trial"):
            idx = 0 if phase == "practice" else at.session_state["ti"] + 1
            b = at.button(key=f"skip_{step}_{idx}")
            assert b.label == si.t(lang, "skip_artwork")
            seen.add(step)

    pid, code = sim.run_participant(at, lang=lang, seed=3, on_step=look, skip_rate=0.25)
    assert seen == {"blind", "reveal", "ab", "attn"}
    b = store.SQLiteStorage(db)
    (p,) = b.participants()
    assert p["status"] == "complete" and p["completion_code"] == code
    trials = b.trials()
    assert len(trials) == 1 + N_TRIALS + 2  # skipping moves on to the next trial; nothing is lost or repeated
    sk = [t for t in trials if t["skipped"]]
    assert sk and all(t["status"] == "skipped" and t["skipped_step"] in ("blind", "reveal", "ab", "attn") for t in sk)
    assert all(t["status"] == "complete" and not t["skipped"] for t in trials if t not in sk)
    assert all(t["phase"] == "pilot" for t in trials)
    for t in sk:  # the skipped step itself holds no answer
        if t["skipped_step"] == "blind":
            assert t["blind_emotion"] is None
        if t["skipped_step"] == "reveal":
            assert t["blind_emotion"] is not None and t["plausibility"] is None
        if t["skipped_step"] == "ab":
            assert t["ab_choice"] is None


def test_closing_midway_keeps_submitted_steps(db):
    import sim
    at = sim.new_app()
    at.checkbox(key="consent_ok").check(); at.button(key="consent_go").click().run()
    at.button(key="pcode_go").click().run()
    at.button(key="go_practice").click().run()
    at.radio(key="b_emo_0").set_value("awe"); at.radio(key="b_str_0").set_value(4)
    sim._click(at, si.t("en", "submit"))
    assert at.session_state["step"] == "reveal"
    # ... and the participant closes the page here
    b = store.SQLiteStorage(db)
    (p,) = b.participants()
    (t,) = b.trials()
    assert p["status"] == "incomplete" and p["completed_at"] is None
    assert t["status"] == "incomplete" and t["blind_emotion"] == "awe" and t["blind_strength"] == 4 and t["plausibility"] is None


def test_pilot_database_is_owner_only(tmp_path):
    d = tmp_path / "pilot_real"
    store.SQLiteStorage(d / "pilot.sqlite")
    assert (d.stat().st_mode & 0o777) == 0o700 and ((d / "pilot.sqlite").stat().st_mode & 0o777) == 0o600
    assert store.default_sqlite("pilot") == STUDY / "private/pilot_real/pilot.sqlite"


def test_old_rows_are_tagged(tmp_path):
    import sqlite3
    old_p = [f for f in store.PARTICIPANT_FIELDS if f not in ("consent_pilot", "future_use_optin", "consent_version", "phase", "status")]
    old_t = [f for f in store.TRIAL_FIELDS if f not in ("status", "skipped", "skipped_step", "phase")]
    with sqlite3.connect(tmp_path / "o.sqlite") as c:
        c.execute(f"CREATE TABLE participants ({', '.join(old_p)})")
        c.execute(f"CREATE TABLE trials ({', '.join(old_t)})")
        c.execute("INSERT INTO participants (participant_id, study_mode, completed_at) VALUES ('a', 'pilot', NULL), ('b', 'pilot', 't')")
        c.execute("INSERT INTO trials (participant_id, trial_index, study_mode) VALUES ('a', 1, 'pilot')")
    b = store.SQLiteStorage(tmp_path / "o.sqlite")
    ps = {p["participant_id"]: p for p in b.participants()}
    assert ps["a"]["status"] == "incomplete" and ps["b"]["status"] == "complete"
    assert {p["phase"] for p in ps.values()} == {"pilot"} and ps["a"]["future_use_optin"] is None  # never invented
    (t,) = b.trials()
    assert t["phase"] == "pilot" and t["status"] == "complete" and t["skipped"] == 0


def test_export_has_the_new_columns(tmp_path):
    b = store.SQLiteStorage(tmp_path / "e.sqlite")
    b.upsert_participant({"participant_id": "u", "consent_version": "v", "future_use_optin": True, "phase": "pilot", "status": "complete"})
    b.add_trial({"participant_id": "u", "trial_index": 1, "skipped": True, "skipped_step": "ab", "status": "skipped", "phase": "pilot"})
    t, p = export.export(b, tmp_path / "out")
    pc = pd.read_csv(tmp_path / "out/participants.csv").columns
    tc = pd.read_csv(tmp_path / "out/trials.csv").columns
    assert {"consent_version", "future_use_optin", "consent_pilot", "phase", "status"} <= set(pc)
    assert {"skipped", "skipped_step", "status", "phase"} <= set(tc)


def test_analysis_treats_skips_as_missing():
    t = pd.DataFrame([
        {"participant_id": "a", "kind": "stimulus", "stimulus_id": "S01", "blind_emotion": "awe", "plausibility": 4,
         "ab_choice_model": "r1", "attention_rating": 3, "skipped": 0, "skipped_step": None},
        {"participant_id": "a", "kind": "stimulus", "stimulus_id": "S02", "blind_emotion": "fear", "plausibility": 5,
         "ab_choice_model": "r3", "attention_rating": 2, "skipped": 1, "skipped_step": "reveal"},
        {"participant_id": "a", "kind": "stimulus", "stimulus_id": "S03", "blind_emotion": None, "plausibility": None,
         "ab_choice_model": None, "attention_rating": None, "skipped": 1, "skipped_step": "blind"},
    ])
    m = analysis.mask_skips(t)
    assert m.loc[0, "plausibility"] == 4 and m.loc[0, "ab_choice_model"] == "r1"
    assert m.loc[1, "blind_emotion"] == "fear"  # submitted before the skip: kept
    assert pd.isna(m.loc[1, "plausibility"]) and pd.isna(m.loc[1, "ab_choice_model"]) and pd.isna(m.loc[1, "attention_rating"])
    assert analysis.preference(m) == analysis.preference(m.iloc[:1]) | {}  # the skipped A/B is not an answer
    assert analysis.human_dist(m.dropna(subset=["blind_emotion"])).index.tolist() == ["S01", "S02"]
    # a skipped attention check is not a pass
    checks = pd.DataFrame([{"participant_id": "a", "kind": "check", "check_pass": None, "skipped": 1, "skipped_step": "blind"},
                           {"participant_id": "a", "kind": "check", "check_pass": 1, "skipped": 0, "skipped_step": None}])
    p = pd.DataFrame([{"participant_id": "a", "completed_at": "t", "phase": "production", "study_mode": "production"}])
    assert analysis.included(pd.concat([t, checks]), p).empty
    assert len(analysis.included(pd.concat([t, checks]), p, max_failed_checks=1)) == 3


def test_analysis_pilot_rule():
    p = pd.DataFrame([{"participant_id": "x", "phase": "pilot", "study_mode": "pilot", "future_use_optin": 1},
                      {"participant_id": "y", "phase": "pilot", "study_mode": "pilot", "future_use_optin": 0},
                      {"participant_id": "z", "phase": None, "study_mode": "pilot", "future_use_optin": None},
                      {"participant_id": "w", "phase": "production", "study_mode": "production", "future_use_optin": None}])
    assert analysis.pilot_filter(p) == {"w"}
    assert analysis.pilot_filter(p, "optin") == {"w", "x"}
    assert analysis.pilot_filter(p, "all") == {"w", "x", "y", "z"}


def _pilot_db(path):
    b = store.SQLiteStorage(path)
    for pid, code, optin, phase in (("p1", "AAAA-2222", True, "pilot"), ("p2", "BBBB-3333", False, "pilot"),
                                    ("p3", "CCCC-4444", None, "pilot"), ("q1", "DDDD-5555", False, "production")):
        b.upsert_participant({"participant_id": pid, "participant_code": code, "future_use_optin": optin, "phase": phase,
                              "study_mode": phase, "status": "complete", "consented_at": "t"})
        b.add_trial({"participant_id": pid, "trial_index": 1, "phase": phase, "study_mode": phase})
    return b


def test_purge_pilot_asks_and_never_runs_unattended(tmp_path, capsys):
    db = tmp_path / "pilot.sqlite"
    b = _pilot_db(db)
    args = ["--sqlite", str(db)]
    assert purge_pilot.main(args, interactive=False) == 3  # no terminal: refuses
    assert purge_pilot.main(args, ask=lambda _: "yes", interactive=True) == 1  # wrong phrase: cancels
    assert purge_pilot.main([*args, "--dry-run"], interactive=True) == 0
    assert len(b.participants()) == 4
    out = capsys.readouterr().out
    assert "will delete 3 pilot participant(s) and 3 trial row(s)" in out and "AAAA-2222" in out and "DDDD-5555" not in out
    assert purge_pilot.main(args, ask=lambda _: purge_pilot.CONFIRM, interactive=True) == 0
    assert {p["participant_id"] for p in b.participants()} == {"q1"}  # production rows untouched
    assert {t["participant_id"] for t in b.trials()} == {"q1"}
    assert b"AAAA-2222" not in db.read_bytes()


def test_purge_pilot_keep_optin(tmp_path, capsys):
    db = tmp_path / "pilot.sqlite"
    b = _pilot_db(db)
    assert purge_pilot.main(["--sqlite", str(db), "--keep-optin"], ask=lambda _: purge_pilot.CONFIRM, interactive=True) == 0
    assert "will keep 1 opted-in pilot participant(s): AAAA-2222" in capsys.readouterr().out
    assert {p["participant_id"] for p in b.participants()} == {"p1", "q1"}
    assert {t["participant_id"] for t in b.trials()} == {"p1", "q1"}
