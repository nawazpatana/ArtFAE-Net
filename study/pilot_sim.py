"""Local pilot simulation: N synthetic participants complete the real study app (AppTest, no browser), into a fresh
SQLite file; then the data are exported to CSV and the analysis skeleton is smoke-tested on them.
Synthetic answers are random: the numbers mean nothing.

usage:  artfae_demo/.venv/bin/python study/pilot_sim.py [--n 5] [--out study/private/pilot_sim]
"""
import argparse
import json
import os
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=str(STUDY / "private/pilot_sim"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "pilot_sim.sqlite"
    db.unlink(missing_ok=True)
    os.environ["STUDY_STORAGE"], os.environ["STUDY_DB"] = "sqlite", str(db)
    import streamlit as st
    import sim
    import export
    import analysis
    import storage as store
    st.cache_resource.clear()
    plan = [("en", False), ("zh", False), ("en", True), ("zh", False), ("en", False)]
    for i in range(a.n):
        lang, fail = plan[i % len(plan)]
        pid, code = sim.run_participant(sim.new_app(), lang=lang, seed=100 + i, fail_checks=fail)
        print(f"participant {i + 1}: lang={lang} fail_checks={fail} id={pid[:8]}… code={code}")
    t, p = export.export(store.SQLiteStorage(db), out / "export")
    print(f"exported {len(p)} participants, {len(t)} trial rows -> {out / 'export'}")
    res, ps = analysis.analyse(out / "export", pilot="all")  # synthetic pilot rows: smoke test only
    (out / "analysis_synthetic.json").write_text(json.dumps(res, indent=1, default=float))
    keep = {k: res[k] for k in ("n_participants", "excluded_participants", "n_trials", "n_stimuli", "preference")}
    print("analysis smoke test (synthetic, meaningless):", json.dumps(keep))
    print("sets per participant:", p[["arrival_index", "set_index", "lang", "checks_passed"]].to_dict("records"))


if __name__ == "__main__":
    main()
