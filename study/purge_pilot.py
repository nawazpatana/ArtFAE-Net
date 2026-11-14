"""Delete pilot data (phase = "pilot"), by hand only.

usage:
  python study/purge_pilot.py                  delete ALL pilot participants and their trials
  python study/purge_pilot.py --keep-optin     delete every pilot participant EXCEPT those with future_use_optin = true
  --sqlite FILE                                the pilot database (default: study/private/pilot_real/pilot.sqlite)
  --dry-run                                    print what would be deleted and stop

It prints what it will delete, then asks you to type the confirmation phrase. It refuses to run without an
interactive terminal and has no "--yes" flag, so it can never run from cron, CI or another script.
Deadline (study/README.md): unless an ethics committee has approved using opted-in pilot answers, delete all pilot
data with this script by 31 March 2027. Rows tagged phase = "production" are never touched.

Afterwards the database is VACUUMed, so the deleted rows do not stay in the file. Exports made earlier (export.py)
are separate files: the script lists CSVs under study/private/ that still contain deleted participants.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
import storage as store  # noqa: E402

CONFIRM = "DELETE PILOT DATA"


def plan(db: Path, keep_optin: bool) -> dict:
    """-> {"delete": [(participant_id, participant_code, status, future_use_optin)], "keep": [...], "trials": n}"""
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT participant_id, participant_code, status, future_use_optin FROM participants "
                         "WHERE COALESCE(phase, study_mode) = 'pilot' ORDER BY consented_at").fetchall()
        delete = [r for r in rows if not (keep_optin and r[3] in (1, True, "1", "true", "True"))]
        keep = [r for r in rows if r not in delete]
        ids = [r[0] for r in delete]
        n_trials = c.execute(f"SELECT COUNT(*) FROM trials WHERE participant_id IN ({','.join('?' * len(ids))})",
                             ids).fetchone()[0] if ids else 0
        # pilot trials whose participant row is already gone (orphans) are deleted too
        n_orphans = c.execute(
            "SELECT COUNT(*) FROM trials WHERE COALESCE(phase, study_mode) = 'pilot' AND participant_id NOT IN "
            "(SELECT participant_id FROM participants)").fetchone()[0]
    return {"delete": delete, "keep": keep, "trials": n_trials, "orphan_trials": n_orphans}


def purge(db: Path, p: dict) -> tuple:
    ids = [r[0] for r in p["delete"]]
    with sqlite3.connect(db) as c:
        q = ",".join("?" * len(ids))
        n_t = c.execute(f"DELETE FROM trials WHERE participant_id IN ({q})", ids).rowcount if ids else 0
        n_t += c.execute("DELETE FROM trials WHERE COALESCE(phase, study_mode) = 'pilot' AND participant_id NOT IN "
                         "(SELECT participant_id FROM participants)").rowcount
        n_p = c.execute(f"DELETE FROM participants WHERE participant_id IN ({q})", ids).rowcount if ids else 0
    with sqlite3.connect(db) as c:
        c.execute("VACUUM")
    return n_p, n_t


def main(argv=None, ask=input, interactive=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--keep-optin", action="store_true")
    ap.add_argument("--sqlite", default=str(store.default_sqlite("pilot")))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    db = Path(a.sqlite)
    if not db.exists():
        print(f"no such database: {db}")
        return 2
    p = plan(db, a.keep_optin)
    print(f"database: {db}")
    print(f"mode: {'delete every pilot participant EXCEPT those who opted in' if a.keep_optin else 'delete ALL pilot data'}")
    print(f"will delete {len(p['delete'])} pilot participant(s) and {p['trials'] + p['orphan_trials']} trial row(s):")
    for pid, code, status, optin in p["delete"]:
        print(f"  {code or '(no code)':10}  status={status or '?':10}  future_use_optin={bool(optin) if optin is not None else '-'}")
    if a.keep_optin:
        print(f"will keep {len(p['keep'])} opted-in pilot participant(s): {', '.join(r[1] or '(no code)' for r in p['keep']) or '-'}")
    if not p["delete"] and not p["orphan_trials"]:
        print("nothing to delete.")
        return 0
    if a.dry_run:
        print("dry run: nothing deleted.")
        return 0
    if not (sys.stdin.isatty() if interactive is None else interactive):
        print("refusing to run without an interactive terminal (this script never runs automatically).")
        return 3
    if ask(f'type "{CONFIRM}" to delete, anything else to cancel: ').strip() != CONFIRM:
        print("cancelled: nothing deleted.")
        return 1
    n_p, n_t = purge(db, p)
    print(f"deleted {n_p} participant row(s) and {n_t} trial row(s); database vacuumed.")
    for f in sorted((STUDY / "private").rglob("*.csv")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        if any(r[0] in text for r in p["delete"]):
            print(f"warning: an earlier export still contains deleted participants: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
