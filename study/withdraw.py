"""Withdraw one participant: delete every row they produced, given the participant code they were shown after
consent (e.g. K7QM-3HTR), from the ACTIVE storage backend, and log the deletion.

usage:
  python study/withdraw.py CODE                    backend from the environment, exactly as the app picks it
                                                   (STUDY_MODE, STUDY_STORAGE, STUDY_DB, DATASET_REPO, HF_TOKEN)
  python study/withdraw.py CODE --sqlite FILE      a SQLite database (the local pilot: study/private/pilot_real/pilot.sqlite)
  python study/withdraw.py CODE --hf-repo REPO     a private HF dataset repo (token from the HF_TOKEN environment variable)
  python study/withdraw.py CODE --jsonl FOLDER     a local JSONL folder (STUDY_DATA_DIR of a running process, or a copy)
  --dry-run                                        report what would be deleted; change and log nothing

What is deleted: the participant's rows in `participants` and `trials` (SQLite), or their lines in every
data/participants-*.jsonl and data/trials-*.jsonl (HF / JSONL). For a SQLite file the database is VACUUMed afterwards.
For an HF dataset the files are rewritten in one commit and the repo history is then squashed
(HfApi.super_squash_history), because the earlier commits would otherwise still contain the rows; pass
--keep-history to skip the squash.

HF caveat: a running Space keeps its own local JSONL file and CommitScheduler re-uploads it whenever it changes. If
the participant used a process that is still running, restart the Space and run this script again; it is idempotent
and reports 0 rows once nothing is left.

Log: one line per withdrawal, "<UTC date>\t<participant code>", appended to study/private/withdrawals.log (or --log).
Nothing else (no participant_id, no answers, no row counts) is written to the log.

Also check exports: rows already exported (export.py) are not touched; this script lists any CSV under study/private/
that still contains the participant, so you can re-export or delete it.
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
import storage as store  # noqa: E402

DEFAULT_LOG = STUDY / "private" / "withdrawals.log"


def log_withdrawal(code: str, log: Path = DEFAULT_LOG) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).date().isoformat()}\t{code}\n")


def find_participant(participants, code: str) -> str | None:
    ids = {p["participant_id"] for p in participants if p.get("participant_code") == code}
    if len(ids) > 1:
        raise RuntimeError(f"participant code {code} matches {len(ids)} participants; resolve by hand")
    return next(iter(ids), None)


# ---------------------------------------------------------------- backends
def withdraw_local(backend, code: str, dry_run=False) -> dict:
    """SQLiteStorage or a local HFDatasetStorage folder (scheduler=None)."""
    pid = find_participant(backend.participants(), code)
    if pid is None:
        return {"found": False, "participant_id": None, "participants": 0, "trials": 0}
    if dry_run:
        n_t = sum(t["participant_id"] == pid for t in backend.trials())
        return {"found": True, "participant_id": pid, "participants": 1, "trials": n_t}
    n_p, n_t = backend.delete_participant(pid)
    return {"found": True, "participant_id": pid, "participants": n_p, "trials": n_t}


def withdraw_hf(repo_id: str, code: str, token=None, dry_run=False, squash=True, api=None) -> dict:
    """Rewrite the dataset repo's JSONL files without the participant's lines (one commit), then squash history."""
    import json
    import tempfile
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    api = api or HfApi(token=token)
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f.startswith(("data/participants-", "data/trials-")) and f.endswith(".jsonl")]
    with tempfile.TemporaryDirectory() as tmp:  # not the shared HF cache, so no copy of the rows is left behind
        texts = {f: Path(api.hf_hub_download(repo_id, f, repo_type="dataset", token=token, cache_dir=tmp))
                 .read_text(encoding="utf-8") for f in files}
    participants = [json.loads(l) for f, t in texts.items() if f.startswith("data/participants-")
                    for l in t.splitlines() if l.strip()]
    pid = find_participant(participants, code)
    out = {"found": pid is not None, "participant_id": pid, "participants": 0, "trials": 0, "files": 0}
    if pid is None:
        return out
    ops = []
    for f, t in texts.items():
        kept, n = store.drop_participant_lines(t, pid)
        if n:
            out["participants" if f.startswith("data/participants-") else "trials"] += n
            out["files"] += 1
            ops.append(CommitOperationAdd(f, kept.encode("utf-8")) if kept else CommitOperationDelete(f))
    if ops and not dry_run:
        api.create_commit(repo_id, repo_type="dataset", operations=ops, token=token,
                          commit_message="Withdraw one participant")  # no code or id in the commit message
        if squash:
            api.super_squash_history(repo_id, repo_type="dataset", token=token,
                                     commit_message="Squash history after a participant withdrawal")
    return out


def exported_copies(pid: str, root: Path = STUDY / "private") -> list:
    """CSV files under study/private/ that still mention the participant (exports made before the withdrawal)."""
    hits = []
    for p in sorted(root.rglob("*.csv")) if root.exists() else []:
        try:
            if pid in p.read_text(encoding="utf-8", errors="ignore"):
                hits.append(p)
        except OSError:
            pass
    return hits


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("code", help="the participant code, e.g. K7QM-3HTR (case, spaces and dashes are ignored)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sqlite", help="SQLite database file")
    g.add_argument("--hf-repo", help="HF dataset repo id (token from HF_TOKEN)")
    g.add_argument("--jsonl", help="local JSONL dataset folder")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-history", action="store_true", help="HF only: do not squash the repo history")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    a = ap.parse_args(argv)
    try:
        code = store.normalize_code(a.code)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    if a.sqlite:
        if not Path(a.sqlite).exists():
            print(f"no such database: {a.sqlite}", file=sys.stderr)
            return 2
        where, res = f"SQLite {a.sqlite}", withdraw_local(store.SQLiteStorage(a.sqlite), code, a.dry_run)
    elif a.jsonl:
        where, res = f"JSONL {a.jsonl}", withdraw_local(store.HFDatasetStorage(a.jsonl, scheduler=None), code, a.dry_run)
    elif a.hf_repo:
        where = f"HF dataset {a.hf_repo}"
        res = withdraw_hf(a.hf_repo, code, os.environ.get("HF_TOKEN"), a.dry_run, squash=not a.keep_history)
    else:  # the active backend, chosen exactly as the app chooses it
        cfg = store.deployment()
        if cfg["problems"]:
            print("storage settings are incomplete: " + "; ".join(cfg["problems"]), file=sys.stderr)
            return 2
        if cfg["storage"] == "sqlite":
            path = os.environ.get("STUDY_DB") or str(store.default_sqlite(cfg["mode"]))
            if not Path(path).exists():
                print(f"no such database: {path} (set STUDY_DB or pass --sqlite)", file=sys.stderr)
                return 2
            where, res = f"SQLite {path}", withdraw_local(store.SQLiteStorage(path), code, a.dry_run)
        else:
            where = f"HF dataset {cfg['repo']}"
            res = withdraw_hf(cfg["repo"], code, os.environ["HF_TOKEN"], a.dry_run, squash=not a.keep_history)

    if not res["found"]:
        print(f"{code}: no participant with this code in {where}. Nothing deleted, nothing logged.")
        return 1
    verb = "would delete" if a.dry_run else "deleted"
    print(f"{code}: {verb} {res['participants']} participant row(s) and {res['trials']} trial row(s) from {where}.")
    if not a.dry_run:
        log_withdrawal(code, Path(a.log))
        print(f"logged in {a.log}")
        if a.hf_repo or where.startswith("HF"):
            print("If the Space is running, restart it and run this again: a live process can re-upload its local file.")
    for p in exported_copies(res["participant_id"]):
        print(f"warning: an earlier export still contains this participant: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
