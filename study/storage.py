"""Response storage: one interface, two backends.

  SQLiteStorage     local pilot (default): study/private/pilot.sqlite
  HFDatasetStorage  deployment: append-only JSON Lines in a local folder that huggingface_hub's CommitScheduler
                    pushes to a PRIVATE dataset repo every few minutes. The token comes from the environment
                    (HF_TOKEN, a Space secret); it is never written in code or files.

Data model (long format): one row per trial in `trials`, plus `participants` (one row per participant; in the
JSONL backend every update is appended and the latest record wins). Only random identifiers identify people: the
internal participant_id (a UUID) and the participant_code shown to the participant once, right after consent, which
they need to withdraw (study/withdraw.py). No names, emails or IP addresses are collected or stored.

Configuration (environment variables) is read and checked by deployment() below:
  STUDY_MODE, CONSENT_APPROVED, STUDY_STORAGE, DATASET_REPO, HF_TOKEN, STUDY_DB (sqlite file), STUDY_DATA_DIR (local JSONL folder),
  STUDY_PUSH_MINUTES (how often CommitScheduler pushes, default 2).
"""
import json
import os
import re
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parent

TRIAL_FIELDS = [
    "participant_id", "trial_index", "kind", "stimulus_id", "lang",
    "blind_emotion", "blind_strength", "blind_reason", "rt_blind_ms",
    "check_expect_emotion", "check_expect_strength", "check_pass",
    "plausibility", "rt_reveal_ms",
    "ab_left_model", "ab_choice", "ab_choice_model", "rt_ab_ms",
    "attention_rating", "attention_skipped", "rt_attention_ms",
    "created_at", "study_mode",
    # status: complete | incomplete (left mid-trial; the steps already submitted are kept) | skipped
    # skipped / skipped_step: "Skip this artwork" and the step it was pressed on (blind, reveal, ab, attn);
    # that step and the later ones are missing data (analysis.py), not answers
    "status", "skipped", "skipped_step", "phase",
]
PARTICIPANT_FIELDS = [
    "participant_id", "participant_code", "arrival_index", "set_index", "plan", "lang", "consented_at", "started_at",
    "completed_at", "completion_code", "age_band", "gender", "art_experience", "country_culture", "native_language",
    *(f"sus_{i}" for i in range(1, 11)), "comment", "checks_passed", "app_version", "study_mode",
    # consent record; status: incomplete (from consent until the questionnaire is submitted) | complete
    "consent_pilot", "future_use_optin", "consent_version", "phase", "status",
]


PILOT_CONSENT_VERSION = "pilot-1.0-2026-11-14"


def default_sqlite(mode: str) -> Path:
    """Pilot data live apart from everything else, in study/private/pilot_real/ (folder 700, file 600)."""
    return STUDY / "private" / ("pilot_real/pilot.sqlite" if mode == "pilot" else f"{mode}.sqlite")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Participant codes: 8 characters in two groups (e.g. K7QM-3HTR), from an alphabet without look-alikes (no 0/O, 1/I/L),
# so they can be copied by hand. 31^8 ~ 8.5e11 codes.
CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"


def new_participant_code(taken=()) -> str:
    taken = set(taken)
    while True:
        raw = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
        code = f"{raw[:4]}-{raw[4:]}"
        if code not in taken:
            return code


def normalize_code(text: str) -> str:
    """'k7qm 3htr', 'K7QM-3HTR', ' k7qm3htr ' -> 'K7QM-3HTR'. Raises ValueError if it cannot be a participant code."""
    raw = re.sub(r"[\s\-_.]", "", str(text)).upper()
    if len(raw) != 8 or any(c not in CODE_ALPHABET for c in raw):
        raise ValueError(f"not a participant code: {text!r} (expected 8 characters like K7QM-3HTR)")
    return f"{raw[:4]}-{raw[4:]}"


class Storage:
    def upsert_participant(self, row: dict) -> None: ...
    def add_trial(self, row: dict) -> None: ...
    def participants(self) -> list: ...
    def trials(self) -> list: ...

    def next_arrival_index(self) -> int:
        return len(self.participants())

    def participant_codes(self) -> set:
        return {p["participant_code"] for p in self.participants() if p.get("participant_code")}

    def delete_participant(self, participant_id: str) -> tuple:
        """Remove every row of one participant. -> (participant rows removed, trial rows removed)."""
        raise NotImplementedError


class SQLiteStorage(Storage):
    def __init__(self, path, secure=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if secure:  # owner-only: folder 700, database 600 (SQLite gives its journal file the same mode)
            old = os.umask(0o077)
            try:
                self.path.touch(exist_ok=True)
            finally:
                os.umask(old)
            for target, mode in ((self.path.parent, 0o700), (self.path, 0o600)):
                try:
                    os.chmod(target, mode)
                except OSError as e:  # e.g. a shared folder we do not own
                    print(f"warning: could not chmod {target} to {oct(mode)}: {e}")
        with self._conn() as c:
            c.execute(f"CREATE TABLE IF NOT EXISTS participants ({', '.join(f + (' PRIMARY KEY' if f == 'participant_id' else '') for f in PARTICIPANT_FIELDS)})")
            c.execute(f"CREATE TABLE IF NOT EXISTS trials ({', '.join(TRIAL_FIELDS)}, PRIMARY KEY (participant_id, trial_index))")
            for table, fields in (("participants", PARTICIPANT_FIELDS), ("trials", TRIAL_FIELDS)):  # columns added later
                have = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
                for f in fields:
                    if f not in have:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {f}")
            # rows written before these columns existed: tag them from what is known, never invent consent answers
            c.execute("UPDATE participants SET phase = study_mode WHERE phase IS NULL")
            c.execute("UPDATE participants SET status = CASE WHEN completed_at IS NULL THEN 'incomplete' ELSE 'complete' END "
                      "WHERE status IS NULL")
            c.execute("UPDATE trials SET phase = study_mode WHERE phase IS NULL")
            c.execute("UPDATE trials SET status = 'complete', skipped = 0 WHERE status IS NULL")  # were saved only when done

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30)

    def upsert_participant(self, row):
        row = {k: row.get(k) for k in PARTICIPANT_FIELDS if k in row}
        cols = list(row)
        with self._lock, self._conn() as c:
            c.execute(f"INSERT INTO participants ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
                      f"ON CONFLICT(participant_id) DO UPDATE SET {', '.join(f'{k}=excluded.{k}' for k in cols if k != 'participant_id')}",
                      [row[k] for k in cols])

    def add_trial(self, row):
        row = {k: row.get(k) for k in TRIAL_FIELDS}
        with self._lock, self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO trials ({', '.join(TRIAL_FIELDS)}) VALUES ({', '.join('?' * len(TRIAL_FIELDS))})",
                      [row[k] for k in TRIAL_FIELDS])

    def _rows(self, table, fields):
        with self._conn() as c:
            return [dict(zip(fields, r)) for r in c.execute(f"SELECT {', '.join(fields)} FROM {table}")]

    def participants(self):
        return self._rows("participants", PARTICIPANT_FIELDS)

    def trials(self):
        return self._rows("trials", TRIAL_FIELDS)

    def next_arrival_index(self):
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM participants").fetchone()[0]

    def delete_participant(self, participant_id):
        with self._lock, self._conn() as c:
            n_t = c.execute("DELETE FROM trials WHERE participant_id = ?", (participant_id,)).rowcount
            n_p = c.execute("DELETE FROM participants WHERE participant_id = ?", (participant_id,)).rowcount
        with self._conn() as c:  # VACUUM rewrites the file, so the deleted rows do not linger in free pages
            c.execute("VACUUM")
        return n_p, n_t


class HFDatasetStorage(Storage):
    """Append-only JSONL in `folder`, mirrored to a private HF dataset by CommitScheduler.

    Each server process writes its own files (data/<kind>-<process uuid>.jsonl), so replicas never conflict.
    `scheduler=None` keeps everything local (tests, dry runs)."""

    def __init__(self, folder, repo_id=None, token=None, every_minutes=5, scheduler="auto"):
        self.folder = Path(folder)
        (self.folder / "data").mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:12]
        self.files = {"participants": self.folder / "data" / f"participants-{tag}.jsonl",
                      "trials": self.folder / "data" / f"trials-{tag}.jsonl"}
        self._lock = threading.Lock()
        self.repo_id = repo_id
        if scheduler == "auto":
            from huggingface_hub import CommitScheduler
            scheduler = CommitScheduler(repo_id=repo_id, repo_type="dataset", folder_path=self.folder,
                                        path_in_repo=".", every=every_minutes, private=True, token=token,
                                        allow_patterns=["data/*.jsonl"])
            self._pull_existing(token)
        self.scheduler = scheduler
        self._offset = 0  # participants already in the remote dataset (other processes, earlier restarts)

    def _pull_existing(self, token):
        """Count participants already in the dataset, so balanced assignment continues across restarts."""
        try:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi(token=token)
            seen = set()
            for f in api.list_repo_files(self.repo_id, repo_type="dataset"):
                if f.startswith("data/participants-") and f.endswith(".jsonl") and Path(f).name != self.files["participants"].name:
                    p = hf_hub_download(self.repo_id, f, repo_type="dataset", token=token)
                    seen |= {json.loads(l)["participant_id"] for l in Path(p).read_text().splitlines() if l.strip()}
            self._remote = seen
        except Exception:  # noqa: BLE001  (new repo, no network): start counting from zero
            self._remote = set()

    def _append(self, kind, row):
        line = json.dumps(row, ensure_ascii=False)
        lock = self.scheduler.lock if self.scheduler is not None else self._lock
        with lock:
            with self.files[kind].open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _read(self, kind):
        out = []
        for p in sorted((self.folder / "data").glob(f"{kind}-*.jsonl")):
            out += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        return out

    def upsert_participant(self, row):
        self._append("participants", {k: row.get(k) for k in PARTICIPANT_FIELDS if k in row} | {"_written_at": now()})

    def add_trial(self, row):
        self._append("trials", {k: row.get(k) for k in TRIAL_FIELDS})

    def participants(self):
        latest = {}
        for r in self._read("participants"):  # merge updates in write order
            latest.setdefault(r["participant_id"], {}).update({k: v for k, v in r.items() if not k.startswith("_")})
        return [{k: r.get(k) for k in PARTICIPANT_FIELDS} for r in latest.values()]

    def trials(self):
        latest = {}
        for r in self._read("trials"):
            latest[(r["participant_id"], r["trial_index"])] = r
        return list(latest.values())

    def next_arrival_index(self):
        local = {r["participant_id"] for r in self._read("participants")}
        return len(local | getattr(self, "_remote", set()))

    def delete_participant(self, participant_id):
        """Local JSONL files only. The copy in the HF dataset repo (and its history) is handled by study/withdraw.py."""
        lock = self.scheduler.lock if self.scheduler is not None else self._lock
        counts = {"participants": 0, "trials": 0}
        with lock:
            for kind in counts:
                for p in sorted((self.folder / "data").glob(f"{kind}-*.jsonl")):
                    kept, n = drop_participant_lines(p.read_text(encoding="utf-8"), participant_id)
                    if n:
                        p.write_text(kept, encoding="utf-8")
                        counts[kind] += n
        return counts["participants"], counts["trials"]


def drop_participant_lines(text: str, participant_id: str) -> tuple:
    """JSONL text -> (the text without that participant's lines, number of lines dropped)."""
    keep, n = [], 0
    for line in text.splitlines():
        if line.strip() and json.loads(line).get("participant_id") == participant_id:
            n += 1
        else:
            keep.append(line)
    return ("\n".join(keep) + "\n") if keep else "", n


# STUDY_MODE -> the only dataset repo that mode may write to
REPOS = {"pilot": "nawazishpatana/artfae-study-pilot", "production": "nawazishpatana/artfae-study-responses"}


def deployment() -> dict:
    """Read and check the deployment settings. -> {"mode", "storage", "repo", "problems"}; the app refuses to start
    while `problems` is not empty.

      STUDY_MODE        pilot (default) | production
      CONSENT_APPROVED  must be 1 for production (set only once the ethics-approved consent text is in place)
      STUDY_STORAGE     hf | sqlite; default hf on a Hugging Face Space (SPACE_ID is set), else sqlite
      DATASET_REPO      hf storage: must be exactly REPOS[STUDY_MODE]
      HF_TOKEN          hf storage: a Space secret with write access to that dataset (never in code or files)
    """
    mode = (os.environ.get("STUDY_MODE") or "pilot").strip().lower()
    problems = []
    if mode not in REPOS:
        problems.append(f"STUDY_MODE must be 'pilot' or 'production', not {mode!r}")
    if mode == "production" and os.environ.get("CONSENT_APPROVED") != "1":
        problems.append("STUDY_MODE=production also needs CONSENT_APPROVED=1, set only after the ethics-approved "
                        "consent text is in place")
    consent_version = PILOT_CONSENT_VERSION if mode == "pilot" else os.environ.get("CONSENT_VERSION")
    if mode == "production" and not consent_version:
        problems.append("STUDY_MODE=production also needs CONSENT_VERSION (the approved consent's version label)")
    kind = os.environ.get("STUDY_STORAGE") or ("hf" if os.environ.get("SPACE_ID") else "sqlite")
    repo = None
    if kind == "hf":
        repo = os.environ.get("DATASET_REPO")
        if mode in REPOS and repo != REPOS[mode]:
            problems.append(f"DATASET_REPO must be {REPOS[mode]} when STUDY_MODE={mode} (it is {repo or 'not set'})")
        if not os.environ.get("HF_TOKEN"):
            problems.append("the HF_TOKEN secret is not set")
    elif kind != "sqlite":
        problems.append(f"STUDY_STORAGE must be 'hf' or 'sqlite', not {kind!r}")
    return {"mode": mode, "storage": kind, "repo": repo, "consent_version": consent_version, "problems": problems}


def from_env(cfg: dict | None = None) -> Storage:
    cfg = cfg or deployment()
    if cfg["problems"]:
        raise RuntimeError("; ".join(cfg["problems"]))
    if cfg["storage"] == "sqlite":
        return SQLiteStorage(os.environ.get("STUDY_DB") or default_sqlite(cfg["mode"]))
    return HFDatasetStorage(os.environ.get("STUDY_DATA_DIR", f"study_data_{cfg['mode']}"), repo_id=cfg["repo"],
                            token=os.environ["HF_TOKEN"], every_minutes=float(os.environ.get("STUDY_PUSH_MINUTES", "2")))
