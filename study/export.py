"""Export the study data to CSV (long format): trials.csv (one row per trial) and participants.csv.

usage:  python study/export.py OUTDIR            (backend from STUDY_STORAGE / STUDY_DB / STUDY_DATA_DIR)
        python study/export.py OUTDIR --sqlite path/to/file.sqlite
        python study/export.py OUTDIR --jsonl path/to/local/dataset/folder   (a downloaded copy of the HF dataset)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import storage as store  # noqa: E402


def export(backend, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(backend.trials(), columns=store.TRIAL_FIELDS).sort_values(["participant_id", "trial_index"])
    p = pd.DataFrame(backend.participants(), columns=store.PARTICIPANT_FIELDS).sort_values("arrival_index")
    t.to_csv(out / "trials.csv", index=False)
    p.to_csv(out / "participants.csv", index=False)
    return t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sqlite")
    ap.add_argument("--jsonl")
    a = ap.parse_args()
    if a.sqlite:
        b = store.SQLiteStorage(a.sqlite)
    elif a.jsonl:
        b = store.HFDatasetStorage(a.jsonl, scheduler=None)
    else:
        b = store.from_env()
    t, p = export(b, Path(a.out))
    print(f"{len(p)} participants, {len(t)} trial rows -> {a.out}")


if __name__ == "__main__":
    main()
