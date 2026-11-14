"""Frozen stimulus set: write and verify study/stimuli_FROZEN.txt.

The manifest records the SHA-256 of every file the study shows or derives its assignment from:
  stimuli.csv, the 63 images (stimuli/images/*.jpg), the precomputed outputs (precomputed/outputs.json,
  precomputed/maps.npz, precomputed/attention/*.jpg), and the 3 assignment sets (assignment.make_sets on
  stimuli.csv, stored as lists and as the SHA-256 of their canonical JSON).
study_app.py calls verify() before showing anything and refuses to start on any difference, including a missing
or an extra file.

freeze (once, when the set is final):  python study/frozen.py --freeze
check:                                 python study/frozen.py
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

STUDY = Path(__file__).resolve().parent
MANIFEST = "stimuli_FROZEN.txt"
FORMAT = "artfae-study-frozen/1"
GLOBS = ["stimuli/images/*.jpg", "precomputed/attention/*.jpg"]
FILES = ["stimuli.csv", "precomputed/outputs.json", "precomputed/maps.npz"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tracked_files(study: Path) -> list:
    out = list(FILES)
    for g in GLOBS:
        out += sorted(p.relative_to(study).as_posix() for p in study.glob(g))
    return out


def assignment_sets(study: Path):
    sys.path.insert(0, str(study))
    import assignment
    with open(study / "stimuli.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == "stimulus"]
    sets = assignment.make_sets(rows)
    return sets, hashlib.sha256(json.dumps(sets, separators=(",", ":")).encode()).hexdigest()


def build(study: Path) -> dict:
    sets, sets_sha = assignment_sets(study)
    return {"format": FORMAT, "frozen_on": date.today().isoformat(),
            "files": {p: sha256(study / p) for p in tracked_files(study)},
            "assignment_sets_sha256": sets_sha, "assignment_sets": sets}


def verify(study: Path = STUDY) -> list:
    """-> list of problems (empty = the files match the frozen manifest exactly)."""
    path = study / MANIFEST
    if not path.exists():
        return [f"{MANIFEST} is missing"]
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"{MANIFEST} is not valid JSON: {e}"]
    if m.get("format") != FORMAT:
        return [f"{MANIFEST}: unknown format {m.get('format')!r}"]
    problems = []
    for rel, want in m["files"].items():
        p = study / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif sha256(p) != want:
            problems.append(f"changed: {rel}")
    extra = set(tracked_files(study)) - set(m["files"])
    problems += [f"not in the frozen set: {rel}" for rel in sorted(extra)]
    if not problems:  # sets are derived from stimuli.csv; recompute only when every file matches
        sets, sets_sha = assignment_sets(study)
        if sets_sha != m["assignment_sets_sha256"] or sets != m["assignment_sets"]:
            problems.append("assignment sets differ from the frozen sets")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="write the manifest (overwrites an existing one)")
    a = ap.parse_args()
    if a.freeze:
        m = build(STUDY)
        (STUDY / MANIFEST).write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
        n_img = sum(1 for k in m["files"] if k.startswith("stimuli/images/"))
        print(f"frozen {len(m['files'])} files ({n_img} images) and 3 assignment sets on {m['frozen_on']}")
        return
    problems = verify(STUDY)
    print("frozen set OK" if not problems else "\n".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
