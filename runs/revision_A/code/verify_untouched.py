"""Check that every runs/final_eval* artifact hashed in final_selection_v2.txt still matches, and that no file
under runs/final_eval* was modified after the last final evaluation (2026-11-02 22:45 UTC)."""
import hashlib
import json
from datetime import datetime, timezone

from common import ROOT

sel = json.loads((ROOT / "runs/final_selection_v2.txt").read_text())
bad = [f for f, h in sel["artifact_sha256"].items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
cut = datetime(2026, 11, 2, 22, 46, tzinfo=timezone.utc).timestamp()
files = [p for d in ("final_eval", "final_eval_v2") for p in (ROOT / "runs" / d).rglob("*") if p.is_file()]
late = [str(p) for p in files if p.stat().st_mtime > cut]
print(f"{len(sel['artifact_sha256'])} hashed artifacts, mismatches: {bad or 'none'}")
print(f"{len(files)} files under runs/final_eval*, modified after 2026-11-02 22:46 UTC: {late or 'none'}")
assert not bad and not late
