"""Write runs/final_selection_v3.txt from DRAFT3: fill HOLDOUT, set status final. Evaluates nothing.

Re-verifies every DRAFT3 hash, checks that the fresh-holdout features align with holdout.csv, then copies DRAFT3
unchanged except: status, finalised_at_utc, derived_from (DRAFT3 sha256), HOLDOUT (manifest, features, output_dir,
sha256 of holdout.csv, excluded.csv, duplicates/classification.csv and every holdout feature .npy), HOLDOUT_composition
(from <prep>/composition.json), HOLDOUT_known_issues, and the code_sha256 entries for the holdout-stage scripts.
Refuses if runs/final_selection_v3.txt exists or the output directory exists.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import ROOT

DRAFT = ROOT / "runs/final_selection_v3_DRAFT3.txt"
FINAL = ROOT / "runs/final_selection_v3.txt"
PREP = "runs/revision_B/fresh_holdout"
FEAT = "features_l14_fresh_holdout"
OUTD = "runs/final_eval_v3"
NEW_CODE = ["runs/revision_B/code/holdout_composition.py", "runs/revision_B/code/make_selection_v3_final.py"]


def sha(p):
    h = hashlib.sha256()
    with open(ROOT / p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if FINAL.exists():
        sys.exit(f"{FINAL} exists; refusing to overwrite")
    if (ROOT / OUTD).exists():
        sys.exit(f"{OUTD} exists; the evaluator's output directory must not exist yet")
    sel = json.loads(DRAFT.read_text())
    bad = [p for s in ("code_sha256", "data_sha256", "checkpoint_sha256") for p, h in sel[s].items() if sha(p) != h]
    if bad:
        sys.exit(f"DRAFT3 hash mismatch: {bad}")
    man = pd.read_csv(ROOT / PREP / "holdout.csv")
    npys = sorted((ROOT / FEAT).glob("*.npy"))
    if not npys:
        sys.exit(f"no features in {FEAT}")
    for f in npys:
        n = np.load(f, mmap_mode="r").shape[0]
        if n != len(man):
            sys.exit(f"{f.name}: {n} rows, manifest has {len(man)}")
    files = [f"{PREP}/holdout.csv", f"{PREP}/excluded.csv", f"{PREP}/duplicates/classification.csv"] + \
            [str(f.relative_to(ROOT)) for f in npys]
    comp = json.loads((ROOT / PREP / "composition.json").read_text())
    comp.pop("no_image_list", None)

    out = {}
    for k, v in sel.items():
        if k == "status":
            out[k] = ("FINAL — pre-registered. HOLDOUT built and frozen; nothing has been evaluated on it. "
                      "eval_holdout.py has not been run.")
            out["finalised_at_utc"] = datetime.now(timezone.utc).isoformat()
            out["derived_from"] = {"file": str(DRAFT.relative_to(ROOT)), "sha256": sha(DRAFT.relative_to(ROOT)),
                                   "changes": "status, finalised_at_utc, HOLDOUT, HOLDOUT_composition, "
                                              "HOLDOUT_known_issues, code_sha256 additions; everything else verbatim"}
        elif k == "HOLDOUT":
            out[k] = {"manifest": f"{PREP}/holdout.csv", "features": FEAT, "output_dir": OUTD,
                      "sha256": {f: sha(f) for f in files}}
            out["HOLDOUT_composition"] = comp
            out["HOLDOUT_known_issues"] = [
                "Painting keys are Path(name).stem (build_fresh_holdout.key, build_balanced_artemis.norm_key): text after "
                "the last '.' is dropped. In ArtEmis every dotted name is a dotted artist slug (a.y.-jackson, "
                "j.-e.-h.-macdonald, m.-h.-maxy, m.c.-escher, n.c.-wyeth, r.-b.-kitaj, t.-c.-steele, william-h.-johnson); "
                "their 528 artworks collapse onto 8 keys and are excluded as 'painting stem maps to >1 ArtEmis "
                "artwork_id' although no full painting name is ambiguous. These are the same 528 artworks reported "
                "as unmatched images (norm_key; the image files exist), so none was used in any earlier dataset. "
                "Conservative (excludes, cannot leak): no holdout painting contains a '.', and no holdout painting's "
                "full name occurs in any used set.",
                "Two zip entries had bad CRCs on extraction. Post_Impressionism/vincent-van-gogh_l-arlesienne-portrait-"
                "of-madame-ginoux-1890.jpg fails full decode (OSError), but it is not an ArtEmis artwork (ArtEmis has "
                "only ...-1890-1, already used), so it never reached the decode check. Baroque/rembrandt_woman-standing-"
                "with-raised-hands.jpg decodes without error despite the CRC failure; its artwork is already used. "
                "Neither file is in the holdout; the decode check excluded 0 artworks.",
            ]
        elif k == "open_items_before_finalising":
            out[k] = v
            out["open_items_status_at_finalisation"] = {
                "done": ["candidates run on the full WikiArt folder", "all 487 pHash pairs classified with a one-line "
                         "reason (uncertain = duplicate); finalize run", "holdout features extracted",
                         "HOLDOUT filled; composition recorded in HOLDOUT_composition"],
                "remaining": ["Prior test exposure statement for the manuscript",
                              "R3_MF1 remains descriptive only"]}
        elif k == "code_sha256":
            out[k] = {**v, **{f: sha(f) for f in NEW_CODE}}
        else:
            out[k] = v
    FINAL.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"written {FINAL} ({len(files)} holdout files hashed)")
    print(f"sha256 {sha(FINAL.relative_to(ROOT))}")


if __name__ == "__main__":
    main()
