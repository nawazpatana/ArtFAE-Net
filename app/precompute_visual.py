"""Torch-free precomputation for the gallery examples (Prof. Lu's second feedback round):

  1. app/assets/precomputed/gallery.json: adds "palette" (visual.palette) and "aesthetic"
     (visual.aesthetic_measures) to each of the 12 examples and sets the format to artfae-gallery/2. Every field
     already in the file (seed logits, attention maps, presence, attributes, description, ...) is kept exactly:
     the script refuses to write if any of them would change. The model outputs are not touched.
  2. app/assets/examples/metadata.json: title, artist, date, style and genre of each example for the iconographic
     level, from the cached WikiArt records in app/assets/examples/rights_works.json (app/check_gallery_rights.py).

run:  python app/precompute_visual.py          (no torch, no weights; numpy + PIL)
      python app/precompute_visual.py --check  (recompute and compare with the stored values; writes nothing)
"""
import argparse
import json
import re
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import core  # noqa: E402
import ui  # noqa: E402
import visual  # noqa: E402

GALLERY = core.ASSETS / "precomputed/gallery.json"
META = core.ASSETS / "examples/metadata.json"
WORKS = core.ASSETS / "examples/rights_works.json"
FORMAT = "artfae-gallery/2"


def visual_record(img) -> dict:
    m = visual.aesthetic_measures(img)
    return {"palette": [{k: p[k] for k in ("hex", "rgb", "share", "percent")} for p in visual.palette(img)],
            "aesthetic": {"values": m["values"], "sides": m["sides"]}}


def clean_date(text) -> str | None:
    """'c.1490 - c. 1512 ; Florence , Italy' -> 'c. 1490 – c. 1512' (place dropped, spacing normalised)."""
    if not text:
        return None
    d = text.split(";")[0].strip()
    d = re.sub(r"c\.\s*", "c. ", d)
    d = re.sub(r"\s*-\s*", " – ", d)
    return d or None


def metadata(examples) -> dict:
    works = json.loads(WORKS.read_text(encoding="utf-8"))
    out = {}
    for ex in examples:
        w = works[ex["artwork_id"]]
        assert "page_error" not in w, ex["artwork_id"]
        artist, title, slug_year = core.example_parts(ex, "")  # with the curated display titles
        out[ex["file"]] = {"artwork_id": ex["artwork_id"], "title": title, "artist": artist,
                           "date": clean_date(w.get("date")) or slug_year or None, "style": w.get("style"),
                           "genre": w.get("genre"), "source": w.get("url"), "fetched_at": w.get("fetched_at")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    examples = core.load_examples()
    doc = json.loads(GALLERY.read_text(encoding="utf-8"))
    before = json.dumps({f: {k: v for k, v in e.items() if k not in ("palette", "aesthetic")}
                         for f, e in doc["examples"].items()}, sort_keys=True)
    new = {ex["file"]: visual_record(core.load_upload(Path(ex["path"]).read_bytes())) for ex in examples}
    meta = metadata(examples)
    if a.check:
        bad = [f for f, v in new.items() if {k: doc["examples"][f].get(k) for k in v} != v]
        stored_meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else None
        print("gallery visual fields:", "OK" if not bad else f"DIFFER {bad}")
        print("metadata:", "OK" if stored_meta == meta else "DIFFERS")
        sys.exit(1 if bad or stored_meta != meta else 0)
    for f, v in new.items():
        doc["examples"][f].update(v)
    doc["format"] = FORMAT
    after = json.dumps({f: {k: v for k, v in e.items() if k not in ("palette", "aesthetic")}
                        for f, e in doc["examples"].items()}, sort_keys=True)
    assert after == before, "an existing field would change"
    GALLERY.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    META.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{GALLERY.relative_to(core.ROOT)}: palette + aesthetic for {len(new)} examples ({FORMAT}); "
          f"{META.relative_to(core.ROOT)}: {len(meta)} entries")


if __name__ == "__main__":
    main()
