"""Image-rights and form check for every image the study shows (60 stimuli, the practice item, 2 attention checks).

An image is kept only if ALL rules hold:
  R1 death     the artist died before 1956 (public domain under life + 70 years from 2026)
  R2 created   the work was created before 1931 (safe under the US 95-year rule as well; the Space is hosted in
               the US). Creation year = the LAST year of the WikiArt date (e.g. 1912-1914 -> 1914), else the year in
               the WikiArt filename; no year -> fail
  R3 2D        a painting, drawing or print: not a sculpture, relief, architecture, ceramic or other 3D object
               (WikiArt genre/media; plus MANUAL_3D, from looking at every image)
  EXCLUDE      items removed by the owner's decision, with the reason
Sources: the WikiArt artist record (https://www.wikiart.org/en/<artist-slug>) for death years, with the artist's
Wikidata "date of death" (P570) only when WikiArt has none; the WikiArt painting page
(https://www.wikiart.org/en/<artist-slug>/<painting-slug>) for genre, media and date. All responses are cached
(artist_meta.json, artwork_meta.json), so the check is reproducible offline.

A failing item is replaced in place (same stimulus ID and stratum) by the next work in the stratum's selection
order (select_stimuli.stratum_pool) that passes every rule, has an image of at least 300 px, and whose artist is
not used by any other item. Every replacement is appended to rights_replacements.csv with its reason.
Afterwards: look at the replacements, then rerun precompute.py and the tests.

run:  python study/artist_rights.py [--dry-run] [--offline]
"""
import argparse
import html as html_lib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
import select_stimuli as sel  # noqa: E402

DEATH_BEFORE = 1956
CREATED_BEFORE = 1931
ARTIST_CACHE = STUDY / "artist_meta.json"
WORK_CACHE = STUDY / "artwork_meta.json"
UA = {"User-Agent": "Mozilla/5.0 (ArtFAE-Net study rights check; research)"}
OFFLINE = False

GENRES_3D = {"sculpture", "installation", "architecture", "design", "utensil", "jewelry", "furniture", "object",
             "assemblage", "ceramics"}
MEDIA_3D = re.compile(r"\b(marble|bronze|terracotta|terra cotta|ceramic|porcelain|plaster|stone|limestone|sandstone|"
                      r"granite|alabaster|wood carving|carved|relief|cast|clay|stoneware|earthenware|maiolica|majolica|"
                      r"glazed|metal|steel|iron)\b", re.I)
# found by looking at every image (metadata alone misses some): artwork_id -> what it shows
MANUAL_3D: dict = {}
# owner's decisions: artwork_id -> reason
EXCLUDE = {
    "Realism/adolf-hitler_the-old-building-in-stand-of-trees-1909":
        "excluded by owner decision (2026-11-12): watercolour by Adolf Hitler",
}


# ---------------------------------------------------------------- fetching with caches
def _get(url, as_json=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        body = r.read().decode("utf-8", errors="replace")
    return json.loads(body) if as_json else body


def slug_variants(slug: str):
    """The holdout slug, then a repaired one: some ArtEmis slugs are UTF-8 read as Latin-1 and lower-cased
    ('joaquã\\xadn-sorolla'), while WikiArt slugs are plain ASCII ('joaquin-sorolla')."""
    out, fixed = [slug], slug
    for cand in (slug, slug.replace("ã", "Ã").replace("â", "Â")):
        try:
            fixed = cand.encode("latin-1").decode("utf-8")
            break
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    ascii_ = unicodedata.normalize("NFKD", fixed).encode("ascii", "ignore").decode()
    return out + [v for v in (fixed, ascii_) if v not in out]


class Cache:
    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def save(self):
        self.path.write_text(json.dumps(dict(sorted(self.data.items())), indent=1, ensure_ascii=False))


def _year(s):
    m = re.findall(r"(\d{3,4})", str(s or ""))
    return int(m[-1]) if m else None


def _ms_year(s):
    m = re.search(r"/Date\((-?\d+)\)/", s or "")
    return datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc).year if m else None


def artist_record(cache: Cache, slug: str) -> dict:
    if slug in cache.data:
        return cache.data[slug]
    if OFFLINE:
        raise RuntimeError(f"artist {slug} not cached (offline)")
    rec = {"slug": slug, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    w = None
    for v in slug_variants(slug):
        try:
            w = _get(f"https://www.wikiart.org/en/{urllib.parse.quote(v)}?json=2", as_json=True)
            rec.pop("wikiart_error", None)
            break
        except Exception as e:  # noqa: BLE001
            rec["wikiart_error"] = f"{type(e).__name__}: {e}"
            time.sleep(1.0)
    if w is not None:
        rec |= {"wikiart_slug": v, "wikiart_url": f"https://www.wikiart.org/en/{v}", "artistName": w.get("artistName"),
                "birthDayAsString": w.get("birthDayAsString"), "deathDayAsString": w.get("deathDayAsString"),
                "birthDay": w.get("birthDay"), "deathDay": w.get("deathDay"),
                "wikipediaUrl": w.get("wikipediaUrl") or w.get("story")}
    time.sleep(1.0)
    if not (rec.get("deathDayAsString") or rec.get("deathDay")) and rec.get("wikipediaUrl"):
        rec |= _wikidata(rec["wikipediaUrl"])
        time.sleep(1.0)
    cache.data[slug] = rec
    cache.save()
    return rec


def _wikidata(wiki_url):
    try:
        title = urllib.parse.unquote(wiki_url.rstrip("/").rsplit("/", 1)[1])
        q = urllib.parse.urlencode({"action": "wbgetentities", "sites": "enwiki", "titles": title,
                                    "props": "claims", "format": "json"})
        qid, ent = next(iter(_get(f"https://www.wikidata.org/w/api.php?{q}", as_json=True)["entities"].items()))
        if qid.startswith("-"):
            return {"wikidata_error": "no entity"}
        out = {"wikidata_id": qid}
        for prop, key in (("P570", "wikidata_death"), ("P569", "wikidata_birth")):
            c = ent.get("claims", {}).get(prop)
            if c:
                out[key] = c[0]["mainsnak"].get("datavalue", {}).get("value", {}).get("time")
        return out
    except Exception as e:  # noqa: BLE001
        return {"wikidata_error": f"{type(e).__name__}: {e}"}


def work_record(cache: Cache, artwork_id: str, artist_slug: str, wikiart_artist: str | None) -> dict:
    """Genre, media and date from the WikiArt painting page."""
    if artwork_id in cache.data:
        return cache.data[artwork_id]
    if OFFLINE:
        raise RuntimeError(f"work {artwork_id} not cached (offline)")
    painting = artwork_id.split("/", 1)[1]
    pslug = painting[len(artist_slug) + 1:] if painting.startswith(artist_slug + "_") else painting.split("_", 1)[-1]
    rec = {"artwork_id": artwork_id, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for a in dict.fromkeys(x for x in [wikiart_artist, *slug_variants(artist_slug)] if x):
        for p in dict.fromkeys(slug_variants(pslug)):
            url = f"https://www.wikiart.org/en/{urllib.parse.quote(a)}/{urllib.parse.quote(p)}"
            try:
                t = _get(url)
            except urllib.error.HTTPError as e:
                rec["page_error"] = f"HTTP {e.code} {url}"
                time.sleep(1.0)
                continue
            fields = {}
            for lab in ("Genre", "Media", "Style", "Date"):
                m = re.search(rf"<s>\s*{lab}:\s*</s>(.*?)</li>", t, re.S)
                if m:
                    fields[lab.lower()] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_lib.unescape(m.group(1)))).strip(" ;")
            rec.pop("page_error", None)
            rec |= {"url": url, **fields}
            time.sleep(1.0)
            cache.data[artwork_id] = rec
            cache.save()
            return rec
    cache.data[artwork_id] = rec
    cache.save()
    return rec


# ---------------------------------------------------------------- rules
def death_info(rec):
    death = _year(rec.get("deathDayAsString")) or _ms_year(rec.get("deathDay"))
    birth = _year(rec.get("birthDayAsString")) or _ms_year(rec.get("birthDay"))
    if death is not None:
        return birth, death, f"WikiArt {rec.get('wikiart_url')} (died {rec.get('deathDayAsString') or death})"
    if rec.get("wikidata_death"):
        b = _year(rec["wikidata_birth"][:5]) if rec.get("wikidata_birth") else None
        return birth or b, _year(rec["wikidata_death"][:5]), \
            f"Wikidata https://www.wikidata.org/wiki/{rec['wikidata_id']} P570 {rec['wikidata_death']} (no death date on WikiArt)"
    return birth, None, f"no death date found (WikiArt {rec.get('wikiart_url') or 'unavailable'})"


def created_info(work, slug_year):
    """Latest year in the WikiArt date ('1912 - 1914' -> 1914; place names after ';' ignored), else the slug year."""
    date = (work.get("date") or "").split(";")[0]
    ys = [int(y) for y in re.findall(r"\b(1\d{3}|20\d{2})\b", date)]
    if ys:
        return max(ys), f"WikiArt date '{date.strip()}' {work.get('url', '')}".strip()
    if slug_year is not None and not pd.isna(slug_year):
        return int(slug_year), "year in the WikiArt filename"
    return None, "no creation year"


def form_info(work, artwork_id):
    if artwork_id in MANUAL_3D:
        return False, f"3D (visual check): {MANUAL_3D[artwork_id]}"
    g, m = (work.get("genre") or "").lower(), work.get("media") or ""
    if any(x.strip() in GENRES_3D for x in re.split(r"[,;]", g)):
        return False, f"3D: WikiArt genre '{work.get('genre')}'"
    if MEDIA_3D.search(m):
        return False, f"3D: WikiArt media '{m}'"
    src = f"WikiArt genre '{work.get('genre') or '?'}', media '{m or '?'}'" if work.get("url") else "no WikiArt page; visual check"
    return True, f"2D ({src})"


def evaluate(artist_cache, work_cache, artwork_id, artist_slug, slug_year):
    """-> dict with every rule's outcome and the failure reasons (None if the item passes)."""
    a = artist_record(artist_cache, artist_slug)
    w = work_record(work_cache, artwork_id, artist_slug, a.get("wikiart_slug"))
    birth, death, dsrc = death_info(a)
    created, csrc = created_info(w, slug_year)
    is_2d, fsrc = form_info(w, artwork_id)
    fails = []
    if artwork_id in EXCLUDE:
        fails.append(EXCLUDE[artwork_id])
    if death is None or death >= DEATH_BEFORE:
        fails.append(f"artist died {death if death else 'unknown'} (rule: before {DEATH_BEFORE})")
    if created is None or created >= CREATED_BEFORE:
        fails.append(f"created {created if created else 'unknown'} (rule: before {CREATED_BEFORE})")
    if not is_2d:
        fails.append(fsrc)
    return {"birth": birth, "death": death, "death_source": dsrc, "created": created, "created_source": csrc,
            "is_2d": is_2d, "form_source": fsrc, "reason": "; ".join(fails) or None}


# ---------------------------------------------------------------- main
def main():
    global OFFLINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    OFFLINE = args.offline
    ac, wc = Cache(ARTIST_CACHE), Cache(WORK_CACHE)
    stim = pd.read_csv(STUDY / "stimuli.csv", keep_default_na=False, dtype=str)
    votes = pd.read_csv(STUDY / "private/votes.csv", dtype={"stimulus_id": str})
    pool = sel.load_pool().set_index("artwork_id", drop=False)
    flat = pool.reset_index(drop=True)
    if "replaced_from" not in stim:
        stim["replaced_from"] = ""
    slug_of = lambda aid: pool.loc[aid, "artist"]  # noqa: E731
    syear = lambda aid: pool.loc[aid, "year"]  # noqa: E731
    used_artists, used_art = {slug_of(a) for a in stim.artwork_id}, set(stim.artwork_id)
    log_path = STUDY / "rights_replacements.csv"
    log = pd.read_csv(log_path, dtype=str, keep_default_na=False) if log_path.exists() else pd.DataFrame()
    if len(log) and "reason" not in log:  # round 1 (2026-11-12): death-year rule only
        log["reason"] = log.old_death_year.map(lambda y: f"artist died {y or 'unknown'} (rule: before {DEATH_BEFORE})")
        log["round"] = "1"
    new_rows = []
    for i, r in stim.iterrows():
        ev = evaluate(ac, wc, r.artwork_id, slug_of(r.artwork_id), syear(r.artwork_id))
        if ev["reason"] is None:
            continue
        print(f"FAIL {r.stimulus_id} [{r.stratum}] {r.artwork_id}: {ev['reason']}")
        if args.dry_run:
            continue
        new = None
        for _, c in sel.stratum_pool(flat, r.stratum).iterrows():
            if c.artwork_id in used_art or c.artist in used_artists or not sel.image_ok(c.image_path):
                continue
            if evaluate(ac, wc, c.artwork_id, c.artist, c.year)["reason"] is None:
                new = c
                break
        if new is None:
            sys.exit(f"{r.stimulus_id}: no replacement in stratum {r.stratum} passes every rule")
        used_art.add(new.artwork_id); used_artists.add(new.artist)
        row = new.to_frame().T.assign(stimulus_id=r.stimulus_id, role=r.role, stratum=r.stratum)
        row["year"] = row["year"].astype(object)
        for k, v in sel.public_rows(row.reset_index(drop=True)).astype(str).iloc[0].items():
            stim.at[i, k] = v
        stim.at[i, "replaced_from"] = (r.replaced_from + ";" if r.replaced_from else "") + r.artwork_id
        votes = pd.concat([votes[votes.stimulus_id != r.stimulus_id],
                           sel.vote_rows(row.reset_index(drop=True).infer_objects())], ignore_index=True)
        sel.save_image(r.stimulus_id, new.image_path, STUDY / "stimuli/images")
        new_rows.append({"stimulus_id": r.stimulus_id, "stratum": r.stratum, "old_artwork_id": r.artwork_id,
                         "old_artist": slug_of(r.artwork_id), "old_death_year": str(ev["death"] or ""),
                         "old_source": ev["death_source"], "new_artwork_id": new.artwork_id, "new_artist": new.artist,
                         "reason": ev["reason"], "round": "2"})
        print(f"   -> {new.artwork_id}")
    if args.dry_run:
        return
    for i, r in stim.iterrows():  # record every rule's outcome for every final item
        ev = evaluate(ac, wc, r.artwork_id, slug_of(r.artwork_id), syear(r.artwork_id))
        assert ev["reason"] is None, (r.stimulus_id, ev["reason"])
        stim.at[i, "artist_slug"] = slug_of(r.artwork_id)
        stim.at[i, "artist_birth_year"] = "" if ev["birth"] is None else str(ev["birth"])
        stim.at[i, "artist_death_year"], stim.at[i, "death_source"] = str(ev["death"]), ev["death_source"]
        stim.at[i, "created_year"], stim.at[i, "created_source"] = str(ev["created"]), ev["created_source"]
        stim.at[i, "is_2d"], stim.at[i, "form_source"] = "True", ev["form_source"]
        stim.at[i, "rights_ok"] = "True"
    order = {s: k for k, s in enumerate(stim.stimulus_id)}
    stim.to_csv(STUDY / "stimuli.csv", index=False)
    votes.sort_values("stimulus_id", key=lambda s: s.map(order)).to_csv(STUDY / "private/votes.csv", index=False)
    pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)
    print(f"{len(stim)} items pass all rules; {len(new_rows)} replaced in this round")


if __name__ == "__main__":
    main()
