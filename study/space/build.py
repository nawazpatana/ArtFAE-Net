"""Assemble the Hugging Face Space upload folder (study/space/dist/). Copies only what the study needs at run time.
Never copies study/private/ (ArtEmis vote shares, pilot data). Does not upload or deploy anything.

run:  python study/space/build.py
"""
import shutil
from pathlib import Path

SPACE = Path(__file__).resolve().parent
STUDY = SPACE.parent
ROOT = STUDY.parent
DIST = SPACE / "dist"

FILES = ["study_app.py", "common.py", "assignment.py", "storage.py", "study_i18n.py", "export.py", "frozen.py",
         "stimuli_FROZEN.txt", "precomputed/maps.npz",
         "consent_en.md", "consent_zh.md", "stimuli.csv", "precomputed/outputs.json"]


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    for f in ["app.py", "requirements.txt", "README.md", "Dockerfile"]:
        shutil.copy2(SPACE / f, DIST / f)
    for f in FILES:
        (DIST / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STUDY / f, DIST / f)
    shutil.copytree(STUDY / "stimuli/images", DIST / "stimuli/images")
    shutil.copytree(STUDY / "precomputed/attention", DIST / "precomputed/attention")
    (DIST / "vendor").mkdir()
    for f in ["ui.py", "i18n.py"]:  # the demo's look and emotion names, unchanged
        shutil.copy2(ROOT / "app" / f, DIST / "vendor" / f)
    (DIST / ".streamlit").mkdir()
    shutil.copy2(ROOT / "app/.streamlit/config.toml", DIST / ".streamlit/config.toml")
    assert not list(DIST.rglob("votes*")) and not (DIST / "private").exists()
    n = sum(1 for p in DIST.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"{DIST}: {n} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
