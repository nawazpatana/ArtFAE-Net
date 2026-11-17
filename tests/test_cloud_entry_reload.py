"""The cloud entry point reloads the app's own modules after their files change (a git pull on Streamlit Cloud
updates files in a running process). Simulated in a copy of the app: run once, add a key to i18n.py and use it in
streamlit_app.py, run again in the same process. Through streamlit_cloud/streamlit_app.py the new key works; through
app/streamlit_app.py directly (control) the stale i18n module raises the KeyError seen on Cloud.

Runs without torch (gallery mode).   pytest tests/test_cloud_entry_reload.py -v
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCENARIO = r"""
import os, sys, time
from pathlib import Path
from streamlit.testing.v1 import AppTest
root, entry = Path(sys.argv[1]), sys.argv[2]
at = AppTest.from_file(str(root / entry), default_timeout=120)
at.run()
assert not at.exception, at.exception
import i18n
assert Path(i18n.__file__).resolve().parent == root / "app", i18n.__file__
time.sleep(0.05)
# the "git pull": a new key in both languages, and the page that uses it
p = root / "app" / "i18n.py"; s = p.read_text(encoding="utf-8")
base = s.index("STRINGS = {")
for lang, text in (("en", "Pulled text"), ("zh", "更新後的文字")):
    i = s.index('        "app_title"', s.index(f'    "{lang}": {{', base))
    s = s[:i] + f'        "zz_pulled_key": "{text}",\n' + s[i:]
p.write_text(s, encoding="utf-8")
a = root / "app" / "streamlit_app.py"; t = a.read_text(encoding="utf-8")
t = t.replace('{"analyse": page_analyse,', 'st.caption(tr("zz_pulled_key"))\n{"analyse": page_analyse,', 1)
a.write_text(t, encoding="utf-8")
at.run()
excs = [str(e.message) for e in at.exception]
print("EXC", excs)
print("CAPTION", any(c.value == "Pulled text" for c in at.caption))
"""


@pytest.fixture
def app_copy(tmp_path):
    for d in ("app", "streamlit_cloud", "artfae_balance"):
        shutil.copytree(ROOT / d, tmp_path / d, ignore=shutil.ignore_patterns("__pycache__", "assets"))
    (tmp_path / "app" / "assets").symlink_to(ROOT / "app" / "assets")
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def run(root, entry):
    env = {**os.environ, "APP_MODE": "gallery"}
    out = subprocess.run([sys.executable, "-c", SCENARIO, str(root), entry], cwd=root, env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = dict(l.split(" ", 1) for l in out.stdout.splitlines() if l.startswith(("EXC", "CAPTION")))
    return lines, out.stderr


def test_cloud_entry_reloads_changed_app_modules(app_copy):
    lines, err = run(app_copy, "streamlit_cloud/streamlit_app.py")
    assert lines["EXC"] == "[]" and lines["CAPTION"] == "True", (lines, err[-2000:])
    assert "reloaded app modules after a file change (i18n)" in err


def test_control_without_the_entry_point_shows_the_stale_module_keyerror(app_copy):
    lines, _ = run(app_copy, "app/streamlit_app.py")
    assert "zz_pulled_key" in lines["EXC"] and lines["CAPTION"] == "False", lines
