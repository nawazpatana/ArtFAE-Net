"""Streamlit Community Cloud entry point: the demo in cloud_live mode.

The 12 gallery examples show precomputed results (app/assets/precomputed/gallery.json); uploads run live on the
CPU with CLIP ViT-L/14@336's vision tower only (fp32) and precomputed text embeddings, within the ~2.7 GB limit.

Streamlit Cloud settings: main file path streamlit_cloud/streamlit_app.py, Python 3.12. Cloud installs
streamlit_cloud/requirements.txt (the dependency file next to the entry point), not app/requirements.txt.
Runs app/streamlit_app.py unchanged with APP_MODE=cloud_live (an APP_MODE set in the environment still wins,
e.g. APP_MODE=gallery for the torch-free version).

Fresh modules after a git pull: Streamlit Cloud updates the files of a running app without restarting the
Python process, and Streamlit only reloads modules next to the main script (this folder). The app's own modules
in app/ would stay the old versions in sys.modules while app/streamlit_app.py (run anew every time) is already
new, e.g. a new page asking an old i18n for a new key (KeyError). So before every run, if any of those modules'
files changed since they were imported, all of them are dropped together (they import each other) and Streamlit's
caches are cleared (cached results and the engine may come from the old code); the app then imports the new set.
"""
import os
import runpy
import sys
import types
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
OWN_MODULES = ("core", "i18n", "ui", "gallery", "inference")  # app/*.py modules the app imports by name
_STATE = "_artfae_module_stamps"  # survives Streamlit's reruns: {module name: (mtime_ns, size) at import}


def _stamp(path: Path):
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def _own(name):
    mod = sys.modules.get(name)
    f = getattr(mod, "__file__", None)
    return mod if f and Path(f).resolve().parent == APP_DIR else None


def drop_stale_modules() -> list:
    """Remove the app's own modules from sys.modules if any of their files changed; returns the changed names."""
    stamps = sys.modules.setdefault(_STATE, types.ModuleType(_STATE)).__dict__.setdefault("stamps", {})
    changed = []
    for name in OWN_MODULES:
        mod = _own(name)
        if mod is None:
            continue
        try:
            now = _stamp(Path(mod.__file__))
        except OSError:  # the file was removed by the update
            now = None
        if name in stamps and stamps[name] != now:
            changed.append(name)
    if changed:
        for name in OWN_MODULES:
            if _own(name) is not None:
                del sys.modules[name]
        stamps.clear()
        try:
            import streamlit as st
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:  # noqa: BLE001  (never block the app on cache clearing)
            pass
        print(f"artfae cloud entry: reloaded app modules after a file change ({', '.join(changed)})",
              file=sys.stderr, flush=True)
    return changed


def record_stamps():
    stamps = sys.modules.setdefault(_STATE, types.ModuleType(_STATE)).__dict__.setdefault("stamps", {})
    for name in OWN_MODULES:
        mod = _own(name)
        if mod is not None and name not in stamps:
            try:
                stamps[name] = _stamp(Path(mod.__file__))
            except OSError:
                pass


os.environ.setdefault("APP_MODE", "cloud_live")
drop_stale_modules()
try:
    runpy.run_path(str(APP_DIR / "streamlit_app.py"), run_name="__main__")
finally:  # also when Streamlit stops or reruns the script with an exception
    record_stamps()
