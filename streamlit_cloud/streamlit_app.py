"""Streamlit Community Cloud entry point: the demo in gallery mode (no torch, no weights).

Streamlit Cloud settings: main file path streamlit_cloud/streamlit_app.py, Python 3.12. Cloud installs
streamlit_cloud/requirements.txt (the dependency file next to the entry point), not app/requirements.txt.
Runs app/streamlit_app.py unchanged with APP_MODE=gallery (an APP_MODE set in the environment still wins).
"""
import os
import runpy
from pathlib import Path

os.environ.setdefault("APP_MODE", "gallery")
runpy.run_path(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), run_name="__main__")
