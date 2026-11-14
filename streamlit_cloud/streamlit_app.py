"""Streamlit Community Cloud entry point: the demo in cloud_live mode.

The 12 gallery examples show precomputed results (app/assets/precomputed/gallery.json); uploads run live on the
CPU with CLIP ViT-L/14@336's vision tower only (fp32) and precomputed text embeddings, within the ~2.7 GB limit.

Streamlit Cloud settings: main file path streamlit_cloud/streamlit_app.py, Python 3.12. Cloud installs
streamlit_cloud/requirements.txt (the dependency file next to the entry point), not app/requirements.txt.
Runs app/streamlit_app.py unchanged with APP_MODE=cloud_live (an APP_MODE set in the environment still wins,
e.g. APP_MODE=gallery for the torch-free version).
"""
import os
import runpy
from pathlib import Path

os.environ.setdefault("APP_MODE", "cloud_live")
runpy.run_path(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), run_name="__main__")
