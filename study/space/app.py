"""Hugging Face Space entry point (CPU only). build.py places study_app.py and its files next to this one."""
import pathlib
import runpy

runpy.run_path(str(pathlib.Path(__file__).with_name("study_app.py")), run_name="__main__")
