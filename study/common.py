"""Shared constants and small helpers for the study (no Streamlit, no torch)."""
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
# ArtEmis label order, identical to artfae_balance.EMOTIONS (checked in tests/test_study.py)
EMOTIONS = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness", "something_else"]
SENTIMENT = {"amusement": "positive", "awe": "positive", "contentment": "positive", "excitement": "positive",
             "anger": "negative", "disgust": "negative", "fear": "negative", "sadness": "negative",
             "something_else": "other"}
N_TRIALS = 20
N_SETS = 3  # 60 stimuli / 20 per participant


def people_counts(probs, n: int = 10) -> np.ndarray:
    """Largest-remainder (Hamilton) rounding of n * probs; ties broken by the larger probability (as in app/ui.py)."""
    p = np.asarray(probs, np.float64)
    q = n * p / p.sum()
    c = np.floor(q).astype(int)
    rem = q - c
    for i in sorted(range(len(p)), key=lambda i: (-rem[i], -p[i]))[: n - int(c.sum())]:
        c[i] += 1
    return c
