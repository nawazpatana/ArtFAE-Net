"""Manifest dataset + class-balanced sampler (Step 3).

Reads the train/val/test CSVs written by build_balanced_artemis.py and the cached
feature arrays written by extract_features.py (same row order as the CSV).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from . import EMOTIONS


class ManifestFeatureDataset(Dataset):
    def __init__(self, manifest_csv, feature_dir):
        self.df = pd.read_csv(manifest_csv)
        d = Path(feature_dir)
        self.g = np.load(d / "global.npy", mmap_mode="r")        # (N, 512)
        self.p = np.load(d / "patches.npy", mmap_mode="r")       # (N, 49, 768) float16
        self.a = np.load(d / "attributes.npy", mmap_mode="r")    # (N, 28) raw, not normalised
        self.t = np.load(d / "attr_text.npy", mmap_mode="r")     # (N, 512)
        assert len(self.g) == len(self.df), "features and manifest are out of sync"
        self.y_soft = self.df[[f"p_{e}" for e in EMOTIONS]].to_numpy(np.float32)
        self.y_present = self.df[[f"present_{e}" for e in EMOTIONS]].to_numpy(np.float32)
        self.a_mean = self.a_std = None

    def fit_attribute_norm(self):
        """Call on the TRAIN dataset only; copy the stats to val/test with set_attribute_norm."""
        a = np.asarray(self.a, dtype=np.float64)
        self.a_mean, self.a_std = a.mean(0), np.maximum(a.std(0), 1e-6)
        return self.a_mean, self.a_std

    def set_attribute_norm(self, mean, std):
        self.a_mean, self.a_std = mean, std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        a = np.asarray(self.a[i], dtype=np.float32)
        if self.a_mean is not None:
            a = (a - self.a_mean) / self.a_std
        return {
            "g": torch.from_numpy(np.asarray(self.g[i], dtype=np.float32)),
            "patches": torch.from_numpy(np.asarray(self.p[i], dtype=np.float32)),
            "attrs": torch.from_numpy(a.astype(np.float32)),
            "text": torch.from_numpy(np.asarray(self.t[i], dtype=np.float32)),
            "y_soft": torch.from_numpy(self.y_soft[i]),
            "y_present": torch.from_numpy(self.y_present[i]),
        }


def make_balanced_sampler(train_csv, num_samples=None, seed=42):
    """Samples artworks using the `sampler_weight` column (inverse dominant-class frequency)."""
    w = pd.read_csv(train_csv)["sampler_weight"].to_numpy(np.float64)
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples or len(w),
                                 replacement=True, generator=gen)


def train_log_prior(train_csv):
    """log of the mean training vote distribution, used by the logit-adjusted loss."""
    df = pd.read_csv(train_csv)
    prior = df[[f"p_{e}" for e in EMOTIONS]].to_numpy().mean(0)
    return torch.log(torch.as_tensor(prior, dtype=torch.float32) + 1e-8)
