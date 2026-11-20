"""Build app/assets/aesthetic_percentiles.json: training-set quantiles of the descriptive aesthetic properties
(app/visual.py aesthetic_measures) over the training paintings of data/artemis_balanced/train.csv.

run (training server, the images under clean_artemis/images/):  python app/build_aesthetic_percentiles.py [--workers 14]

Every training image is loaded exactly as in the training pipeline (core.load_image = artfae_balance's load_image)
and measured with visual.aesthetic_measures. Check: edge_density, rms_contrast and luminance_range are the same
quantities as the stored attributes edge_density, luminance_std and contrast_p95_p05 in
features_l14/train/attributes.npy (row order = train.csv); they must agree for every image (float32 tolerance),
which also confirms that the images and their order are the ones the model was trained on. Read-only otherwise.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import core  # noqa: E402
import visual  # noqa: E402

TRAIN = core.ROOT / "data/artemis_balanced/train.csv"
ATTRS = core.ROOT / "features_l14/train/attributes.npy"
OUT = core.ASSETS / "aesthetic_percentiles.json"
SAME = {"edge_density": "edge_density", "rms_contrast": "luminance_std", "luminance_range": "contrast_p95_p05"}


def measure(path):
    return visual.measure_vector(visual.aesthetic_measures(core.load_image(core.ROOT / path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    paths = pd.read_csv(TRAIN).image_path.tolist()
    t0 = time.time()
    with Pool(a.workers) as pool:
        vals = np.array(pool.map(measure, paths, chunksize=64))
    attrs = np.load(ATTRS)
    assert len(attrs) == len(vals) == len(paths), (len(attrs), len(vals))
    names = list(core.NAMES)
    worst = {}
    for k, attr in SAME.items():
        d = np.abs(vals[:, visual.MEASURE_NAMES.index(k)] - attrs[:, names.index(attr)])
        worst[k] = float(d.max())
        assert d.max() < 1e-5, f"{k} differs from the training attribute {attr}: max {d.max():.2e}"
    levels = np.linspace(0, 1, 1001)
    OUT.write_text(json.dumps({
        "names": visual.MEASURE_NAMES, "levels": levels.tolist(), "n_train": int(len(vals)),
        "quantiles": np.quantile(vals, levels, axis=0).T.round(7).tolist(),
        "source": "app/visual.py aesthetic_measures on data/artemis_balanced/train.csv images (core.load_image)",
        "check_vs_attributes_npy_max_abs_diff": worst}))
    print(f"{OUT.relative_to(core.ROOT)}: {len(vals):,} training paintings in {time.time() - t0:.0f} s; "
          f"max |diff| vs attributes.npy: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))


if __name__ == "__main__":
    main()
