"""Strict artwork manifests; utterances never enter model inputs."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from artfae import EMOTIONS


def read_manifest(path: Path, data_dir: Path) -> list[dict]:
    """Read observed manifest schema, rejecting invalid distributions and IDs."""
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'painting', 'filename', 'n_annotations'} | {f'p_{e}' for e in EMOTIONS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {required-set(reader.fieldnames or [])}')
        rows = list(reader)
    seen = set()
    for r in rows:
        if r['painting'] in seen: raise ValueError('Duplicate artwork ID')
        seen.add(r['painting'])
        filename = Path(r['filename']).name
        if Path(filename).stem != r['painting']: raise ValueError('Filename/ID mismatch')
        r['path'] = data_dir / 'images' / filename
        if not r['path'].is_file(): raise FileNotFoundError(r['path'])
        r['target'] = np.array([float(r[f'p_{e}']) for e in EMOTIONS], dtype=np.float32)
        validate_targets(r['target'][None])
        if int(r['n_annotations']) <= 0: raise ValueError('Missing annotations')
    if not rows: raise ValueError('Empty manifest')
    return rows


def validate_targets(targets: np.ndarray) -> None:
    if targets.ndim != 2 or targets.shape[1] != 9 or not np.isfinite(targets).all() or (targets < 0).any() or not np.allclose(targets.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid nine-class probability distributions')


def validate_disjoint(*splits: list[dict]) -> None:
    seen = set()
    for split in splits:
        ids = [r['painting'] for r in split]
        if len(ids) != len(set(ids)) or seen.intersection(ids): raise ValueError('Artwork overlap')
        seen.update(ids)


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.draft('RGB', (512, 512))
        if image.width * image.height > 40_000_000: raise ValueError(f'Image too large: {path}')
        return ImageOps.exif_transpose(image).convert('RGB')
