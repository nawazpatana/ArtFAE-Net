"""Small machine-readable software and source provenance records."""
from pathlib import Path
import hashlib
import importlib.metadata
import platform
import sys
import torch


def capture(device: str='cpu') -> dict:
    root=Path(__file__).resolve().parents[3]
    packages={}
    for name in ('torch','torchvision','transformers','numpy','pillow','scikit-learn','PyYAML','matplotlib'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'device':device,'torch_threads':torch.get_num_threads(),'packages':packages,'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src'/'artfae').rglob('*.py'))},'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name() if device=='cuda' and torch.cuda.is_initialized() else None}
