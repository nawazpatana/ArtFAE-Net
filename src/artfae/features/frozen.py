"""Bounded image decoding for reusable frozen visual feature extraction."""
from collections.abc import Callable
import torch
from torch.utils.data import Dataset,DataLoader
from artfae.datasets.manifests import open_image

class FeatureRows(Dataset):
    """Small tensor feature corpus with artwork identity preserved."""
    def __init__(self,state: dict):self.state=state
    def __len__(self) -> int:return len(self.state['ids'])
    def __getitem__(self,index: int):return self.state['features'][index],self.state['targets'][index],self.state['ids'][index]


def extract_frozen(rows: list[dict],transform: Callable,encoder: torch.nn.Module,device: torch.device,batch_size: int=32,guard: Callable | None=None) -> dict:
    """Decode only one batch; never retain the original artwork pixel corpus."""
    class Images(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,index):
            row=rows[index];return transform(open_image(row['path'])),torch.as_tensor(row['target']),row['painting']
    features=[];targets=[];ids=[];encoder.eval()
    with torch.no_grad():
        for batch,(pixels,y,names) in enumerate(DataLoader(Images(),batch_size=batch_size)):
            if guard is not None:guard()
            encoded=encoder(pixels.to(device)).flatten(1).cpu()
            if not torch.isfinite(encoded).all():raise FloatingPointError('Nonfinite frozen image features')
            features.append(encoded);targets.append(y);ids.extend(names)
            if batch%25==0:print(f'Frozen visual features: {len(ids)}/{len(rows)}',flush=True)
    return {'features':torch.cat(features),'targets':torch.cat(targets),'ids':ids}
