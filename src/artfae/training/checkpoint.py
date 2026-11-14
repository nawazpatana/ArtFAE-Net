"""Atomic, resumable research checkpoints."""
from pathlib import Path
import os
import torch
from artfae import EMOTIONS,MODEL_ID


def save(path: Path, model, optimizer, epoch: int, config: dict, normalizer: dict, **extra) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,config=config,normalizer=normalizer,emotions=list(EMOTIONS),backbone=MODEL_ID,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],**extra)
    save_state(path,state)


def save_state(path: Path,state: dict) -> None:
    """Atomically persist an already captured model/optimizer selection snapshot."""
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp'); torch.save(state,temporary); os.replace(temporary,path)


def selection_for_resume(state: dict,source: Path) -> dict | None:
    """Recover the actual selected checkpoint, including its matching optimizer state.

    New last checkpoints embed the selection. Legacy runs need their original
    sibling best.pt; equal validation scores alone cannot identify equal weights.
    """
    if state.get('best',-1)<0:return None
    selected=state.get('selection_state')
    if selected is None:
        if state.get('is_selected_checkpoint') or source.name=='best.pt':selected=state
        else:
            sibling=source.parent/'best.pt'
            if not sibling.is_file():raise ValueError('Legacy resume needs its original sibling best.pt to preserve model selection')
            selected=torch.load(sibling,map_location='cpu',weights_only=True)
    if selected['config']!=state['config'] or selected['emotions']!=state['emotions'] or selected['backbone']!=state['backbone'] or selected['best']!=state['best'] or selected['epoch']>state['epoch']:
        raise ValueError('Resume selection snapshot does not match the interrupted run')
    return {key:value for key,value in selected.items() if key!='selection_state'}


def load(path: Path, model, optimizer=None) -> dict:
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['emotions']!=list(EMOTIONS): raise ValueError('Class order mismatch')
    model.load_state_dict(state['model'])
    if optimizer is not None: optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'])
    if state.get('cuda_rng') and torch.cuda.is_initialized(): torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state
