"""Distributional classification objective with explicit validation."""
import torch


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or not torch.isfinite(targets).all() or (targets<0).any() or not torch.allclose(targets.sum(-1),torch.ones_like(targets[:,0]),atol=1e-5): raise ValueError('Invalid soft targets')
    return -(targets*logits.log_softmax(-1)).sum(-1).mean()
