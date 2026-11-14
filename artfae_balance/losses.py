"""Losses for minority-class recovery (Step 5).

total = logit-adjusted soft CE  +  lambda_present * presence BCE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedSoftCE(nn.Module):
    """Soft-target CE with logit adjustment (Menon et al., 2021).

    Training uses logits + tau * log_prior, which forces the raw logits to give
    rare classes (anger, fear) more mass. At inference use the RAW logits.
    tau = 0 recovers the original ArtFAE-Net loss.
    """

    def __init__(self, log_prior, tau=1.0):
        super().__init__()
        self.register_buffer("log_prior", log_prior)
        self.tau = tau

    def forward(self, logits, y_soft):
        adj = logits + self.tau * self.log_prior
        return -(y_soft * F.log_softmax(adj, dim=1)).sum(1).mean()


class PresenceBCE(nn.Module):
    """Multi-label 'did any annotator choose emotion c' loss.

    pos_weight up-weights rare positives; computed from the training present_* columns.
    """

    def __init__(self, pos_rate, max_pos_weight=20.0):
        super().__init__()
        pw = ((1 - pos_rate) / pos_rate.clamp_min(1e-4)).clamp(max=max_pos_weight)
        self.register_buffer("pos_weight", pw)

    def forward(self, presence_logits, y_present):
        return F.binary_cross_entropy_with_logits(presence_logits, y_present,
                                                  pos_weight=self.pos_weight)


class ArtFAELoss(nn.Module):
    def __init__(self, log_prior, pos_rate, tau=1.0, lambda_present=0.3):
        super().__init__()
        self.main = LogitAdjustedSoftCE(log_prior, tau)
        self.presence = PresenceBCE(pos_rate)
        self.lambda_present = lambda_present

    def forward(self, logits, presence_logits, y_soft, y_present):
        l_main = self.main(logits, y_soft)
        l_pres = self.presence(presence_logits, y_present)
        return l_main + self.lambda_present * l_pres, {"main": l_main.item(), "presence": l_pres.item()}
