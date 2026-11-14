"""ArtFAE-Net M1 with the optional Step 4 add-ons wired into its actual forward pass.

With every option off, use the paper's ArtFAENet itself (see build_model): the paper path
is not re-implemented here.

  presence=True        PresenceHead on the fused representation h        (R3)
  text_emb=(9, 512)    q_c = W_c h + U e_c inside M1's ECSA, where W_c is
                       M1's own ECSA query layer (heads.TextAnchoredQueries) (R4)
  regional_gate_init   logits = head(h) + sigmoid(beta) * regional       (R4)
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artfae.models.network import ArtFAENet  # noqa: E402

from .heads import PresenceHead, RegionalGate  # noqa: E402


class ArtFAENetPlus(ArtFAENet):
    def __init__(self, *, presence=False, text_emb=None, regional_gate_init=None, **m1_kwargs):
        # All M1 parameters are created first, so under a fixed seed their initial values are
        # identical to the paper model's; add-on parameters are drawn afterwards.
        super().__init__(**m1_kwargs)
        hidden = self.head.in_features
        self.presence_head = PresenceHead(hidden, self.head.out_features) if presence else None
        if text_emb is not None:
            if self.ecsa is None:
                raise ValueError("text-anchored queries need the ECSA branch (spatial=True)")
            # TextAnchoredQueries.text_proj; its query_proj is M1's existing self.ecsa.query
            self.register_buffer("text_emb", text_emb.float())
            self.text_proj = nn.Linear(text_emb.shape[1], hidden, bias=False)
        else:
            self.text_emb = self.text_proj = None
        self.regional_gate = RegionalGate(regional_gate_init) if regional_gate_init is not None else None
        if self.regional_gate is not None and self.ecsa is None:
            raise ValueError("regional gate needs the ECSA branch (spatial=True)")

    def _ecsa(self, h, patches):
        e = self.ecsa
        if self.text_proj is None:
            return e(h, patches)
        # Same computation as network.ECSA.forward, with the text anchor added to q.
        q = e.query(h).reshape(-1, e.classes, e.hidden) + self.text_proj(self.text_emb).unsqueeze(0)
        maps = torch.softmax(torch.einsum("bcd,bpd->bcp", q, e.key(patches)) / e.hidden ** .5, -1)
        regions = torch.einsum("bcp,bpd->bcd", maps, e.value(patches))
        return (regions * e.region_weights).sum(-1), maps

    def forward(self, g, a, t, patches, return_attention=False, return_presence=False):
        h = self.agf(g, a * self.attribute_mask if self.numerical else torch.zeros_like(a),
                     t if self.text else torch.zeros_like(t))
        logits = self.head(h); maps = None
        if self.ecsa is not None:
            regional, maps = self._ecsa(h, patches)
            logits = self.regional_gate(logits, regional) if self.regional_gate is not None else logits + regional
        out = (logits,)
        if return_attention:
            out += (maps,)
        if return_presence:
            out += (self.presence_head(h) if self.presence_head is not None else None,)
        return out if len(out) > 1 else logits


def build_model(dims, architecture, presence=False, text_emb=None, regional_gate_init=None):
    """Paper ArtFAENet when no add-on is enabled, otherwise ArtFAENetPlus."""
    if not presence and text_emb is None and regional_gate_init is None:
        return ArtFAENet(**dims, **architecture)
    return ArtFAENetPlus(presence=presence, text_emb=text_emb, regional_gate_init=regional_gate_init,
                         **dims, **architecture)
