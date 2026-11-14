"""Attribute-guided fusion and class-conditioned spatial evidence learning."""
import torch
from torch import nn
import torch.nn.functional as F


class AGF(nn.Module):
    def __init__(self, visual_dim: int, attribute_dim: int, text_dim: int, hidden: int = 128, gated: bool = True):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(d,hidden) for d in (visual_dim,attribute_dim,text_dim)])
        self.gate = nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.Linear(hidden,3*hidden),nn.Sigmoid()) if gated else None
        self.fusion = nn.Sequential(nn.Linear(3*hidden,hidden),nn.LayerNorm(hidden),nn.GELU())

    def forward(self, g: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u = torch.cat([p(x) for p,x in zip(self.projections,(g,a,t))],-1)
        return self.fusion(u * self.gate(u) if self.gate is not None else u)


class ECSA(nn.Module):
    def __init__(self, patch_dim: int, hidden: int, classes: int = 9):
        super().__init__(); self.classes=classes; self.hidden=hidden
        self.query=nn.Linear(hidden,classes*hidden); self.key=nn.Linear(patch_dim,hidden); self.value=nn.Linear(patch_dim,hidden)
        self.region_weights=nn.Parameter(torch.randn(classes,hidden)*.02)

    def forward(self, fused: torch.Tensor, patches: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        q=self.query(fused).reshape(-1,self.classes,self.hidden)
        maps=torch.softmax(torch.einsum('bcd,bpd->bcp',q,self.key(patches))/self.hidden**.5,-1)
        regions=torch.einsum('bcp,bpd->bcd',maps,self.value(patches))
        return (regions*self.region_weights).sum(-1),maps


class ArtFAENet(nn.Module):
    def __init__(self, visual_dim: int, attribute_dim: int, text_dim: int, patch_dim: int, hidden: int=128, gated: bool=True, spatial: bool=True, numerical: bool=True, text: bool=True, attribute_indices: list[int] | None=None):
        super().__init__(); self.numerical=numerical; self.text=text
        mask=torch.ones(attribute_dim) if attribute_indices is None else torch.zeros(attribute_dim)
        if attribute_indices is not None:
            if any(i<0 or i>=attribute_dim for i in attribute_indices): raise ValueError("Invalid attribute index")
            mask[attribute_indices]=1
        self.register_buffer("attribute_mask",mask,persistent=False)
        self.agf=AGF(visual_dim,attribute_dim,text_dim,hidden,gated); self.head=nn.Linear(hidden,9)
        self.ecsa=ECSA(patch_dim,hidden) if spatial else None

    def forward(self, g: torch.Tensor, a: torch.Tensor, t: torch.Tensor, patches: torch.Tensor, return_attention: bool=False):
        h=self.agf(g,a*self.attribute_mask if self.numerical else torch.zeros_like(a),t if self.text else torch.zeros_like(t))
        logits=self.head(h); maps=None
        if self.ecsa is not None:
            regional,maps=self.ecsa(h,patches); logits=logits+regional
        return (logits,maps) if return_attention else logits


class FusionBaseline(nn.Module):
    def __init__(self, visual_dim: int, attribute_dim: int, text_dim: int, variant: str):
        super().__init__(); self.variant=variant
        dims={'B2':visual_dim,'B3':visual_dim+attribute_dim,'B4':visual_dim+text_dim,'B5':visual_dim+attribute_dim+text_dim}
        d=dims[variant]; self.head=nn.Linear(d,9) if variant=='B2' else nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,9))

    def forward(self,g,a,t,patches=None):
        x={'B2':[g],'B3':[g,a],'B4':[g,t],'B5':[g,a,t]}[self.variant]
        return self.head(torch.cat(x,-1))


class ResNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50,ResNet50_Weights
        weights=ResNet50_Weights.DEFAULT; self.preprocess=weights.transforms(); self.network=resnet50(weights=weights)
        self.network.fc=nn.Linear(self.network.fc.in_features,9)

    def forward(self,pixels): return self.network(pixels)
