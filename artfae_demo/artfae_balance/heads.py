"""Model add-ons (Step 4).

1. PresenceHead     - auxiliary multi-label head on the fused representation h.
2. TextAnchoredQueries - adds a CLIP-text-initialised bias to each ECSA class query,
                         so 'fear' / 'anger' queries start from language, not noise.
3. RegionalGate     - learnable scalar on the regional logit term (starts near 0),
                      which protects calibration (the no-ECSA ablation had better Brier).
"""
import torch
import torch.nn as nn

from . import EMOTIONS

# Content-level prompts. Avoid colour words: the attribute branch already covers colour.
EMOTION_PROMPTS = {
    "amusement": ["a funny, playful painting", "a humorous scene with caricatures"],
    "awe": ["a majestic, sublime landscape painting", "a grand cathedral or vast sky"],
    "contentment": ["a peaceful, calm everyday scene", "a quiet pastoral painting"],
    "excitement": ["a lively scene full of motion", "a festive crowd celebrating"],
    "anger": ["a painting of violence and rage", "people fighting, a brutal battle scene"],
    "disgust": ["a grotesque, repulsive painting", "a rotting, deformed figure"],
    "fear": ["a terrifying, threatening scene", "a monster, death or a dark storm at sea"],
    "sadness": ["a mournful painting of grief", "a lonely, weeping figure"],
    "something_else": ["an abstract painting with no clear subject", "an ambiguous composition"],
}


@torch.no_grad()
def clip_emotion_text_embeddings(model_name="openai/clip-vit-base-patch32", device="cpu", revision=None):
    """Returns (9, 512) L2-normalised mean prompt embeddings, in EMOTIONS order."""
    from transformers import CLIPModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(model_name, revision=revision)
    model = CLIPModel.from_pretrained(model_name, revision=revision).to(device).eval()
    out = []
    for e in EMOTIONS:
        t = tok(EMOTION_PROMPTS[e], padding=True, return_tensors="pt").to(device)
        z = model.get_text_features(**t)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.mean(0)
        out.append(z / z.norm())
    return torch.stack(out).cpu()


class PresenceHead(nn.Module):
    def __init__(self, d=128, n_classes=9, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, h):
        return self.net(h)


class TextAnchoredQueries(nn.Module):
    """q_c = W_c h + U e_c,  with e_c the frozen CLIP text embedding of emotion c.

    Replace your current per-class query projection with this module.
    """

    def __init__(self, text_emb, d=128):
        super().__init__()
        n, t_dim = text_emb.shape
        self.register_buffer("text_emb", text_emb)       # (9, 512), frozen
        self.query_proj = nn.Linear(d, n * d)            # the original h -> q_c path
        self.text_proj = nn.Linear(t_dim, d, bias=False)  # trainable text anchor
        self.n, self.d = n, d

    def forward(self, h):                                # h: (B, d)
        q = self.query_proj(h).view(-1, self.n, self.d)  # (B, 9, d)
        return q + self.text_proj(self.text_emb).unsqueeze(0)


class RegionalGate(nn.Module):
    """logit_c = global_c + sigmoid(beta) * regional_c,  beta initialised to init."""

    def __init__(self, init=-2.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init)))

    def forward(self, global_logits, regional_logits):
        return global_logits + torch.sigmoid(self.beta) * regional_logits
