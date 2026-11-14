"""Steps 1-2: controlled L/14 ablations with one recipe and checkpoint selection by val soft CE.

Recipe = artfae_balance.train / PAPER_M1, unchanged: AdamW lr 3e-4 wd 0.01, cosine over 20 epochs, batch 32,
fp16 autocast + GradScaler, grad-clip 1.0, patience 5, the same shard-shuffled order (seed + epoch) and
torch.manual_seed(seed) before the model is built. The one change for Phase B: the monitored quantity for
early stopping and checkpoint selection is the selection-split soft cross-entropy (T = 1, lower is better)
for every condition, R1 included. artfae_balance/ is imported, never modified.

Conditions (all on CLIP L/14@336 features, 768 / 28 / 768 / 49x1024):
  R1        ArtFAENet, gated AGF + ECSA, unweighted soft CE
  probe     CLIP-only linear probe: Linear(768, 9) on the global embedding
  B5        FusionBaseline('B5'): concat(global, attributes, attribute text) -> MLP 256 -> 9
  no_num    AGF without the attribute projection (its parameters removed; gate/fusion sized for 2 inputs)
  no_text   AGF without the attribute-text projection (parameters removed)
  no_attr   AGF with the visual projection only (both attribute branches and parameters removed)
  no_ecsa   ArtFAENet(spatial=False): no regional term, ECSA parameters removed
  no_agf    ArtFAENet(gated=False): same 3 projections, concatenated (no gate) -> fusion
  weighted  R1 + the paper's class-weighted soft CE: targets * w, w = bounded_vote_weights(train vote counts,
            power 0.25, min 0.5, max 2.0) (copied from /home/dev/ArtFAE-Net src/artfae/training/imbalance.py)
  crop      R1 on attributes + attribute text recomputed on CLIP's 336-px center crop
Follow-up conditions:
  probe100_lr{1e-3,3e-4,1e-4}  linear probe with its own budget: up to 100 epochs (cosine over 100), patience 10
  mlp       Linear(768, 722) -> GELU -> Linear(722, 9) on the global embedding (561,725 params, matched to no_attr)
  agf_only  no_attr without ECSA: gated AGF on the global embedding only -> head
  R3        L/14 R3 with softCE selection: artfae_balance.model.build_model(presence=True) (ArtFAENetPlus +
            PresenceHead), balanced sampler (make_balanced_sampler(train csv, seed), redrawn every epoch),
            loss = soft CE + 0.3 * PresenceBCE(train present rates); mirrors artfae_balance/train.py --sampler
            --lambda-present 0.3

usage: python train_abl.py --step 1 --conditions R1 probe ... --seeds 42 43 44 45 46
       python train_abl.py --step 2 --seeds 42 43 44          (artist-held-out, see s2_split.py)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import EMOTIONS, OUT, ROOT

from artfae.features.aesthetic import fit_normalizer, normalize  # noqa: E402
from artfae.models.network import AGF, ECSA, ArtFAENet, FusionBaseline  # noqa: E402
from artfae_balance.data import make_balanced_sampler  # noqa: E402
from artfae_balance.losses import PresenceBCE  # noqa: E402
from artfae_balance.model import build_model  # noqa: E402
from artfae_balance.train import PAPER_M1, batch, predict, shard_order  # noqa: E402

ARCH = PAPER_M1["architecture"]
CONDITIONS = ["R1", "probe", "B5", "no_num", "no_text", "no_attr", "no_ecsa", "no_agf", "weighted", "crop",
              "probe100_lr1e-3", "probe100_lr3e-4", "probe100_lr1e-4", "mlp", "agf_only", "R3",
              "probe100_lr3e-3", "probe100_lr1e-2"]
R3_LAMBDA = 0.3
MLP_HIDDEN = 722  # 778 h + 9 params; h = 722 -> 561,725 (no_attr: 561,545)


def overrides(cond):
    """Recipe overrides per condition (empty = PAPER_M1)."""
    if cond.startswith("probe100_lr"):
        return {"epochs": 100, "patience": 10, "lr": float(cond.split("_lr")[1])}
    return {}


def bounded_vote_weights(vote_counts, power=.5, minimum=.5, maximum=2.):
    """Verbatim logic of /home/dev/ArtFAE-Net/src/artfae/training/imbalance.py::bounded_vote_weights."""
    counts = np.asarray(vote_counts, dtype=np.float64)
    frequencies = counts / counts.sum()
    raw = np.maximum(frequencies, 1e-12) ** (-power)
    low, high = 0., 1. / raw.min() * maximum
    for _ in range(100):
        scale = (low + high) / 2
        if np.dot(frequencies, np.clip(raw * scale, minimum, maximum)) < 1:
            low = scale
        else:
            high = scale
    return np.clip(raw * ((low + high) / 2), minimum, maximum)


class AGFSubset(nn.Module):
    """network.AGF restricted to a subset of the three inputs; identical layer structure for the kept inputs."""

    def __init__(self, dims, keep, hidden=128, gated=True):
        super().__init__()
        self.keep = keep
        self.projections = nn.ModuleList([nn.Linear(dims[k], hidden) for k in keep])
        n = len(keep) * hidden
        self.gate = nn.Sequential(nn.Linear(n, hidden), nn.GELU(), nn.Linear(hidden, n), nn.Sigmoid()) if gated else None
        self.fusion = nn.Sequential(nn.Linear(n, hidden), nn.LayerNorm(hidden), nn.GELU())

    def forward(self, g, a, t):
        x = {"g": g, "a": a, "t": t}
        u = torch.cat([p(x[k]) for p, k in zip(self.projections, self.keep)], -1)
        return self.fusion(u * self.gate(u) if self.gate is not None else u)


class AblNet(nn.Module):
    """ArtFAENet with attribute branches removed (not zeroed): AGFSubset -> head (+ ECSA regional term)."""

    def __init__(self, d, keep, hidden=128, spatial=True):
        super().__init__()
        self.agf = AGFSubset({"g": d["visual_dim"], "a": d["attribute_dim"], "t": d["text_dim"]}, keep, hidden)
        self.head = nn.Linear(hidden, 9)
        self.ecsa = ECSA(d["patch_dim"], hidden) if spatial else None

    def forward(self, g, a, t, patches):
        h = self.agf(g, a, t)
        logits = self.head(h)
        if self.ecsa is not None:
            logits = logits + self.ecsa(h, patches)[0]
        return logits


class MLP(nn.Module):
    def __init__(self, d, hidden=MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d["visual_dim"], hidden), nn.GELU(), nn.Linear(hidden, 9))

    def forward(self, g, a, t, patches):
        return self.net(g)


class Probe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d["visual_dim"], 9)

    def forward(self, g, a, t, patches):
        return self.head(g)


def build(cond, d):
    if cond == "R3":
        return build_model(d, ARCH, presence=True)
    if cond in ("R1", "weighted", "crop"):
        return ArtFAENet(**d, **ARCH)
    if cond == "no_ecsa":
        return ArtFAENet(**d, **{**ARCH, "spatial": False})
    if cond == "no_agf":
        return ArtFAENet(**d, **{**ARCH, "gated": False})
    if cond == "probe" or cond.startswith("probe100"):
        return Probe(d)
    if cond == "mlp":
        return MLP(d)
    if cond == "agf_only":
        return AblNet(d, ["g"], spatial=False)
    if cond == "B5":
        return FusionBaseline(d["visual_dim"], d["attribute_dim"], d["text_dim"], "B5")
    keep = {"no_num": ["g", "t"], "no_text": ["g", "a"], "no_attr": ["g"]}[cond]
    return AblNet(d, keep)


def load_rows(manifest_df, feature_dir, rows, dev):
    """Features for the given row indices of feature_dir (aligned with manifest_df row order)."""
    d = Path(feature_dir)
    f = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("global", "attributes", "attr_text", "patches")}
    rows = np.asarray(rows)
    return {
        "df": manifest_df.iloc[rows].reset_index(drop=True),
        "raw_attrs": np.asarray(f["attributes"][rows], dtype=np.float32),
        "g": torch.from_numpy(np.ascontiguousarray(f["global"][rows])).to(dev),
        "t": torch.from_numpy(np.ascontiguousarray(f["attr_text"][rows])).to(dev),
        "p": torch.from_numpy(np.ascontiguousarray(f["patches"][rows])).to(dev),
        "y": torch.from_numpy(manifest_df.iloc[rows][[f"p_{e}" for e in EMOTIONS]].to_numpy(np.float32)).to(dev),
        "present": torch.from_numpy(manifest_df.iloc[rows][[f"present_{e}" for e in EMOTIONS]]
                                    .to_numpy(np.float32)).to(dev),
    }


def soft_ce(logits, y, w=None):
    t = y * w if w is not None else y
    return -(t * logits.log_softmax(-1)).sum(-1).mean()


def train_one(cond, seed, tr, va, out, dev, weights=None, extra_cfg=None):
    cfg = dict(PAPER_M1, seed=seed, condition=cond, selection="min selection-split soft CE (T=1)", **(extra_cfg or {}))
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    norm = fit_normalizer(tr["raw_attrs"])
    for s in (tr, va):
        s["a"] = torch.from_numpy(normalize(s["raw_attrs"], norm)).to(dev)
    d = {"visual_dim": tr["g"].shape[1], "attribute_dim": tr["raw_attrs"].shape[1],
         "text_dim": tr["t"].shape[1], "patch_dim": tr["p"].shape[2]}
    model = build(cond, d).to(dev)
    cfg.update(dims=d, model_class=type(model).__name__,
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               class_weights=None if weights is None else weights.tolist())
    w = None if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=dev)
    r3 = cond == "R3"
    if r3:  # as artfae_balance/train.py: PresenceBCE from train present rates; sampler built once, redrawn per epoch
        pres_loss = PresenceBCE(torch.as_tensor(tr["df"][[f"present_{e}" for e in EMOTIONS]]
                                                .to_numpy(np.float32).mean(0))).to(dev)
        sampler = make_balanced_sampler(ROOT / cfg["train_manifest"], seed=seed)
        cfg.update(sampler=True, lambda_present=R3_LAMBDA)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "normalizer.json").write_text(json.dumps(norm))
    n, bs = len(tr["y"]), cfg["batch_size"]
    best, stale, history, best_state = float("inf"), 0, [], None
    t0 = time.monotonic()
    for epoch in range(cfg["epochs"]):
        if stale >= cfg["patience"]:
            break
        model.train()
        order = list(sampler) if r3 else shard_order(n, torch.Generator().manual_seed(seed + epoch))
        order = torch.as_tensor(order, device=dev)
        total, skipped = 0.0, 0
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda", dtype=torch.float16):
                if r3:
                    logits, pres = model(*batch(tr, idx), return_presence=True)
                    loss = soft_ce(logits.float(), tr["y"][idx]) + R3_LAMBDA * pres_loss(pres.float(), tr["present"][idx])
                else:
                    logits = model(*batch(tr, idx))
                    loss = soft_ce(logits.float(), tr["y"][idx], w)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            scale = scaler.get_scale()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            skipped += scaler.get_scale() < scale
            total += loss.item() * len(idx)
        vl = predict(model, va, bs)
        ce = float(soft_ce(vl, va["y"]))
        sched.step()
        improved = ce < best
        best, stale = min(best, ce), 0 if improved else stale + 1
        history.append({"epoch": epoch + 1, "train_loss": total / n, "sel_soft_ce": ce,
                        "amp_skipped_steps": int(skipped), "seconds": time.monotonic() - t0})
        if improved:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "config": cfg, "normalizer": norm, "emotions": list(EMOTIONS)}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    sel = min(history, key=lambda h: h["sel_soft_ce"])
    print(f"{out.name}: selected epoch {sel['epoch']} sel softCE {sel['sel_soft_ce']:.4f}; {len(history)} epochs, "
          f"{time.monotonic() - t0:.0f}s, {cfg['trainable_parameters']:,} params", flush=True)
    return model


def step1(a, dev):
    tr_df = pd.read_csv(ROOT / "data/artemis_balanced/train.csv")
    va_df = pd.read_csv(ROOT / "data/artemis_balanced/val.csv")
    counts = tr_df[[f"n_{e}" for e in EMOTIONS]].to_numpy(float).sum(0)
    weights = bounded_vote_weights(counts, power=.25, minimum=.5, maximum=2.)
    groups = {}
    for c in a.conditions:
        fdir = "runs/revision_B/features_l14_crop336" if c == "crop" else "features_l14"
        groups.setdefault(fdir, []).append(c)
    for fdir, conds in groups.items():
        base_tr = load_rows(tr_df, ROOT / fdir / "train", np.arange(len(tr_df)), dev)
        base_va = load_rows(va_df, ROOT / fdir / "val", np.arange(len(va_df)), dev)
        for c in conds:
            for s in a.seeds:
                out = OUT / a.out_subdir / f"{c}_s{s}"
                if (out / "val_logits.npy").exists():
                    print(f"skip {out}"); continue
                tr, va = dict(base_tr), dict(base_va)
                model = train_one(c, s, tr, va, out, dev, weights if c == "weighted" else None,
                                  {"train_manifest": "data/artemis_balanced/train.csv",
                                   "val_manifest": "data/artemis_balanced/val.csv", "features": fdir,
                                   **overrides(c)})
                np.save(out / "val_logits.npy", predict(model, va, 32).cpu().numpy())


def step2(a, dev):
    sp = pd.read_csv(OUT / "step2/split.csv")
    tr_df = pd.read_csv(ROOT / "data/artemis_balanced/train.csv")
    assert (sp.artwork_id.values == tr_df.artwork_id.values).all()
    rows = {k: np.flatnonzero(sp.role.values == k) for k in ("train", "select", "holdout_unseen", "holdout_seen")}
    tr = load_rows(tr_df, ROOT / "features_l14/train", rows["train"], dev)
    se = load_rows(tr_df, ROOT / "features_l14/train", rows["select"], dev)
    ho = {k: load_rows(tr_df, ROOT / "features_l14/train", rows[k], dev) for k in ("holdout_unseen", "holdout_seen")}
    for s in a.seeds:
        out = OUT / "step2" / f"R1_s{s}"
        if (out / "holdout_unseen_logits.npy").exists():
            print(f"skip {out}"); continue
        model = train_one("R1", s, dict(tr), dict(se), out, dev, None,
                          {"train_manifest": "runs/revision_B/step2/split.csv role=train",
                           "selection_split": "runs/revision_B/step2/split.csv role=select", "features": "features_l14/train"})
        norm = json.loads((out / "normalizer.json").read_text())
        for k, h in ho.items():
            h = dict(h); h["a"] = torch.from_numpy(normalize(h["raw_attrs"], norm)).to(dev)
            np.save(out / f"{k}_logits.npy", predict(model, h, 32).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=[1, 2], required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--out-subdir", default="step1", help="run directory under runs/revision_B")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (step1 if a.step == 1 else step2)(a, dev)


if __name__ == "__main__":
    main()
