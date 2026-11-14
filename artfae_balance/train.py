"""Train ArtFAE-Net M1 (+ optional balance add-ons) on artfae_balance features. Never reads test.

Optimisation copies the paper's M1 run (artfae.cli train / artfae_m1 c2c10f53 config): AdamW
lr 3e-4 wd 0.01, cosine schedule over 20 epochs, batch 32, AMP fp16 + GradScaler, grad-clip 1.0,
patience 5, checkpoint = best val macro-F1 (artfae.evaluation.metrics.evaluate, strict >).
One deliberate difference: an fp16 overflow step is skipped by GradScaler (standard AMP) instead of
raising, as artfae.cli's error_if_nonfinite=True did (it crashed the paper's own B5 run). Steps with
finite gradients are unchanged; skipped steps are counted in history.json.

  R1  defaults                                  paper loss and shard-shuffled order
  R2  --sampler --tau-train T                   balanced sampler + logit-adjusted soft CE
  R3  R2 + --lambda-present L                   PresenceHead on h + PresenceBCE
  R4  R3 + --text-queries --regional-gate -2    text-anchored ECSA queries + regional gate

usage: python -m artfae_balance.train --out runs/R1_s42 --seed 42 [options]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import EMOTIONS
from .data import make_balanced_sampler, train_log_prior
from .losses import LogitAdjustedSoftCE, PresenceBCE
from .model import build_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artfae.evaluation.metrics import evaluate as paper_evaluate  # noqa: E402
from artfae.features.aesthetic import fit_normalizer, normalize  # noqa: E402
from artfae.training.losses import soft_cross_entropy  # noqa: E402

PAPER_M1 = {"epochs": 20, "batch_size": 32, "lr": 3e-4, "weight_decay": 0.01, "patience": 5,
            "accumulation": 1, "clip_grad": 1.0, "mixed_precision": True,
            "architecture": {"hidden": 128, "gated": True, "spatial": True, "numerical": True, "text": True}}
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SHARD = 32  # the paper shuffled 32-artwork cache shards, then artworks within each shard


def load_split(manifest, feature_dir, device):
    df = pd.read_csv(manifest)
    d = Path(feature_dir)
    f = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("global", "attributes", "attr_text", "patches")}
    if len(f["global"]) != len(df):
        raise ValueError(f"{feature_dir} and {manifest} are out of sync")
    return {
        "df": df,
        "raw_attrs": np.asarray(f["attributes"], dtype=np.float32),
        "g": torch.from_numpy(np.array(f["global"])).to(device),
        "t": torch.from_numpy(np.array(f["attr_text"])).to(device),
        "p": torch.from_numpy(np.array(f["patches"])).to(device),          # float16 on GPU
        "y": torch.from_numpy(df[[f"p_{e}" for e in EMOTIONS]].to_numpy(np.float32)).to(device),
        "present": torch.from_numpy(df[[f"present_{e}" for e in EMOTIONS]].to_numpy(np.float32)).to(device),
    }


def batch(s, idx):
    return s["g"][idx], s["a"][idx], s["t"][idx], s["p"][idx].float()


@torch.no_grad()
def predict(model, s, bs):
    """Validation pass as in artfae.cli.report_predictions: eval mode, fp32, raw logits."""
    model.eval()
    out = [model(*batch(s, torch.arange(i, min(i + bs, len(s["y"])), device=s["y"].device)))
           for i in range(0, len(s["y"]), bs)]
    return torch.cat(out).float()


def shard_order(n, generator):
    blocks = [list(range(i, min(i + SHARD, n))) for i in range(0, n, SHARD)]
    order = []
    for b in torch.randperm(len(blocks), generator=generator).tolist():
        order.extend(blocks[b][i] for i in torch.randperm(len(blocks[b]), generator=generator).tolist())
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-manifest", default="data/artemis_balanced/train.csv")
    ap.add_argument("--val-manifest", default="data/artemis_balanced/val.csv")
    ap.add_argument("--train-features", default="features/train")
    ap.add_argument("--val-features", default="features/val")
    ap.add_argument("--sampler", action="store_true", help="make_balanced_sampler (sampler_weight)")
    ap.add_argument("--tau-train", type=float, default=0.0, help="LogitAdjustedSoftCE tau; 0 = paper soft CE")
    ap.add_argument("--lambda-present", type=float, default=0.0, help=">0 adds PresenceHead + PresenceBCE")
    ap.add_argument("--text-queries", action="store_true", help="CLIP-text-anchored ECSA queries")
    ap.add_argument("--regional-gate", type=float, default=None, help="RegionalGate init (e.g. -2.0)")
    a = ap.parse_args()
    if "test" in (a.train_manifest + a.val_manifest + a.train_features + a.val_features):
        raise ValueError("training never reads test data")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(PAPER_M1, seed=a.seed, sampler=a.sampler, tau_train=a.tau_train, lambda_present=a.lambda_present,
               text_queries=a.text_queries, regional_gate=a.regional_gate,
               train_manifest=a.train_manifest, val_manifest=a.val_manifest)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    tr = load_split(a.train_manifest, a.train_features, dev)
    va = load_split(a.val_manifest, a.val_features, dev)
    if set(tr["df"].artwork_id) & set(va["df"].artwork_id):
        raise ValueError("train/val overlap")
    norm = fit_normalizer(tr["raw_attrs"])  # new train split statistics
    for s in (tr, va):
        s["a"] = torch.from_numpy(normalize(s["raw_attrs"], norm)).to(dev)

    text_emb = None
    if a.text_queries:
        from .heads import clip_emotion_text_embeddings
        text_emb = clip_emotion_text_embeddings(revision=CLIP_REVISION)
    dims = {"visual_dim": 512, "attribute_dim": 28, "text_dim": 512, "patch_dim": 768}
    model = build_model(dims, PAPER_M1["architecture"], presence=a.lambda_present > 0,
                        text_emb=text_emb, regional_gate_init=a.regional_gate).to(dev)
    cfg["model_class"] = type(model).__name__

    main_loss = LogitAdjustedSoftCE(train_log_prior(a.train_manifest), a.tau_train).to(dev) if a.tau_train else None
    pres_loss = (PresenceBCE(torch.as_tensor(tr["df"][[f"present_{e}" for e in EMOTIONS]].to_numpy(np.float32).mean(0)))
                 .to(dev) if a.lambda_present > 0 else None)
    sampler = make_balanced_sampler(a.train_manifest, seed=a.seed) if a.sampler else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1))
    (a.out / "normalizer.json").write_text(json.dumps(norm))

    n, bs = len(tr["y"]), cfg["batch_size"]
    best, stale, history, best_state = -1.0, 0, [], None
    t0 = time.monotonic()
    for epoch in range(cfg["epochs"]):
        if stale >= cfg["patience"]:
            break
        model.train()
        order = list(sampler) if sampler is not None else shard_order(n, torch.Generator().manual_seed(a.seed + epoch))
        order = torch.as_tensor(order, device=dev)
        total, skipped = 0.0, 0
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda", dtype=torch.float16):
                out = model(*batch(tr, idx), return_presence=True) if pres_loss is not None else model(*batch(tr, idx))
                logits, pres = out if pres_loss is not None else (out, None)
                y = tr["y"][idx]
                loss = main_loss(logits.float(), y) if main_loss is not None else soft_cross_entropy(logits.float(), y)
                if pres_loss is not None:
                    loss = loss + a.lambda_present * pres_loss(pres.float(), tr["present"][idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            scale = scaler.get_scale()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            skipped += scaler.get_scale() < scale  # GradScaler skipped a non-finite step
            total += loss.item() * len(idx)
        m = paper_evaluate(predict(model, va, bs).cpu(), va["y"].cpu())
        sched.step()
        improved = m["macro_f1"] > best
        best, stale = max(best, m["macro_f1"]), 0 if improved else stale + 1
        history.append({"epoch": epoch + 1, "train_loss": total / n, "val_macro_f1": m["macro_f1"],
                        "val_accuracy": m["accuracy"], "val_soft_ce": m["soft_ce"], "amp_skipped_steps": int(skipped),
                        "seconds": time.monotonic() - t0})
        if improved:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch + 1, "config": cfg, "normalizer": norm,
                        "emotions": list(EMOTIONS)}, a.out / "best.pt")
        (a.out / "history.json").write_text(json.dumps(history, indent=1))
        print(f"epoch={epoch + 1} macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
              f"loss={total / n:.4f} skipped={skipped} {'*' if improved else ''}", flush=True)

    model.load_state_dict(best_state)
    np.save(a.out / "val_logits.npy", predict(model, va, bs).cpu().numpy())
    sel = max(history, key=lambda h: h["val_macro_f1"])
    print(f"selected epoch {sel['epoch']} val macro_f1={sel['val_macro_f1']:.4f}; "
          f"{len(history)} epochs, {time.monotonic() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
