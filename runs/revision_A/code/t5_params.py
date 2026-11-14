"""Task 5: total and trainable parameter counts for every model checkpoint used so far.

Counts nn.Parameters only (buffers such as BatchNorm running stats, num_batches_tracked, the R4
text_emb buffer and ArtFAENet's non-persistent attribute_mask are excluded).

Trainable rules (from the training code):
  artfae_balance heads (runs/*), src/artfae heads (B3/B4/B5, M1, simple fusion, component runs),
  B2 linear head: all head parameters; the CLIP encoder is frozen and features are cached.
  B1 ResNet50 --freeze-backbone: only network.fc (scripts/train_resnet.py lines 30-32).
  /home/dev pipeline fine-tunes (src/artfae/pipeline/finetune.py param_groups): encoder blocks
  i >= frozen_blocks, post-norms/pooling, head; embeddings and pre-layernorm frozen when
  frozen_blocks > 0.
Frozen feature extractors (CLIP B/32, CLIP L/14@336) are counted from the local HF cache.
"""
import glob
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch

from common import OUT, ROOT

HOME = Path("/home/dev/ArtFAE-Net/results")
BUFFER = re.compile(r"(running_mean|running_var|num_batches_tracked|^text_emb$|attribute_mask)")


def params(sd):
    return OrderedDict((k, v) for k, v in sd.items() if torch.is_tensor(v) and not BUFFER.search(k))


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def head_row(group, path, sd, note=""):
    n = sum(v.numel() for v in params(sd).values())
    return {"group": group, "checkpoint": str(path), "total": n, "trainable": n, "note": note}


def pipeline_row(group, path):
    c = load(path)
    sd, fb = params(c["model"]), c["config"].get("frozen_blocks", 0)
    total = trainable = 0
    for k, v in sd.items():
        total += v.numel()
        m = re.search(r"encoder\.layers\.(\d+)\.", k)
        frozen = (m and int(m.group(1)) < fb) or (fb > 0 and ("embeddings" in k or "pre_layrnorm" in k))
        trainable += 0 if frozen else v.numel()
    n_layers = 1 + max(int(m.group(1)) for k in sd if (m := re.search(r"encoder\.layers\.(\d+)\.", k)))
    return {"group": group, "checkpoint": str(path), "total": total, "trainable": trainable,
            "note": f"{c['config'].get('encoder')}, frozen_blocks={fb}/{n_layers}"}


def clip_rows():
    """Counts depend only on the architecture config: instantiate from the cached config.json."""
    from transformers import CLIPConfig, CLIPModel
    hub = Path.home() / ".cache/huggingface/hub"
    rows = []
    for name, snap in (("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
                       ("openai/clip-vit-large-patch14-336", "ce19dc912ca5cd21c8a653c79e251e808ccabcd1")):
        cfg = hub / f"models--{name.replace('/', '--')}/snapshots/{snap}/config.json"
        m = CLIPModel(CLIPConfig.from_pretrained(cfg))
        n = lambda mod: sum(p.numel() for p in mod.parameters())
        vis = n(m.vision_model) + n(m.visual_projection)
        txt = n(m.text_model) + n(m.text_projection)
        rows.append({"group": "frozen feature extractor", "checkpoint": name, "total": vis + txt, "trainable": 0,
                     "note": f"vision+proj {vis:,}; text+proj {txt:,} (text encodes the describe() sentence); "
                             f"logit_scale excluded"})
    return rows


def main():
    rows = []
    # artfae_balance ladder (this workspace)
    for d in sorted(glob.glob(str(ROOT / "runs/*_s4[2-6]"))):
        p = Path(d) / "best.pt"
        if p.exists():
            fam = Path(d).name.rsplit("_s", 1)[0]
            rows.append(head_row(f"runs/{fam}", p, load(p)["model"]))
    # historical workspace baselines
    for p in sorted(glob.glob(str(ROOT / "results/cpu_*_20261022/*/best.pt"))):
        rows.append(head_row(f"workspace {load(p)['config']['variant']}", p, load(p)["model"]))
    rows.append(head_row("workspace B2 (CLIP B/32 linear head)", ROOT / "results/clip_b32_full/best_head.pt",
                         load(ROOT / "results/clip_b32_full/best_head.pt")["head_state_dict"]))
    p = ROOT / "results/resnet50_frozen_cpu_20261022/best.pt"
    sd = params(load(p)["model"])
    rows.append({"group": "workspace B1 (ResNet50, frozen backbone)", "checkpoint": str(p),
                 "total": sum(v.numel() for v in sd.values()),
                 "trainable": sum(v.numel() for k, v in sd.items() if k.startswith("network.fc.")), "note": ""})
    # /home/dev project (paper M1 and its baselines / components)
    for sub in ("artfae_m1", "baseline_simple_fusion", "matched_component_runs", "research_baselines/*"):
        for p in sorted(glob.glob(str(HOME / sub / "*/best_model.pt"))):
            c = load(p)
            arch = c["config"].get("architecture", {})
            note = ",".join(k for k in ("gated", "spatial", "numerical", "text") if arch.get(k)) or "-"
            rows.append(head_row(f"home {sub.split('/')[0]} {c['config'].get('variant')}", p, c["model"],
                                 f"hidden {arch.get('hidden')}, branches: {note}" + (" (paper M1, runs/m1_paper)" if "c2c10f53" in p else "")))
    for p in sorted(glob.glob(str(HOME / "clip_b32_full/best_head.pt"))):
        rows.append(head_row("home B2 (CLIP B/32 linear head)", p, load(p)["head_state_dict"]))
    for base in (HOME / "pipeline", HOME / "updated_artemis_20261031/pipeline"):
        tag = "home pipeline 43K" if "updated" in str(base) else "home pipeline 14K"
        for p in sorted(glob.glob(str(base / "runs/*/model.pt"))):
            rows.append(pipeline_row(f"{tag} {Path(p).parent.name.rsplit('_s', 1)[0]}", p))
        for b in sorted(glob.glob(str(base / "bundles/*"))):
            members = sorted(glob.glob(f"{b}/member*.pt"))
            rs = [pipeline_row("", m) for m in members]
            rows.append({"group": f"{tag} bundle {Path(b).name}", "checkpoint": b,
                         "total": sum(r["total"] for r in rs), "trainable": sum(r["trainable"] for r in rs),
                         "note": f"{len(rs)} members: " + "; ".join(r["note"] for r in rs)})
    rows += clip_rows()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t5_params_all_checkpoints.csv", index=False)

    # one line per distinct architecture within a family (identical counts collapse)
    df["family"] = [c if g == "frozen feature extractor" else g for g, c in zip(df.group, df.checkpoint)]
    g = df.groupby(["family", "total", "trainable", "note"], sort=False).size().reset_index(name="n")
    md = ["## Task 5. Parameter counts", "",
          "nn.Parameters only (buffers excluded). One row per distinct architecture within a model family; "
          "`n ckpt` = number of checkpoints with exactly these counts. Heads on cached features add the frozen "
          "extractor at inference (last rows). Per-checkpoint list: t5_params_all_checkpoints.csv.", "",
          "| model family | n ckpt | total params | trainable params | note |", "|---|---|---|---|---|"]
    for _, r in g.iterrows():
        md.append(f"| {r.family} | {r.n} | {r.total:,} | {r.trainable:,} | {r.note} |")
    md += ["", "Inference-time totals for the Phase-A models (head + frozen extractor):", ""]
    clip = dict(zip(df[df.group == "frozen feature extractor"].checkpoint, df[df.group == "frozen feature extractor"].total))
    for lab, fam, enc in (("R0n / B/32 R1", "runs/R1", "openai/clip-vit-base-patch32"),
                          ("L/14 R1", "runs/L14_R1", "openai/clip-vit-large-patch14-336"),
                          ("L/14 R3 / R3c", "runs/L14_R3_lam0.3", "openai/clip-vit-large-patch14-336")):
        h = int(df[df.group == fam].total.iloc[0])
        md.append(f"- {lab}: head {h:,} + {enc.split('/')[1]} {clip[enc]:,} = {h + clip[enc]:,} "
                  f"(trainable {h:,} = {100 * h / (h + clip[enc]):.2f}%)")
    (OUT / "t5_params.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
