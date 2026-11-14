#!/usr/bin/env python3
"""ArtFAE-Net Stage 2: reproducible frozen CLIP ViT-B/32 emotion baseline.

Uses one IMAGE per artwork and its nine official ArtEmis vote probabilities.
Never uses human utterances as inputs. The --smoke command processes a small
subset and saves its artifacts separately from the full baseline.

Examples (run on the GPU server):
  python scripts/artfae_clip_baseline.py --mode smoke
  python scripts/artfae_clip_baseline.py --mode full

Before full evaluation, note these are CUSTOM PILOT splits, not official
ArtEmis benchmark splits. This program never touches test_artworks.csv.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

EMOTIONS = ["amusement", "awe", "contentment", "excitement", "anger",
            "disgust", "fear", "sadness", "something_else"]
MODEL_ID = "openai/clip-vit-base-patch32"


def load_manifest(csv_path, data_dir, max_rows=None):
    result = []
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"painting", "filename", "n_annotations"} | {f"p_{x}" for x in EMOTIONS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing manifest columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            filename = Path(row["filename"]).name
            if Path(filename).stem != row["painting"]:
                raise ValueError(f"Painting/filename mismatch: {row['painting']} / {filename}")
            path = data_dir / "images" / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing image: {path}")
            label = [float(row[f"p_{c}"]) for c in EMOTIONS]
            if abs(sum(label) - 1) > 0.005:
                raise ValueError(f"Non-normalized vote distribution for {row['painting']}: {label}")
            if int(row["n_annotations"]) <= 0:
                raise ValueError(f"No votes for painting {row['painting']}")
            result.append((row["painting"], path, label))
            if max_rows is not None and len(result) >= max_rows:
                break
    if not result:
        raise ValueError(f"No artworks read from {csv_path}")
    return result


def validate_unique(train, val):
    a = [r[0] for r in train]
    b = [r[0] for r in val]
    if len(a) != len(set(a)) or len(b) != len(set(b)):
        raise ValueError("Repeated painting ID inside one split")
    if set(a) & set(b):
        raise ValueError("Train/validation overlap detected")


def open_image(path):
    """Use JPEG decoder scaling before RGB conversion where possible.

    Pillow's decompression warning does not mean an image is corrupt.
    Giant non-JPEG images are stopped rather than exhausting RAM.
    """
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im.draft("RGB", (256, 256))
        if im.width * im.height > 40_000_000:
            raise ValueError(f"Very large image requires preprocessing: {path} ({im.size})")
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def build_data_loader(manifest, processor, batch_size, device):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class PaintingDataset(Dataset):
        def __len__(self):
            return len(manifest)

        def __getitem__(self, idx):
            painting, path, probs = manifest[idx]
            try:
                img = open_image(path)
                pixels = processor(images=img, return_tensors="pt")["pixel_values"][0]
            except Exception as exc:
                raise RuntimeError(f"Failed decoding/preprocessing {painting}: {path}") from exc
            return painting, pixels, torch.tensor(probs, dtype=torch.float32)

    return DataLoader(PaintingDataset(), batch_size=batch_size,
                      shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"),
                      persistent_workers=True)


def extract_features(model, processor, manifest, batch_size, device):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    loader = build_data_loader(manifest, processor, batch_size, device)
    feature_parts, target_parts, paintings = [], [], []
    model.eval()
    with torch.inference_mode():
        for ids, images, probs in tqdm(loader, desc=f"CLIP features ({len(manifest)})", unit="batch"):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(pixel_values=images)
            else:
                out = model(pixel_values=images)
            features = F.normalize(out.image_embeds.float(), dim=-1)
            feature_parts.append(features.cpu())
            target_parts.append(probs)
            paintings.extend(ids)
    return torch.cat(feature_parts), torch.cat(target_parts), paintings


def macro_f1_unambiguous(y_true, preds, n_class=9):
    import torch
    # A hard-label metric is undefined when the top human vote is tied.
    max_scores = y_true.max(dim=1).values
    tie_counts = torch.isclose(y_true, max_scores[:, None], atol=1e-5).sum(dim=1)
    mask = tie_counts == 1
    if not mask.any():
        return None, 0
    hard = y_true[mask].argmax(dim=1)
    predicted = preds[mask]
    f1 = []
    for class_idx in range(n_class):
        tp = ((hard == class_idx) & (predicted == class_idx)).sum().item()
        fp = ((hard != class_idx) & (predicted == class_idx)).sum().item()
        fn = ((hard == class_idx) & (predicted != class_idx)).sum().item()
        d = 2*tp + fp + fn
        f1.append((2*tp / d) if d else 0.0)
    return sum(f1) / len(f1), int(mask.sum().item())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["smoke", "full"])
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        ap.error("batch-size and epochs must be positive")

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; PyTorch: {torch.__version__}; CLIP: {MODEL_ID}", flush=True)
    print("Custom ArtEmis pilot splits. Test data remain untouched.", flush=True)
    data_dir = args.data_dir.resolve()
    max_train = 32 if args.mode == "smoke" else None
    max_val = 16 if args.mode == "smoke" else None
    train = load_manifest(data_dir / "train_artworks.csv", data_dir, max_train)
    val = load_manifest(data_dir / "val_artworks.csv", data_dir, max_val)
    validate_unique(train, val)
    print(f"Training: {len(train)} artworks; Validation: {len(val)} artworks", flush=True)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    train_features, train_targets, train_ids = extract_features(
        model, processor, train, args.batch_size, device)
    val_features, val_targets, val_ids = extract_features(
        model, processor, val, args.batch_size, device)
    print(f"Feature shapes: train={tuple(train_features.shape)} val={tuple(val_features.shape)}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    val_features = val_features.to(device)
    val_targets = val_targets.to(device)
    head = nn.Linear(train_features.shape[1], len(EMOTIONS)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(len(train_features), device=device)
        total = 0.0
        for batch_ids in perm.split(256):
            logits = head(train_features[batch_ids])
            loss = -(train_targets[batch_ids] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_ids)
        head.eval()
        with torch.no_grad():
            val_logits = head(val_features)
            val_loss = -(val_targets * F.log_softmax(val_logits, dim=-1)).sum(dim=-1).mean().item()
            f1, n_unambiguous = macro_f1_unambiguous(val_targets.cpu(), val_logits.argmax(-1).cpu())
        log = {"epoch": epoch, "train_soft_ce": total / len(train),
               "val_soft_ce": val_loss, "val_macro_f1_unique_top_only": f1,
               "val_unique_top_artworks": n_unambiguous}
        history.append(log)
        print(f"epoch={epoch:02d} train_soft_CE={log['train_soft_ce']:.4f} "
              f"val_soft_CE={val_loss:.4f} val_macro_F1_unambiguous={f1}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    dest = args.out_dir / f"clip_b32_{args.mode}"
    dest.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": best_state, "model_id": MODEL_ID, "emotions": EMOTIONS,
                "feature_dim": train_features.shape[1], "seed": args.seed}, dest / "best_head.pt")
    (dest / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Saved trained head and history to {dest}; best validation soft CE={best_loss:.4f}", flush=True)
    if args.mode == "smoke":
        print("SMOKE TEST PASSED. Full baseline: python scripts/artfae_clip_baseline.py --mode full")
    else:
        print("FULL BASELINE TRAINED. Do not claim test results until independent final evaluation.")


if __name__ == "__main__":
    main()
