"""Final five-seed evaluation of the training ladder.

  select : val summary (mean ± SD over seeds) -> primary model by the pre-registered rule
           (highest mean val macro-F1; within 0.005, higher mean anger+fear presence AP wins).
           Writes runs/final_selection.txt (timestamp + sha256 of every code file) and
           runs/final_eval/val_summary.md. Never reads test.
  test   : requires runs/final_selection.txt with unchanged code hashes; runs once.
           Test logits for every seed, artfae_balance.evaluate reports, val-fitted temperature,
           test summary, paired bootstrap, per-class F1. Output: runs/final_eval/.

usage: python scripts/final_eval.py select | test
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # before scripts/, whose artfae.py would shadow the package
from artfae_balance import EMOTIONS  # noqa: E402
from artfae_balance.evaluate import evaluate, load_targets, softmax  # noqa: E402

RUNS = {"R1": "R1", "R2b": "R2b_sampler", "R3": "R3_lam0.3", "R3g": "R3g_gate", "R4b": "R4b_textq_gate"}
SEEDS = [42, 43, 44, 45, 46]
LADDER = [("R2b", "R1"), ("R3", "R2b"), ("R3g", "R3"), ("R4b", "R3g")]
TIE = 0.005
OUT = ROOT / "runs/final_eval"
SELECTION = ROOT / "runs/final_selection.txt"
CODE = ["scripts/final_eval.py", "scripts/run_ladder.sh", "scripts/m1_paper_logits.py",
        "artfae_balance/train.py", "artfae_balance/model.py", "artfae_balance/heads.py",
        "artfae_balance/losses.py", "artfae_balance/data.py", "artfae_balance/evaluate.py",
        "artfae_balance/extract_features.py", "src/artfae/models/network.py",
        "src/artfae/evaluation/metrics.py", "src/artfae/features/aesthetic.py"]
I_ANGER, I_FEAR = EMOTIONS.index("anger"), EMOTIONS.index("fear")


def code_hashes():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE}


def run_dir(name, seed):
    return ROOT / "runs" / f"{RUNS[name]}_s{seed}"


def fit_temperature(logits, y_soft):
    ce = lambda lt: float(-(y_soft * np.log(softmax(logits / np.exp(lt)) + 1e-12)).sum(1).mean())
    return float(np.exp(minimize_scalar(ce, bounds=(-3, 3), method="bounded").x))


def calibrated(logits, y_soft, t):
    p = softmax(logits / t)
    return float(-(y_soft * np.log(p + 1e-12)).sum(1).mean()), float(((p - y_soft) ** 2).sum(1).mean())


def metrics_row(r, ce_t, brier_t, t):
    c = r["per_class"]
    return {"macro_f1": r["macro_f1"], "acc": r["accuracy"], "balAcc": r["balanced_accuracy"],
            "softCE": r["soft_ce"], "Brier": r["brier"], "T": t, "softCE(T)": ce_t, "Brier(T)": brier_t,
            "anger_AP": c["anger"]["present_ap"], "fear_AP": c["fear"]["present_ap"],
            "mean_anger_fear_AP": (c["anger"]["present_ap"] + c["fear"]["present_ap"]) / 2,
            "anger_npred": c["anger"]["n_predicted"], "fear_F1": c["fear"]["f1"]}


COLS = ["macro_f1", "acc", "balAcc", "softCE", "Brier", "softCE(T)", "Brier(T)", "anger_AP", "fear_AP",
        "mean_anger_fear_AP", "anger_npred", "fear_F1"]


def summary_table(rows, title):
    """rows: list of dicts with 'run', 'seed', metric columns -> markdown mean ± SD (ddof=1)."""
    df = pd.DataFrame(rows)
    lines = [f"### {title}", "", "| run | n | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for run, g in df.groupby("run", sort=False):
        cells = []
        for c in COLS:
            v = g[c].to_numpy(float)
            fmt = "{:.1f}" if c == "anger_npred" else "{:.4f}"
            cells.append(fmt.format(v.mean()) + (f" ± {fmt.format(v.std(ddof=1))}" if len(v) > 1 else ""))
        lines.append(f"| {run} | {len(g)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def val_rows():
    _, y_val, _, _ = load_targets(ROOT / "data/artemis_balanced/val.csv")
    rows = []
    for name in RUNS:
        for seed in SEEDS:
            d = run_dir(name, seed)
            r = json.loads((d / "val_report.json").read_text())
            logits = np.load(d / "val_logits.npy").astype(np.float64)
            t = fit_temperature(logits, y_val)
            rows.append({"run": name, "seed": seed, **metrics_row(r, *calibrated(logits, y_val, t), t)})
    return rows


def select():
    if SELECTION.exists():
        sys.exit(f"{SELECTION} already exists; selection is fixed")
    rows = val_rows()
    df = pd.DataFrame(rows)
    means = df.groupby("run", sort=False)[["macro_f1", "mean_anger_fear_AP"]].mean()
    top = means["macro_f1"].idxmax()
    contenders = means[means["macro_f1"] >= means.loc[top, "macro_f1"] - TIE]
    primary = contenders["mean_anger_fear_AP"].idxmax()
    OUT.mkdir(parents=True, exist_ok=True)
    table = summary_table(rows, "Validation, mean ± SD over seeds 42–46")
    (OUT / "val_summary.md").write_text(table)
    pd.DataFrame(rows).to_csv(OUT / "val_per_seed.csv", index=False)
    rule = (f"highest mean val macro-F1 ({top} = {means.loc[top, 'macro_f1']:.4f}); runs within {TIE}: "
            f"{', '.join(f'{k} ({v:.4f})' for k, v in contenders['macro_f1'].items())}; "
            f"tie-break by mean anger+fear presence AP: "
            f"{', '.join(f'{k} ({v:.4f})' for k, v in contenders['mean_anger_fear_AP'].items())}")
    SELECTION.write_text(json.dumps({
        "primary_model": primary, "run_prefix": RUNS[primary], "seeds": SEEDS,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(), "rule": rule,
        "val_mean_macro_f1": means["macro_f1"].round(6).to_dict(),
        "val_mean_anger_fear_ap": means["mean_anger_fear_AP"].round(6).to_dict(),
        "code_sha256": code_hashes()}, indent=1) + "\n")
    print(table); print(f"PRIMARY = {primary}\n{rule}\nwritten {SELECTION}")


@torch.no_grad()
def test_logits(d):
    from artfae.features.aesthetic import normalize
    from artfae_balance.heads import clip_emotion_text_embeddings
    from artfae_balance.model import build_model
    from artfae_balance.train import CLIP_REVISION, load_split, predict
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    text_emb = clip_emotion_text_embeddings(revision=CLIP_REVISION) if cfg["text_queries"] else None
    dims = {"visual_dim": 512, "attribute_dim": 28, "text_dim": 512, "patch_dim": 768}
    model = build_model(dims, cfg["architecture"], presence=cfg["lambda_present"] > 0, text_emb=text_emb,
                        regional_gate_init=cfg["regional_gate"])
    model.load_state_dict(ck["model"], strict=True)
    model.to(dev)
    s = load_split(ROOT / "data/artemis_balanced/test.csv", ROOT / "features/test", dev)
    s["a"] = torch.from_numpy(normalize(s["raw_attrs"], ck["normalizer"])).to(dev)
    return predict(model, s, cfg["batch_size"]).cpu().numpy()


def macro_f1_fast(y, p, k=9):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm); denom = cm.sum(0) + cm.sum(1)
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0).mean())


def boot_stats(probs, y_soft, present, uniq, idx):
    u = uniq[idx]
    f1 = macro_f1_fast(y_soft[idx][u].argmax(1), probs[idx][u].argmax(1))
    ap = np.mean([average_precision_score(present[idx, c], probs[idx, c]) for c in (I_ANGER, I_FEAR)])
    return f1, ap


def bootstrap(pa, pb, y_soft, present, uniq, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    full = np.arange(len(y_soft))
    fa, aa = boot_stats(pa, y_soft, present, uniq, full)
    fb, ab = boot_stats(pb, y_soft, present, uniq, full)
    d = np.empty((n, 2))
    for i in range(n):
        idx = rng.integers(0, len(y_soft), len(y_soft))
        f1a, apa = boot_stats(pa, y_soft, present, uniq, idx)
        f1b, apb = boot_stats(pb, y_soft, present, uniq, idx)
        d[i] = f1a - f1b, apa - apb
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return {"macro_f1_diff": fa - fb, "macro_f1_ci": [lo[0], hi[0]],
            "ap_diff": aa - ab, "ap_ci": [lo[1], hi[1]],
            "p_macro_f1_le_0": float((d[:, 0] <= 0).mean()), "p_ap_le_0": float((d[:, 1] <= 0).mean())}


def test():
    if not SELECTION.exists():
        sys.exit("run `select` first: the primary model must be fixed before test")
    sel = json.loads(SELECTION.read_text())
    changed = [f for f, h in code_hashes().items() if sel["code_sha256"][f] != h]
    if changed:
        sys.exit(f"code changed after selection: {changed}")
    tdir = OUT / "test"
    tdir.mkdir(parents=True, exist_ok=False)  # refuses a second test evaluation
    primary = sel["primary_model"]
    test_csv = ROOT / "data/artemis_balanced/test.csv"
    _, y_val, _, _ = load_targets(ROOT / "data/artemis_balanced/val.csv")
    _, y_test, present, uniq = load_targets(test_csv)

    rows, probs, reports = [], {}, {}
    entries = [("R0", None, ROOT / "runs/m1_paper")] + [(n, s, run_dir(n, s)) for n in RUNS for s in SEEDS]
    for name, seed, d in entries:
        tag = name if seed is None else f"{name}_s{seed}"
        if seed is None:
            logits = np.load(d / "test_logits.npy")
        else:
            logits = test_logits(d)
        np.save(tdir / f"{tag}_test_logits.npy", logits)
        out = subprocess.run([sys.executable, "-m", "artfae_balance.evaluate", "--manifest", str(test_csv),
                              "--logits", str(tdir / f"{tag}_test_logits.npy"),
                              "--out", str(tdir / f"{tag}_test_report.json")],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        (tdir / f"{tag}_test_report.txt").write_text(out)
        r = json.loads((tdir / f"{tag}_test_report.json").read_text())
        t = fit_temperature(np.load(d / "val_logits.npy").astype(np.float64), y_val)  # T from val only
        rows.append({"run": name, "seed": seed if seed is not None else "paper",
                     **metrics_row(r, *calibrated(logits.astype(np.float64), y_test, t), t)})
        probs.setdefault(name, []).append(softmax(logits.astype(np.float64)))
        reports.setdefault(name, []).append(r)
        print(f"{tag}: test macro-F1 {r['macro_f1']:.4f}  (T from val = {t:.3f})", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "test_per_seed.csv", index=False)
    table = summary_table(rows, "Test, mean ± SD over seeds 42–46 (R0: single paper checkpoint)")

    # per-class F1, mean ± SD over seeds
    pc = ["### Test per-class F1 (mean ± SD over seeds)", "",
          "| emotion | support | " + " | ".join(["R0", "R1", primary] if primary != "R1" else ["R0", "R1"]) + " |",
          "|---|---|" + "---|" * (3 if primary != "R1" else 2)]
    for e in EMOTIONS:
        cells = []
        for name in (["R0", "R1", primary] if primary != "R1" else ["R0", "R1"]):
            v = np.array([r["per_class"][e]["f1"] for r in reports[name]])
            cells.append(f"{v.mean():.3f}" + (f" ± {v.std(ddof=1):.3f}" if len(v) > 1 else ""))
        pc.append(f"| {e} | {reports['R0'][0]['per_class'][e]['support']} | " + " | ".join(cells) + " |")

    # paired bootstrap on seed-averaged probabilities
    avg = {k: np.mean(v, 0) for k, v in probs.items()}
    pairs = [(primary, "R1")] + [p for p in LADDER if p != (primary, "R1")]
    boot, bl = {}, ["### Paired bootstrap over test artworks (2,000 resamples, seed-averaged probabilities)", "",
                    "| comparison | Δ macro-F1 | 95% CI | Δ mean anger+fear AP | 95% CI |", "|---|---|---|---|---|"]
    for a_, b_ in pairs:
        res = bootstrap(avg[a_], avg[b_], y_test, present, uniq)
        boot[f"{a_} - {b_}"] = res
        bl.append(f"| {a_} − {b_}{' (primary vs R1)' if (a_, b_) == (primary, 'R1') else ''} | "
                  f"{res['macro_f1_diff']:+.4f} | [{res['macro_f1_ci'][0]:+.4f}, {res['macro_f1_ci'][1]:+.4f}] | "
                  f"{res['ap_diff']:+.4f} | [{res['ap_ci'][0]:+.4f}, {res['ap_ci'][1]:+.4f}] |")
    (OUT / "bootstrap.json").write_text(json.dumps(boot, indent=1))
    ens = ["### Seed-averaged (ensemble-of-5) test macro-F1 used in the bootstrap", "",
           "| run | macro-F1 | mean anger+fear AP |", "|---|---|---|"]
    full = np.arange(len(y_test))
    for k, p in avg.items():
        f1, ap = boot_stats(p, y_test, present, uniq, full)
        ens.append(f"| {k} | {f1:.4f} | {ap:.4f} |")
    report = "\n".join([f"Primary model (fixed {sel['selected_at_utc']}): **{primary}**", "", table,
                        "\n".join(pc) + "\n", "\n".join(bl) + "\n", "\n".join(ens) + "\n"])
    (OUT / "test_summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
