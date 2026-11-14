"""Third and last test evaluation: L/14@336 (7x7-pooled) R1 / R3, pre-registered.

  select : 5-seed val table (raw and prior-corrected) for every model, then writes
           runs/final_selection_v2.txt (models, comparisons, metric definitions, timestamp, sha256 of
           all code and of every checkpoint / logit file used). Never reads test.
  test   : refuses unless final_selection_v2.txt exists and every hash still matches; runs once
           (runs/final_eval_v2/ is create-once). Writes test logits for the L/14 runs, summary,
           per-class, sentiment, bootstrap and ensemble tables.

Definitions (fixed here, recorded in the selection file):
  prior correction  logits + log(val_prior) - log(train_prior); val_prior = mean val p_*, train_prior =
                    mean p_* of the run's own train manifest. Same constant on val and test.
  softCE(T)/Brier(T) per seed: T fitted on val logits of that seed and that kind (raw / corrected).
  ensemble          mean over seeds of softmax(logits [+ delta]). Its T is fitted on val on
                    log(mean val probs) and applied to log(mean test probs) for softCE(T).
  macro-F1, acc, balAcc, disgust F1   unique-top artworks (artfae_balance.evaluate).
  anger/fear AP, anger AUROC          presence (any annotator), all artworks.
  sentiment         positive = amusement+awe+contentment+excitement, negative = anger+disgust+fear+sadness,
                    something_else; F1 on artworks with a unique top sentiment.
  bootstrap         paired, 2,000 resamples of test artworks, rng seed 0, seed-averaged probabilities;
                    d = A - B; 95% percentile CI; two-sided p = min(1, 2 min((#d<=0 + 1), (#d>=0 + 1)) / (n + 1));
                    Holm adjustment across the 4 comparisons separately for each metric.

usage: python scripts/final_eval_v2.py select | test
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # before scripts/, whose artfae.py would shadow the package
from artfae_balance import EMOTIONS  # noqa: E402
from artfae_balance.evaluate import evaluate, load_targets, softmax  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
OLD_TEST = ROOT / "runs/final_eval/test"
# key: (label, run prefix, existing test-logit file pattern or None = compute from best.pt on L/14 features)
MODELS = {
    "R0n": ("R0n  B/32 paper 14K", "R0n_paper14k", ROOT / "runs/final_eval/baseline_test/R0n_s{s}_test_logits.npy"),
    "B32_R1": ("R1   B/32", "R1", OLD_TEST / "R1_s{s}_test_logits.npy"),
    "B32_R3": ("R3   B/32", "R3_lam0.3", OLD_TEST / "R3_s{s}_test_logits.npy"),
    "B32_R4b": ("R4b  B/32", "R4b_textq_gate", OLD_TEST / "R4b_s{s}_test_logits.npy"),
    "L14_R1": ("R1   L/14@336", "L14_R1", None),
    "L14_R3": ("R3   L/14@336", "L14_R3_lam0.3", None),
}
KINDS = ("raw", "corrected")
PRIMARY = ("L14_R3", "corrected")
SECONDARY = [("L14_R1", "raw"), ("L14_R1", "corrected"), ("L14_R3", "raw")]
CONTEXT = [(m, k) for m in ("R0n", "B32_R1", "B32_R3", "B32_R4b") for k in KINDS]
COMPARISONS = [  # (name, A, B)
    ("(a) L/14 R3c − R0n", ("L14_R3", "corrected"), ("R0n", "raw")),
    ("(b) L/14 R1 − B/32 R1", ("L14_R1", "raw"), ("B32_R1", "raw")),
    ("(c) L/14 R3 − L/14 R1", ("L14_R3", "raw"), ("L14_R1", "raw")),
    ("(d) L/14 R3c − L/14 R3", ("L14_R3", "corrected"), ("L14_R3", "raw")),
]
BOOT_METRICS = ["macro-F1", "mean anger+fear AP", "neg-sentiment F1", "softCE(T)"]
N_BOOT, BOOT_SEED = 2000, 0
SENT = [["amusement", "awe", "contentment", "excitement"], ["anger", "disgust", "fear", "sadness"], ["something_else"]]
SIDX = [[EMOTIONS.index(e) for e in g] for g in SENT]
COLS = ["macro-F1", "acc", "balAcc", "softCE(T)", "Brier(T)", "anger AP", "fear AP", "mean anger+fear AP",
        "disgust F1", "neg-sentiment F1", "sentiment macro-F1", "anger AUROC"]
OUT = ROOT / "runs/final_eval_v2"
SELECTION = ROOT / "runs/final_selection_v2.txt"
CODE = ["scripts/final_eval_v2.py", "scripts/run_l14_ladder.sh", "scripts/run_l14_ladder_s45_46.sh",
        "scripts/run_extract_l14.sh", "scripts/val_prior_table.py", "scripts/final_eval.py",
        "scripts/baseline_eval.py", "scripts/run_ladder.sh", "scripts/m1_paper_logits.py",
        "artfae_balance/train_backbone.py", "artfae_balance/extract_features_backbone.py",
        "artfae_balance/train.py", "artfae_balance/model.py", "artfae_balance/heads.py",
        "artfae_balance/losses.py", "artfae_balance/data.py", "artfae_balance/evaluate.py",
        "artfae_balance/extract_features.py", "src/artfae/models/network.py",
        "src/artfae/evaluation/metrics.py", "src/artfae/features/aesthetic.py"]
VAL_CSV, TEST_CSV = ROOT / "data/artemis_balanced/val.csv", ROOT / "data/artemis_balanced/test.csv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_dir(m, s):
    return ROOT / "runs" / f"{MODELS[m][1]}_s{s}"


def artifacts():
    """Every file the test step reads besides code and manifests."""
    files = []
    for m, (_, _, pat) in MODELS.items():
        for s in SEEDS:
            d = run_dir(m, s)
            files += [d / "val_logits.npy", d / "config.json"]
            files += [Path(str(pat).format(s=s))] if pat else [d / "best.pt", d / "dims.json"]
    files += [ROOT / f"features_l14/test/{k}.npy" for k in ("global", "attributes", "attr_text", "patches")]
    files += [VAL_CSV, TEST_CSV]
    return {str(f.relative_to(ROOT)): sha(f) for f in files}


def delta(m, s, log_val_prior):
    tr = pd.read_csv(ROOT / json.loads((run_dir(m, s) / "config.json").read_text())["train_manifest"])
    return log_val_prior - np.log(tr[[f"p_{e}" for e in EMOTIONS]].to_numpy().mean(0) + 1e-8)


def fit_t(logits, y):
    ce = lambda lt: -(y * np.log(softmax(logits / np.exp(lt)) + 1e-12)).sum(1).mean()
    return float(np.exp(minimize_scalar(ce, bounds=(-3, 3), method="bounded").x))


class Targets:
    def __init__(self, csv):
        _, self.y, self.present, self.uniq = load_targets(csv)
        ys = np.stack([self.y[:, i].sum(1) for i in SIDX], 1)
        self.su = np.isclose(ys, ys.max(1, keepdims=True), atol=1e-5).sum(1) == 1
        self.ys = ys.argmax(1)


def sentiment_f1(p, tg, idx=None):
    idx = np.arange(len(p)) if idx is None else idx
    m = tg.su[idx]
    ps = np.stack([p[idx][:, i].sum(1) for i in SIDX], 1)
    return f1_score(tg.ys[idx][m], ps[m].argmax(1), labels=[0, 1, 2], average=None, zero_division=0)


def metrics(logits, t, tg):
    """Full row for one seed: logits already corrected if needed, t fitted on val."""
    p = softmax(logits)
    r = evaluate(p, tg.y, tg.present, tg.uniq)
    pt = softmax(logits / t)
    sf = sentiment_f1(p, tg)
    c = r["per_class"]
    row = dict(zip(COLS, [r["macro_f1"], r["accuracy"], r["balanced_accuracy"],
                          float(-(tg.y * np.log(pt + 1e-12)).sum(1).mean()), float(((pt - tg.y) ** 2).sum(1).mean()),
                          c["anger"]["present_ap"], c["fear"]["present_ap"],
                          (c["anger"]["present_ap"] + c["fear"]["present_ap"]) / 2,
                          c["disgust"]["f1"], sf[1], sf.mean(), c["anger"]["present_auroc"]]))
    row.update({f"F1 {e}": c[e]["f1"] for e in EMOTIONS})
    row.update({"sent F1 positive": sf[0], "sent F1 negative": sf[1], "sent F1 something_else": sf[2],
                "sent n": int(tg.su.sum())})
    return row


def fmt(v, nd=4):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f}" + (f" ± {v.std(ddof=1):.{nd}f}" if len(v) > 1 else "")


def table(rows, cols, title, nd=4, models=None, kinds=KINDS):
    df = pd.DataFrame(rows)
    lines = [f"### {title}", "", "| model | n | logits | " + " | ".join(cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for m, k in models or [(m, k) for m in MODELS for k in kinds]:
        g = df[(df.model == m) & (df.kind == k)]
        lines.append(f"| {MODELS[m][0]} | {len(g)} | {k} | " + " | ".join(fmt(g[c], nd) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def val_logits_all(tg):
    lvp = np.log(tg.y.mean(0) + 1e-8)
    out = {}
    for m in MODELS:
        for s in SEEDS:
            raw = np.load(run_dir(m, s) / "val_logits.npy").astype(np.float64)
            dl = delta(m, s, lvp)
            out[(m, s)] = {"raw": raw, "corrected": raw + dl, "delta": dl}
    return out


def select():
    if SELECTION.exists():
        sys.exit(f"{SELECTION} already exists; selection is fixed")
    tv = Targets(VAL_CSV)
    vl = val_logits_all(tv)
    rows = []
    for (m, s), d in vl.items():
        for k in KINDS:
            rows.append({"model": m, "seed": s, "kind": k, **metrics(d[k], fit_t(d[k], tv.y), tv)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "val_per_seed.csv", index=False)
    t = table(rows, COLS, "Validation, mean ± SD over seeds 42–46 (T refit on val per row: in-sample)")
    (OUT / "val_summary.md").write_text(t)
    vdf = pd.DataFrame(rows)
    lab = lambda mk: f"{MODELS[mk[0]][0].strip()} [{mk[1]}]"
    sel = {
        "evaluation": "third and last test evaluation; no tuning",
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": SEEDS,
        "primary_model": lab(PRIMARY) + " = L/14 R3 (lambda 0.3) + prior correction log(val_prior) - log(train_prior)",
        "secondary_models": [lab(x) for x in SECONDARY],
        "context_models": [lab(x) for x in CONTEXT] + ["B/32 test logits reused from runs/final_eval (existing)"],
        "run_prefixes": {m: v[1] for m, v in MODELS.items()},
        "comparisons": [{"name": n, "A": lab(a), "B": lab(b)} for n, a, b in COMPARISONS],
        "bootstrap": {"metrics": BOOT_METRICS, "resamples": N_BOOT, "rng_seed": BOOT_SEED,
                      "probabilities": "seed-averaged softmax over 5 seeds",
                      "p_value": "two-sided, min(1, 2*min(#d<=0 + 1, #d>=0 + 1)/(n + 1))",
                      "multiplicity": "Holm across the 4 comparisons, separately per metric",
                      "softCE(T)": "one T per ensemble, fitted on val on log(mean val probs)"},
        "definitions": __doc__.split("Definitions")[1].split("usage:")[0].strip(),
        "val_mean": {lab((m, k)): {c: round(float(vdf[(vdf.model == m) & (vdf.kind == k)][c].mean()), 6)
                                   for c in ("macro-F1", "mean anger+fear AP", "neg-sentiment F1")}
                     for m in MODELS for k in KINDS},
        "code_sha256": {f: sha(ROOT / f) for f in CODE},
        "artifact_sha256": artifacts(),
    }
    SELECTION.write_text(json.dumps(sel, indent=1) + "\n")
    print(t); print(f"written {SELECTION}")


@torch.no_grad()
def l14_logits(d, split):
    from artfae.features.aesthetic import normalize
    from artfae_balance.model import build_model
    from artfae_balance.train import load_split, predict
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert not cfg["text_queries"] and cfg["regional_gate"] is None
    dims = json.loads((d / "dims.json").read_text())
    model = build_model(dims, cfg["architecture"], presence=cfg["lambda_present"] > 0)
    model.load_state_dict(ck["model"], strict=True)
    model.to(dev)
    s = load_split(ROOT / f"data/artemis_balanced/{split}.csv", ROOT / f"features_l14/{split}", dev)
    s["a"] = torch.from_numpy(normalize(s["raw_attrs"], ck["normalizer"])).to(dev)
    return predict(model, s, cfg["batch_size"]).cpu().numpy()


def boot_stats(p, logp_t, tg, idx):
    u = idx[tg.uniq[idx]]
    yt, pt = tg.y[u].argmax(1), p[u].argmax(1)
    cm = np.bincount(yt * 9 + pt, minlength=81).reshape(9, 9)
    den = cm.sum(0) + cm.sum(1)
    f1 = float(np.where(den > 0, 2 * np.diag(cm) / np.maximum(den, 1), 0).mean())
    ap = np.mean([average_precision_score(tg.present[idx, c], p[idx, c]) for c in (EMOTIONS.index("anger"),
                                                                                   EMOTIONS.index("fear"))])
    neg = sentiment_f1(p, tg, idx)[1]
    ce = float(-(tg.y[idx] * logp_t[idx]).sum(1).mean())
    return np.array([f1, ap, neg, ce])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    adj = np.maximum.accumulate([min(1.0, (m - i) * p[j]) for i, j in enumerate(o)])
    out = np.empty(m); out[o] = adj
    return out


def test():
    if not SELECTION.exists():
        sys.exit("run `select` first")
    sel = json.loads(SELECTION.read_text())
    bad = [f for f in CODE if sel["code_sha256"].get(f) != sha(ROOT / f)]
    bad += [f for f, h in sel["artifact_sha256"].items() if sha(ROOT / f) != h]
    if bad or set(sel["code_sha256"]) != set(CODE):
        sys.exit(f"changed since selection: {bad}")
    tdir = OUT / "test"
    tdir.mkdir(parents=False, exist_ok=False)  # create-once: refuses a second test evaluation
    tv, tt = Targets(VAL_CSV), Targets(TEST_CSV)
    vl = val_logits_all(tv)

    rows, ens_p = [], {}
    for m, (_, _, pat) in MODELS.items():
        for s in SEEDS:
            d = run_dir(m, s)
            if pat is None:
                chk = l14_logits(d, "val")  # integrity: recomputed val logits must match the saved ones
                assert np.abs(chk - vl[(m, s)]["raw"]).max() < 1e-3, f"{d}: val logits do not reproduce"
                raw = l14_logits(d, "test")
                np.save(tdir / f"{m}_s{s}_test_logits.npy", raw)
            else:
                raw = np.load(str(pat).format(s=s))
            raw = raw.astype(np.float64)
            for k in KINDS:
                lt = raw + (vl[(m, s)]["delta"] if k == "corrected" else 0)
                t = fit_t(vl[(m, s)][k], tv.y)  # T from val only
                rows.append({"model": m, "seed": s, "kind": k, "T": t, **metrics(lt, t, tt)})
                ens_p.setdefault((m, k), {"val": [], "test": []})
                ens_p[(m, k)]["val"].append(softmax(vl[(m, s)][k])); ens_p[(m, k)]["test"].append(softmax(lt))
            print(f"{m}_s{s}: test macro-F1 raw {rows[-2]['macro-F1']:.4f} corrected {rows[-1]['macro-F1']:.4f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT / "test_per_seed.csv", index=False)

    # ensembles: seed-averaged probabilities; T fitted on val on log(mean val probs)
    ens = {}
    for mk, v in ens_p.items():
        pv, pt = np.mean(v["val"], 0), np.mean(v["test"], 0)
        t = fit_t(np.log(pv + 1e-12), tv.y)
        ens[mk] = (pt, np.log(softmax(np.log(pt + 1e-12) / t) + 1e-12), t)

    full = np.arange(len(tt.y))
    rng = np.random.default_rng(BOOT_SEED)
    idxs = [rng.integers(0, len(tt.y), len(tt.y)) for _ in range(N_BOOT)]
    boot = {}
    for name, a, b in COMPARISONS:
        pa, la, _ = ens[a]; pb, lb, _ = ens[b]
        obs = boot_stats(pa, la, tt, full) - boot_stats(pb, lb, tt, full)
        d = np.array([boot_stats(pa, la, tt, i) - boot_stats(pb, lb, tt, i) for i in idxs])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        p = np.minimum(1, 2 * np.minimum((d <= 0).sum(0) + 1, (d >= 0).sum(0) + 1) / (N_BOOT + 1))
        boot[name] = {mt: {"diff": float(obs[j]), "ci": [float(lo[j]), float(hi[j])], "p": float(p[j])}
                      for j, mt in enumerate(BOOT_METRICS)}
        print(f"bootstrap {name} done", flush=True)
    for mt in BOOT_METRICS:
        for name, ph in zip(boot, holm([boot[n][mt]["p"] for n in boot])):
            boot[name][mt]["p_holm"] = float(ph)
    (OUT / "bootstrap.json").write_text(json.dumps(boot, indent=1))

    pc_models = [("R0n", "raw"), ("B32_R1", "raw"), ("L14_R1", "raw"), ("L14_R3", "corrected")]
    df = pd.DataFrame(rows)
    sup = evaluate(softmax(np.zeros_like(tt.y)), tt.y, tt.present, tt.uniq)["per_class"]
    pcl = ["### Test per-class F1 (unique-top artworks, mean ± SD over 5 seeds)", "",
           "| emotion | support | " + " | ".join(f"{MODELS[m][0].strip()} [{k}]" for m, k in pc_models) + " |",
           "|---|---|" + "---|" * len(pc_models)]
    for e in EMOTIONS:
        pcl.append(f"| {e} | {sup[e]['support']} | " + " | ".join(
            fmt(df[(df.model == m) & (df.kind == k)][f"F1 {e}"], 3) for m, k in pc_models) + " |")

    bl = [f"### Paired bootstrap on test, seed-averaged probabilities ({N_BOOT:,} resamples); "
          "p two-sided, Holm across the 4 comparisons per metric", "",
          "| comparison | metric | Δ (A − B) | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|"]
    for name in boot:
        for mt in BOOT_METRICS:
            r = boot[name][mt]
            bl.append(f"| {name} | {mt} | {r['diff']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | "
                      f"{r['p']:.4f} | {r['p_holm']:.4f} |")

    el = ["### Secondary: seed-averaged (5-seed ensemble) test metrics", "",
          "| model | logits | macro-F1 | mean anger+fear AP | neg-sentiment F1 | softCE(T) | T (val) |",
          "|---|---|---|---|---|---|---|"]
    for (m, k), (p, lp, t) in ens.items():
        v = boot_stats(p, lp, tt, full)
        el.append(f"| {MODELS[m][0]} | {k} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f} | {t:.3f} |")

    scols = ["sent F1 positive", "sent F1 negative", "sent F1 something_else", "sentiment macro-F1"]
    report = "\n".join([
        f"Pre-registered {sel['selected_at_utc']}. Primary: {sel['primary_model']}", "",
        table(rows, COLS, "Test, mean ± SD over seeds 42–46 (T fitted on val per seed)"),
        "\n".join(pcl) + "\n",
        table(rows, scols, f"Test sentiment level (n = {int(tt.su.sum())} artworks with a unique top sentiment)", 3),
        "\n".join(bl) + "\n", "\n".join(el) + "\n"])
    (OUT / "test_summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
