"""Phase C holdout evaluation, as pre-registered in runs/final_selection_v3.txt. Runs once.

Before any holdout target is read it verifies, and refuses to run on any mismatch:
  - every sha256 in the selection file (code incl. this file, data/features, every checkpoint, val logits, configs)
  - the HOLDOUT block: manifest, duplicate classification, exclusion list and every holdout feature file
  - the selection is final (status does not start with "DRAFT") and HOLDOUT is filled in
  - every checkpoint reproduces its saved val logits on features_l14/val (max |diff| < 1e-4)
Output directory is create-once (refuses if it exists).

Analysis (all from the selection file): primary metric softCE (T = 1); secondary KL, Brier, macro-F1, mean
anger+fear AP, negative-sentiment F1. Paired artwork bootstrap: 2,000 resamples, default_rng(0), integers(0, n, n);
the same artwork draws for every model, every seed and every metric.
  PRIMARY CONTRAST (decisions): per-seed mean. For each model the metric is computed for each of its 5 seeds on the
    resampled artworks and averaged over the seeds, inside each resample; H1-H5 are tested on differences of these
    per-seed means; Holm across H1-H5.
  SECONDARY CONTRAST (descriptive only): 5-seed ensemble (seed-averaged probabilities), same draws, same tests,
    reported without decisions.
Primary population: full holdout. Secondary population: artworks by artists absent from the 43K train (same
statistics, own bootstrap, descriptive). Seed variability: per-seed metrics and per-seed paired differences.

usage (real):    python eval_holdout.py --selection runs/final_selection_v3.txt
usage (dry run): python eval_holdout.py --selection runs/final_selection_v3_DRAFT3.txt --dry-run \
                     --holdout-manifest M --holdout-features DIR --out SCRATCH_DIR [--dry-fake-unseen-frac 0.1]
A dry run accepts a DRAFT selection and an unregistered stand-in holdout, still verifies every registered hash,
and labels every output "DRY RUN — not a holdout result".
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from common import EMOTIONS, EPS, ROOT, F, softmax

from artfae.features.aesthetic import normalize  # noqa: E402
from artfae_balance.model import build_model  # noqa: E402
from artfae_balance.train import PAPER_M1, predict  # noqa: E402
from train_abl import build, load_rows  # noqa: E402

METRICS = ["softCE", "KL", "Brier", "macro-F1", "mean anger+fear AP", "neg-sentiment F1"]
AF = (EMOTIONS.index("anger"), EMOTIONS.index("fear"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def verify(sel, holdout_files):
    bad = []
    for block in ("code_sha256", "data_sha256", "checkpoint_sha256"):
        for f, h in sel[block].items():
            p = ROOT / f
            if not p.exists() or sha(p) != h:
                bad.append(f)
    for f, h in holdout_files.items():
        if not Path(f).exists() or sha(f) != h:
            bad.append(f)
    me = str(Path(__file__).resolve().relative_to(ROOT))
    if me not in sel["code_sha256"]:
        bad.append(f"{me} (not registered)")
    if bad:
        sys.exit("REFUSING TO RUN: hash mismatch or missing file:\n  " + "\n  ".join(bad))


def load_run(run, dev):
    ck = torch.load(ROOT / run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if "condition" in cfg:  # runs/revision_B/step1 (train_abl.py)
        model = build(cfg["condition"], cfg["dims"])
    else:  # existing artfae_balance runs (macro-F1-selected R3)
        dims = json.loads((ROOT / run / "dims.json").read_text())
        model = build_model(dims, cfg["architecture"], presence=cfg["lambda_present"] > 0)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(dev).eval(), ck["normalizer"]


@torch.no_grad()
def logits(model, norm, s):
    s = dict(s)
    s["a"] = torch.from_numpy(normalize(s["raw_attrs"], norm)).to(s["g"].device)
    return predict(model, s, 64).cpu().numpy().astype(np.float64)


def stat_vec(p, tg, idx):
    """[softCE, KL, Brier, macro-F1, mean anger+fear AP, neg-sentiment F1] on rows idx (bootstrap-safe)."""
    y, pp = tg.y[idx], p[idx]
    ce = -(y * np.log(pp + EPS)).sum(1).mean()
    kl = (y * (np.log(y + EPS) - np.log(pp + EPS))).sum(1).mean()
    br = ((pp - y) ** 2).sum(1).mean()
    u = idx[tg.uniq[idx]]
    cm = np.bincount(tg.y[u].argmax(1) * 9 + p[u].argmax(1), minlength=81).reshape(9, 9)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.where(den > 0, 2 * np.diag(cm) / np.maximum(den, 1), 0).mean()
    ap = np.mean([average_precision_score(tg.present[idx, c], pp[:, c]) for c in AF])
    neg = F.sentiment_f1(p, tg, idx)[1]
    return np.array([ce, kl, br, f1, ap, neg])


def hypothesis_tests(H, obs, dist, decisions=True):
    """obs[model] -> metric vector; dist[model] -> (B, metrics). decisions=False marks results descriptive."""
    B = next(iter(dist.values())).shape[0]
    j = {m: i for i, m in enumerate(METRICS)}
    res = {}
    for h in H:
        a, b = h["A"], h["B"]
        d_obs = obs[a] - obs[b]
        d = dist[a] - dist[b]
        r = {"A": a, "B": b, "diff": dict(zip(METRICS, d_obs.round(6).tolist())),
             "ci95": {m: np.percentile(d[:, j[m]], [2.5, 97.5]).round(6).tolist() for m in METRICS}}
        ce = d[:, j["softCE"]]
        if h["id"] in ("H1", "H5"):
            m = h["margin"]
            pl, ph = (np.sum(ce <= -m) + 1) / (B + 1), (np.sum(ce >= m) + 1) / (B + 1)
            r.update(p_low=pl, p_high=ph, p=max(pl, ph), ci90_softCE=np.percentile(ce, [5, 95]).round(6).tolist())
        elif h["id"] == "H2":
            r["p"] = (np.sum(ce >= 0) + 1) / (B + 1)
        else:  # H3 / H4: intersection-union, A has higher macro-F1 AND higher softCE than B
            p1 = (np.sum(d[:, j["macro-F1"]] <= 0) + 1) / (B + 1)
            p2 = (np.sum(ce <= 0) + 1) / (B + 1)
            r.update(p_macroF1=p1, p_softCE=p2, p=max(p1, p2))
        res[h["id"]] = r
    ids = list(res)
    for i, ph in zip(ids, F.holm([res[i]["p"] for i in ids])):
        res[i]["p_holm"] = float(ph)
        if decisions:
            res[i]["supported_at_0.05"] = bool(ph < 0.05)
        else:
            res[i]["descriptive_only"] = True
    return res


def analyse(pop, seeds, tg, sel, n_boot, rng_seed):
    """seeds[model] -> list of per-seed probability arrays (rows = this population)."""
    n = len(tg.y)
    rng = np.random.default_rng(rng_seed)
    idxs = [rng.integers(0, n, n) for _ in range(n_boot)]
    full = np.arange(n)
    ens = {m: np.mean(ps, 0) for m, ps in seeds.items()}
    obs_s = {m: np.mean([stat_vec(p, tg, full) for p in ps], 0) for m, ps in seeds.items()}
    dist_s = {m: np.array([np.mean([stat_vec(p, tg, i) for p in ps], 0) for i in idxs]) for m, ps in seeds.items()}
    obs_e = {m: stat_vec(p, tg, full) for m, p in ens.items()}
    dist_e = {m: np.array([stat_vec(p, tg, i) for i in idxs]) for m, p in ens.items()}

    def block(obs, dist, decisions):
        ci = {m: {k: np.percentile(dist[m][:, i], [2.5, 97.5]).round(6).tolist() for i, k in enumerate(METRICS)}
              for m in obs}
        return {"models": {m: {"metrics": dict(zip(METRICS, obs[m].round(6).tolist())), "ci95": ci[m]} for m in obs},
                "hypotheses": hypothesis_tests(sel["hypotheses"], obs, dist, decisions)}
    return {"population": pop, "n": n, "n_unique_top": int(tg.uniq.sum()),
            "per_seed_mean (PRIMARY contrast)": block(obs_s, dist_s, True),
            "ensemble (secondary, descriptive)": block(obs_e, dist_e, False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--holdout-manifest"); ap.add_argument("--holdout-features"); ap.add_argument("--out")
    ap.add_argument("--dry-fake-unseen-frac", type=float, default=0.0)
    a = ap.parse_args()
    sel = json.loads(Path(a.selection).read_text())
    if a.dry_run:
        if not (a.holdout_manifest and a.holdout_features and a.out):
            sys.exit("dry run needs --holdout-manifest, --holdout-features, --out")
        man, fdir, out = Path(a.holdout_manifest), Path(a.holdout_features), Path(a.out)
        holdout_files = {}  # stand-in holdout is not registered; registered hashes are still all verified
        label = "DRY RUN — not a holdout result (stand-in data; nothing here is evidence)"
    else:
        if sel["status"].upper().startswith("DRAFT") or not sel.get("HOLDOUT"):
            sys.exit("REFUSING TO RUN: selection is a draft or HOLDOUT is empty")
        if a.dry_fake_unseen_frac:
            sys.exit("--dry-fake-unseen-frac is for dry runs only")
        H = sel["HOLDOUT"]
        man, fdir, out = ROOT / H["manifest"], ROOT / H["features"], ROOT / H["output_dir"]
        holdout_files = {str(ROOT / f): h for f, h in H["sha256"].items()}
        label = f"Phase C holdout evaluation, pre-registered {sel.get('finalised_at_utc')}"
    verify(sel, holdout_files)
    print("all registered hashes verified", flush=True)
    out.mkdir(parents=True, exist_ok=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hold_df = pd.read_csv(man)
    val_df = pd.read_csv(ROOT / "data/artemis_balanced/val.csv")
    hs = load_rows(hold_df, fdir, np.arange(len(hold_df)), dev)
    vs = load_rows(val_df, ROOT / "features_l14/val", np.arange(len(val_df)), dev)
    per_seed, seeds = [], {}
    tg = F.Targets(man)
    for m, spec in sel["models"].items():
        ps = []
        for run in spec["runs"]:
            model, norm = load_run(run, dev)
            dv = np.abs(logits(model, norm, vs) - np.load(ROOT / run / "val_logits.npy")).max()
            if dv >= 1e-4:
                sys.exit(f"REFUSING TO CONTINUE: {run} does not reproduce its val logits (max |diff| {dv:.2e})")
            z = logits(model, norm, hs)
            np.save(out / f"{m}_{Path(run).name}_holdout_logits.npy", z)
            p = softmax(z)
            ps.append(p)
            per_seed.append({"model": m, "run": run, "val_repro_max_abs_diff": float(dv),
                             **dict(zip(METRICS, stat_vec(p, tg, np.arange(len(p)))))})
        seeds[m] = ps
        print(f"{m}: {len(ps)} runs scored", flush=True)
    psd = pd.DataFrame(per_seed)
    psd.to_csv(out / "per_seed.csv", index=False)

    inf = sel["inference"]
    n_boot, rng_seed = inf.get("n_boot", 2000), inf.get("rng_seed", 0)
    results = {"label": label, "selection": str(a.selection), "selection_sha256": sha(a.selection)}
    results["primary"] = analyse("full holdout", seeds, tg, sel, n_boot, rng_seed)
    unseen = ~hold_df["artist_in_43k_train"].astype(bool).to_numpy()
    if a.dry_run and a.dry_fake_unseen_frac:
        arts = hold_df.painting.str.split("_").str[0].str.lower()
        u = np.random.default_rng(1).permutation(arts.unique())
        fake = set(u[:max(1, int(a.dry_fake_unseen_frac * len(u)))])
        unseen = arts.isin(fake).to_numpy()
        results["secondary_note"] = "DRY RUN: artist subset faked (random artists), only to exercise the code path"
    if unseen.any():
        from common import Sub
        results["secondary"] = analyse("artists absent from the 43K train",
                                       {m: [p[unseen] for p in ps] for m, ps in seeds.items()},
                                       Sub(tg, unseen), sel, n_boot, rng_seed)
    else:
        results["secondary"] = "no artworks by artists absent from the 43K train"

    # seed variability (descriptive): per-seed metrics and per-seed paired differences per hypothesis pair
    sv = psd.groupby("model")[METRICS].agg(["mean", "std"])
    results["seed_variability"] = {m: {k: [float(sv.loc[m, (k, "mean")]), float(sv.loc[m, (k, "std")])]
                                       for k in METRICS} for m in sv.index}
    for h in sel["hypotheses"]:
        A = psd[psd.model == h["A"]][METRICS].to_numpy(); B = psd[psd.model == h["B"]][METRICS].to_numpy()
        d = A - B  # runs are listed in seed order 42-46 for every model
        results["seed_variability"][f"{h['id']} per-seed diff (A-B)"] = {
            k: d[:, i].round(6).tolist() for i, k in enumerate(METRICS)}
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))

    md = [f"# {label}", "", f"Selection: {a.selection} (sha256 {results['selection_sha256'][:12]}…)", ""]
    for key in ("primary", "secondary"):
        r = results[key]
        if isinstance(r, str):
            md += [f"## {key}: {r}", ""]; continue
        md += [f"## {key} population: {r['population']} (n = {r['n']}, unique-top {r['n_unique_top']})", ""]
        for cname, c in ((k, v) for k, v in r.items() if isinstance(v, dict)):
            md += [f"### {cname}", "", "| model | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
            for m, v in c["models"].items():
                md.append(f"| {m} | " + " | ".join(f"{v['metrics'][k]:.4f}" for k in METRICS) + " |")
            md += ["", "| hypothesis | A − B softCE | 95% CI | other | p | p (Holm) | supported |",
                   "|---|---|---|---|---|---|---|"]
            for hid, h in c["hypotheses"].items():
                other = (f"90% CI [{h['ci90_softCE'][0]:+.4f}, {h['ci90_softCE'][1]:+.4f}]" if "ci90_softCE" in h else
                         f"Δ macro-F1 {h['diff']['macro-F1']:+.4f} (p {h['p_macroF1']:.4f}), p softCE {h['p_softCE']:.4f}"
                         if "p_macroF1" in h else "")
                dec = ("yes" if h["supported_at_0.05"] else "no") if "supported_at_0.05" in h else "descriptive"
                if key == "secondary" and dec != "descriptive":
                    dec += " (secondary population: descriptive)"
                md.append(f"| {hid} ({h['A']} − {h['B']}) | {h['diff']['softCE']:+.4f} | [{h['ci95']['softCE'][0]:+.4f}, "
                          f"{h['ci95']['softCE'][1]:+.4f}] | {other} | {h['p']:.4f} | {h['p_holm']:.4f} | {dec} |")
            md.append("")
    md += ["## Seed variability (per-seed metrics, mean ± SD over 5 seeds; descriptive)", "",
           "| model | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for m in sel["models"]:
        v = results["seed_variability"][m]
        md.append(f"| {m} | " + " | ".join(f"{v[k][0]:.4f} ± {v[k][1]:.4f}" for k in METRICS) + " |")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
