"""Write runs/final_selection_v3_DRAFT.txt: the Phase C pre-registration draft. Evaluates nothing on any holdout.

Reads only val logits / val targets (to record the val means the hypotheses were formed from) and hashes files.
Refuses to overwrite an existing draft and refuses to run if a non-draft runs/final_selection_v3.txt exists.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np

from common import OUT, ROOT, F
from s1_eval import row

SEEDS = [42, 43, 44, 45, 46]
DRAFT = ROOT / "runs/final_selection_v3_DRAFT.txt"

MODELS = {  # key: (label, run directory pattern relative to ROOT, checkpoint selection)
    "R1": ("L/14 R1 (softCE-selected)", "runs/revision_B/step1/R1_s{s}", "min val softCE"),
    "NO_ATTR": ("L/14 − both attribute branches", "runs/revision_B/step1/no_attr_s{s}", "min val softCE"),
    "AGF_ONLY": ("L/14 AGF-only (− both attribute branches, − ECSA)", "runs/revision_B/step1/agf_only_s{s}", "min val softCE"),
    "NO_ECSA": ("L/14 − ECSA", "runs/revision_B/step1/no_ecsa_s{s}", "min val softCE"),
    "WEIGHTED": ("L/14 R1 + class-weighted soft CE (power 0.25, [0.5, 2.0])", "runs/revision_B/step1/weighted_s{s}", "min val softCE"),
    "PROBE": ("L/14 linear probe, own budget (lr 1e-3, 100 epochs, patience 10)", "runs/revision_B/step1/probe100_lr1e-3_s{s}", "min val softCE"),
    "MLP": ("L/14 MLP on global embedding (768-722-9)", "runs/revision_B/step1/mlp_s{s}", "min val softCE"),
    "R3": ("L/14 R3 (lambda 0.3, sampler), existing runs", "runs/L14_R3_lam0.3_s{s}", "max val macro-F1 (existing runs; NOT softCE, see open_items)"),
}

HYPOTHESES = [
    {"id": "H1", "type": "equivalence (TOST)", "A": "NO_ATTR", "B": "R1", "metric": "softCE", "margin": 0.005,
     "claim": "|softCE(−both attribute branches) − softCE(R1)| < 0.005",
     "test": "two one-sided bootstrap tests on d = A − B: p_low = (#{d* <= −0.005} + 1)/(B + 1) for H0: d <= −0.005, "
             "p_high = (#{d* >= +0.005} + 1)/(B + 1) for H0: d >= +0.005; p = max(p_low, p_high). Equivalence "
             "also reported as the 90% percentile CI lying inside (−0.005, +0.005)."},
    {"id": "H2", "type": "superiority, one-sided", "A": "AGF_ONLY", "B": "R1", "metric": "softCE",
     "claim": "softCE(AGF-only) < softCE(R1)",
     "test": "one-sided bootstrap p = (#{d* >= 0} + 1)/(B + 1) for d = A − B (H0: d >= 0)."},
    {"id": "H3", "type": "compound (intersection-union)", "A": "R1", "B": "NO_ECSA", "metric": "macro-F1 AND softCE",
     "claim": "R1 has higher macro-F1 than − ECSA AND higher (worse) softCE than − ECSA",
     "test": "component 1: d_F1 = macroF1(A) − macroF1(B), one-sided p1 = (#{d* <= 0} + 1)/(B + 1); component 2: "
             "d_CE = softCE(A) − softCE(B), one-sided p2 = (#{d* <= 0} + 1)/(B + 1); intersection-union test, "
             "p = max(p1, p2) (both components must be rejected). Both components use the same bootstrap draws."},
    {"id": "H4", "type": "compound (intersection-union)", "A": "WEIGHTED", "B": "R1", "metric": "macro-F1 AND softCE",
     "claim": "class-weighted R1 has higher macro-F1 AND higher (worse) softCE than R1",
     "test": "as H3 with A = class-weighted R1, B = R1; p = max(p1, p2)."},
    {"id": "H5", "type": "equivalence (TOST)", "A": "PROBE", "B": "R1", "metric": "softCE", "margin": 0.005,
     "claim": "|softCE(linear probe, own budget) − softCE(R1)| < 0.005", "test": "as H1."},
]

CODE = [
    "runs/revision_B/code/train_abl.py", "runs/revision_B/code/common.py", "runs/revision_B/code/s1_eval.py",
    "runs/revision_B/code/s1b_followup.py", "runs/revision_B/code/contact_sheets.py",
    "runs/revision_B/code/s0_dedup_sensitivity.py", "runs/revision_B/code/make_selection_v3_draft.py",
    "scripts/final_eval_v2.py", "scripts/run_l14_ladder.sh", "scripts/run_l14_ladder_s45_46.sh",
    "scripts/run_extract_l14.sh", "artfae_balance/__init__.py", "artfae_balance/train.py",
    "artfae_balance/train_backbone.py", "artfae_balance/model.py", "artfae_balance/heads.py",
    "artfae_balance/losses.py", "artfae_balance/data.py", "artfae_balance/evaluate.py",
    "artfae_balance/extract_features.py", "artfae_balance/extract_features_backbone.py",
    "src/artfae/models/network.py", "src/artfae/features/aesthetic.py", "src/artfae/training/losses.py",
]
DATA = ["data/artemis_balanced/train.csv", "data/artemis_balanced/val.csv",
        "runs/revision_B/duplicates/classification.csv", "runs/revision_A/t6_near_duplicate_cross_split.csv",
        "Data/artemis_full_v1_20261031/duplicate_groups.csv"] + \
       [f"features_l14/{s}/{k}.npy" for s in ("train", "val") for k in ("global", "attributes", "attr_text", "patches")]


def sha(p):
    h = hashlib.sha256()
    with open(ROOT / p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if DRAFT.exists() or (ROOT / "runs/final_selection_v3.txt").exists():
        sys.exit(f"{DRAFT} (or a final v3) already exists; not overwriting")
    tv = F.Targets(F.VAL_CSV)
    ckpt, val_mean = {}, {}
    for k, (lab, pat, _) in MODELS.items():
        rows = []
        for s in SEEDS:
            d = pat.format(s=s)
            for f in ("best.pt", "val_logits.npy", "config.json"):
                ckpt[f"{d}/{f}"] = sha(f"{d}/{f}")
            rows.append(row(np.load(ROOT / d / "val_logits.npy").astype(np.float64), tv))
        val_mean[k] = {m: round(float(np.mean([r[m] for r in rows])), 6) for m in
                       ("softCE", "KL", "Brier", "macro-F1", "mean anger+fear AP", "neg-sentiment F1")}
        val_mean[k].update({f"{m} seed SD": round(float(np.std([r[m] for r in rows], ddof=1)), 6)
                            for m in ("softCE", "macro-F1")})
    params = {k: json.loads((ROOT / pat.format(s=42) / "config.json").read_text()).get("trainable_parameters")
              for k, (_, pat, _) in MODELS.items()}
    params["R3"] = 763410  # runs/revision_A/t5_params.md (artfae_balance config has no parameter field)

    sel = {
        "status": "DRAFT — not pre-registered, not finalised. Nothing has been evaluated on any holdout for Phase C.",
        "drafted_at_utc": datetime.now(timezone.utc).isoformat(),
        "HOLDOUT": "",
        "HOLDOUT_note": "Left empty on purpose; to be confirmed by the user. Must name the manifest, its sha256, and "
                        "the duplicate-exclusion list applied to it before this file can be finalised.",
        "task": "distribution prediction (9-way human vote distribution) is the primary task",
        "backbone_features": "CLIP ViT-L/14@336 (revision ce19dc91…), features_l14 (global 768, attributes 28, "
                             "attribute text 768, 7x7-pooled patches 1024); frozen; heads trained on cached features",
        "seeds": SEEDS,
        "logits": "raw logits for every model (no prior correction, no temperature) unless stated",
        "models": {k: {"label": lab, "runs": [pat.format(s=s) for s in SEEDS], "checkpoint_selection": cs,
                       "trainable_parameters": params[k]} for k, (lab, pat, cs) in MODELS.items()},
        "hypotheses": HYPOTHESES,
        "metrics": {
            "primary": "softCE = mean over holdout artworks of −Σ_c y_c log p_c (T = 1, eps 1e-12), all artworks "
                       "(tied included), on seed-averaged probabilities p = mean over the 5 seeds of softmax(logits)",
            "secondary": ["KL(y || p)", "Brier = Σ_c (p_c − y_c)^2", "macro-F1 (9 classes, unique-top artworks, "
                          "isclose atol 1e-5)", "mean anger+fear presence AP (all artworks)",
                          "negative-sentiment F1 (artworks with a unique top sentiment; final_eval_v2 definition)"],
            "secondary_role": "reported with bootstrap CIs for every hypothesis pair and every model; not tested, "
                              "not multiplicity-corrected, except macro-F1 inside the compound hypotheses H3/H4",
            "definitions_source": "scripts/final_eval_v2.py (metrics, boot_stats, sentiment_f1, Targets)",
        },
        "inference": {
            "bootstrap": "paired artwork bootstrap on the (deduplicated) holdout, 2,000 resamples, "
                         "rng = numpy.random.default_rng(0), each resample rng.integers(0, n, n); the same draws for "
                         "every model, hypothesis and metric; seed-averaged probabilities",
            "point_estimate": "metric on the full deduplicated holdout with seed-averaged probabilities",
            "intervals": "95% percentile CI for every difference; 90% percentile CI additionally for H1/H5",
            "multiplicity": "Holm across the five hypothesis p-values H1–H5 (primary family, alpha 0.05). For H3/H4 "
                            "the hypothesis p-value is the intersection-union p = max(p_macroF1, p_softCE).",
            "seed_variability": "per-seed metrics (each seed's own probabilities) reported separately as mean ± SD "
                                "over the 5 seeds for every model and as per-seed paired differences for every "
                                "hypothesis pair; descriptive only, not used for any decision",
        },
        "duplicate_exclusion": {
            "rule": "holdout artworks classified 'same artwork' or 'same artwork, different reproduction' against any "
                    "artwork in the 43K training set (data/artemis_balanced/train.csv) are excluded from the holdout "
                    "for every model",
            "source": "runs/revision_B/duplicates/classification.csv (visual classification of the pHash candidates, "
                      "Hamming <= 4, from Data/artemis_full_v1_20261031/duplicate_groups.csv)",
            "if_holdout_is_current_test": "15 artworks excluded (2,999 → 2,984): " + ", ".join(sorted(
                __import__("pandas").read_csv(OUT / "duplicates/classification.csv")
                .query("classification != 'different' and eval_side_splits == 'test'").eval_side.unique())),
            "if_holdout_is_new": "the pHash candidate search and visual classification must be run against the 43K "
                                 "train before finalisation, and the resulting exclusion list hashed here",
            "limitation": "pHash candidates (Hamming <= 4) are not exhaustive; undetected reproductions may remain",
        },
        "val_mean_used_to_form_hypotheses": val_mean,
        "open_items_before_finalising": [
            "HOLDOUT must be named and hashed (see HOLDOUT_note).",
            "If HOLDOUT is the current test split, this would be its fourth use (final_eval, final_eval_v2, revision_A "
            "analyses, revision_B Step 0b sensitivity). The revision_B hypotheses were formed on val only, but the "
            "manuscript must state the prior test exposure.",
            "L/14 R3 checkpoints were selected by val macro-F1 (existing runs), unlike all other models (val softCE). "
            "Either keep it as descriptive context only (it is in no hypothesis) or retrain R3 with softCE selection "
            "before finalising.",
            "The holdout evaluation script does not exist yet; its sha256 must be added to code_sha256 before "
            "finalisation, and it must refuse to run unless every hash below still matches.",
            "Linear-probe lr 1e-3 was the largest value in the grid chosen on val; the probe may be slightly "
            "under-tuned, which biases H5 against equivalence only if the probe is worse.",
            "train_abl.py was extended after the Step 1 runs (follow-up conditions only; earlier code paths "
            "unchanged); the hash recorded is of the current file, which produced the probe/MLP/AGF-only runs.",
        ],
        "code_sha256": {f: sha(f) for f in CODE},
        "data_sha256": {f: sha(f) for f in DATA},
        "checkpoint_sha256": ckpt,
    }
    DRAFT.write_text(json.dumps(sel, indent=1, ensure_ascii=False) + "\n")
    print(f"written {DRAFT}: {len(ckpt)} checkpoint files, {len(CODE)} code files, {len(DATA)} data files hashed")
    for k, v in val_mean.items():
        print(f"{k:9s} params {params[k]:>9,}  val softCE {v['softCE']:.4f}  macro-F1 {v['macro-F1']:.4f}")


if __name__ == "__main__":
    main()
