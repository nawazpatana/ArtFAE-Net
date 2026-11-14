"""Write runs/final_selection_v3_DRAFT3.txt: the third Phase C pre-registration draft. Evaluates nothing.

Reads only val logits / val targets (val means, probe grid) and hashes files. Refuses to run if a final
runs/final_selection_v3.txt exists; refuses to overwrite DRAFT3 unless --replace (DRAFT3 only, used to record the
evaluator dry run after it has been performed).
"""
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np

from common import OUT, ROOT, F
from s1_eval import row

SEEDS = [42, 43, 44, 45, 46]
DRAFT = ROOT / "runs/final_selection_v3_DRAFT3.txt"
DRY_RUN_LOG = OUT / "logs/eval_holdout_dryrun3.log"

MODELS = {  # key: (label, run directory pattern relative to ROOT, checkpoint selection)
    "R1": ("L/14 R1 (softCE-selected)", "runs/revision_B/step1/R1_s{s}", "min val softCE"),
    "NO_ATTR": ("L/14 − both attribute branches", "runs/revision_B/step1/no_attr_s{s}", "min val softCE"),
    "AGF_ONLY": ("L/14 AGF-only (− both attribute branches, − ECSA)", "runs/revision_B/step1/agf_only_s{s}", "min val softCE"),
    "NO_ECSA": ("L/14 − ECSA", "runs/revision_B/step1/no_ecsa_s{s}", "min val softCE"),
    "WEIGHTED": ("L/14 R1 + class-weighted soft CE (power 0.25, [0.5, 2.0])", "runs/revision_B/step1/weighted_s{s}", "min val softCE"),
    "PROBE": ("L/14 linear probe, own budget (lr 1e-3, 100 epochs, patience 10)", "runs/revision_B/step1/probe100_lr1e-3_s{s}", "min val softCE"),
    "MLP": ("L/14 MLP on global embedding (768-722-9)", "runs/revision_B/step1/mlp_s{s}", "min val softCE"),
    "R3": ("L/14 R3 (lambda 0.3, sampler), retrained with softCE selection", "runs/revision_B/step1/R3_s{s}", "min val softCE"),
    "R3_MF1": ("L/14 R3, existing macro-F1-selected runs (descriptive only; different protocol)", "runs/L14_R3_lam0.3_s{s}",
               "max val macro-F1 (existing runs)"),
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
    "runs/revision_B/code/make_selection_v3_draft2.py", "runs/revision_B/code/make_selection_v3_draft3.py",
    "runs/revision_B/code/eval_holdout.py",
    "runs/revision_B/code/build_fresh_holdout.py", "runs/revision_B/code/phash.py",
    "runs/revision_B/code/extract_holdout_features.sh", "build_balanced_artemis.py",
    "runs/revision_B/code/v2_inclusion_rates.py",
    "scripts/final_eval_v2.py", "scripts/run_l14_ladder.sh", "scripts/run_l14_ladder_s45_46.sh",
    "scripts/run_extract_l14.sh", "artfae_balance/__init__.py", "artfae_balance/train.py",
    "artfae_balance/train_backbone.py", "artfae_balance/model.py", "artfae_balance/heads.py",
    "artfae_balance/losses.py", "artfae_balance/data.py", "artfae_balance/evaluate.py",
    "artfae_balance/extract_features.py", "artfae_balance/extract_features_backbone.py",
    "src/artfae/models/network.py", "src/artfae/features/aesthetic.py", "src/artfae/training/losses.py",
]
DATA = ["data/artemis_balanced/train.csv", "data/artemis_balanced/val.csv",
        "runs/revision_B/duplicates/classification.csv", "runs/revision_A/t6_near_duplicate_cross_split.csv",
        "Data/artemis_full_v1_20261031/duplicate_groups.csv", "Data/artemis_full_v1_20261031/excluded_records.csv",
        "artemis_official_data/official_data/artemis_dataset_release_v0.csv",
        "runs/revision_B/holdout_prep/reference_phash.csv", "runs/revision_B/holdout_prep/phash_validation.csv"] + \
       [f"features_l14/{s}/{k}.npy" for s in ("train", "val") for k in ("global", "attributes", "attr_text", "patches")]


def sha(p):
    h = hashlib.sha256()
    with open(ROOT / p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if (ROOT / "runs/final_selection_v3.txt").exists():
        sys.exit("a final runs/final_selection_v3.txt exists; drafts are closed")
    if DRAFT.exists() and "--replace" not in sys.argv:
        sys.exit(f"{DRAFT} exists; pass --replace to regenerate it")
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
    params["R3_MF1"] = 763410  # runs/revision_A/t5_params.md (artfae_balance config has no parameter field)
    probe_grid = {}
    for lr in ("1e-2", "3e-3", "1e-3", "3e-4", "1e-4"):
        probe_grid[lr] = round(float(np.mean([min(json.loads((ROOT / f"runs/revision_B/step1/probe100_lr{lr}_s{s}/history.json")
                                                         .read_text()), key=lambda h: h["sel_soft_ce"])["sel_soft_ce"]
                                              for s in SEEDS])), 6)
    repro = float(np.abs(np.load(OUT / "step1_repro/no_num_s42/val_logits.npy")
                         - np.load(OUT / "step1/no_num_s42/val_logits.npy")).max())
    used = sorted(__import__("build_fresh_holdout").used_keys())
    used_sha = hashlib.sha256("\n".join(used).encode()).hexdigest()

    sel = {
        "status": "DRAFT 3 — not pre-registered, not finalised. Nothing has been evaluated on any holdout for Phase C.",
        "drafted_at_utc": datetime.now(timezone.utc).isoformat(),
        "HOLDOUT": "",
        "HOLDOUT_note": "Empty until the full WikiArt images exist and the construction below has been executed. At "
                        "finalisation it becomes {manifest, features, output_dir, sha256: {holdout.csv, excluded.csv, "
                        "duplicates/classification.csv, every holdout feature .npy}} (keys read by eval_holdout.py).",
        "HOLDOUT_construction": {
            "decision": "fresh holdout: ArtEmis artworks whose images are in the full WikiArt set but that were never "
                        "used (user decision, 2026-11-03)",
            "code": "runs/revision_B/code/build_fresh_holdout.py (stages candidates, finalize); "
                    "runs/revision_B/code/phash.py; runs/revision_B/code/extract_holdout_features.sh",
            "labels": "official ArtEmis release v0 (artemis_dataset_release_v0.csv, sha256 in data_sha256), aggregated "
                      "with build_balanced_artemis.aggregate (9 emotions, 'something else' -> something_else, "
                      "soft targets, present_*, unique_top isclose atol 1e-5)",
            "image_matching": "build_balanced_artemis.index_images / attach_images (lower-cased stem; style folder, "
                              "then style-free)",
            "eligibility": "painting stem not among already-used artworks: the 49,011 clean_artemis images, every "
                           "manifest ever used (43K train/val/test, v1 20K, v2 20K, paper 14K, locked20k, 30k) and the "
                           "13 cleaning-QC invalid images; painting stem maps to exactly one ArtEmis artwork_id; image "
                           "fully decodes",
            "used_keys": {"n": len(used), "sha256_of_sorted_newline_joined_keys": used_sha,
                          "unused_artemis_artworks_now": "30,479 (upper bound on the holdout before image matching)"},
            "duplicate_exclusion": "pinned 64-bit DCT pHash (phash.py); candidates = Hamming <= 8 against every "
                                   "reference image (43K train, evaluation val/test, all of v2 20K; "
                                   "holdout_prep/reference_phash.csv); one contact sheet per pair; each pair classified "
                                   "visually as 'same artwork' / 'same artwork, different reproduction' / 'uncertain' "
                                   "/ 'different' (revision_B Step 0 protocol). A pair classified 'uncertain' counts as "
                                   "a duplicate: holdout artworks in every pair not classified 'different' are "
                                   "excluded. Each pair's class and a one-line reason are recorded in the contact-sheet "
                                   "log <prep>/duplicates/classification.csv (columns classification, note); finalize "
                                   "refuses while any pair lacks a class or a reason.",
            "phash_threshold_rationale": "pinned pHash reproduces 184/191 recorded QC pairs at <= 4 and 190/191 at <= 8 "
                                         "(holdout_prep/phash_validation.csv); dry run on val as a stand-in recovered "
                                         "16/16 confirmed val–train duplicates",
            "populations": {"primary": "full holdout",
                            "secondary": "holdout artworks whose artist (painting slug before the first underscore) "
                                         "is absent from the 43K train (artist generalisation); descriptive, own "
                                         "bootstrap, not part of the Holm family"},
            "features": "extract_holdout_features.sh: artfae_balance.extract_features_backbone, "
                        "openai/clip-vit-large-patch14-336 @ ce19dc912ca5cd21c8a653c79e251e808ccabcd1, batch 64, "
                        "--pool-grid 7, whole-image attributes + describe() text — identical to features_l14",
            "use": "nothing is trained, selected or tuned on the holdout; it is read once by eval_holdout.py",
        },
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
                       "(tied included), p = softmax(logits) of one seed; a model's value is the mean of its 5 "
                       "per-seed values (see inference.primary_contrast)",
            "secondary": ["KL(y || p)", "Brier = Σ_c (p_c − y_c)^2", "macro-F1 (9 classes, unique-top artworks, "
                          "isclose atol 1e-5)", "mean anger+fear presence AP (all artworks)",
                          "negative-sentiment F1 (artworks with a unique top sentiment; final_eval_v2 definition)"],
            "secondary_role": "reported with bootstrap CIs for every hypothesis pair and every model; not tested, "
                              "not multiplicity-corrected, except macro-F1 inside the compound hypotheses H3/H4",
            "definitions_source": "scripts/final_eval_v2.py (metrics, boot_stats, sentiment_f1, Targets)",
        },
        "inference": {
            "n_boot": 2000, "rng_seed": 0,
            "evaluator": "runs/revision_B/code/eval_holdout.py (refuses on any hash mismatch, draft status, empty "
                         "HOLDOUT, or a checkpoint that does not reproduce its val logits; create-once output)",
            "bootstrap": "paired artwork bootstrap on the (deduplicated) holdout, 2,000 resamples, "
                         "rng = numpy.random.default_rng(0), each resample rng.integers(0, n, n); the same artwork "
                         "draws for every model, every seed, every hypothesis and every metric",
            "primary_contrast": "per-seed mean (user decision 2026-11-03, option (b)): in each resample, every metric "
                                "is computed separately for each of a model's 5 seeds on the resampled artworks and "
                                "averaged over the seeds; H1–H5 are tested on differences of these per-seed means "
                                "between models; Holm decisions use only this contrast",
            "secondary_contrast": "5-seed ensemble (seed-averaged probabilities), same draws and tests; descriptive "
                                  "only, no decisions. Reason for the change: ensembling lowers softCE by ~0.0046 "
                                  "for R1 but ~0.0000 for the linear probe (identical seeds), so ensemble contrasts "
                                  "with a 0.005 margin would largely test seed diversity, not architecture",
            "point_estimate": "primary: mean over the 5 seeds of the metric on the full deduplicated holdout; "
                              "secondary: metric of the ensemble",
            "intervals": "95% percentile CI for every difference; 90% percentile CI additionally for H1/H5",
            "multiplicity": "Holm across the five hypothesis p-values H1–H5 (primary family, alpha 0.05). For H3/H4 "
                            "the hypothesis p-value is the intersection-union p = max(p_macroF1, p_softCE).",
            "seed_variability": "per-seed metrics (each seed's own probabilities) reported separately as mean ± SD "
                                "over the 5 seeds for every model and as per-seed paired differences for every "
                                "hypothesis pair; descriptive only, not used for any decision",
        },
        "duplicate_exclusion": {
            "rule": "fresh holdout: see HOLDOUT_construction.duplicate_exclusion (pairs classified 'same artwork', "
                    "'same artwork, different reproduction' or 'uncertain' are excluded for every model; class and "
                    "one-line reason per pair in the contact-sheet log)",
            "limitation": "pHash candidates (pinned pHash, Hamming <= 8) are not exhaustive; undetected "
                          "reproductions may remain",
        },
        "val_mean_used_to_form_hypotheses": val_mean,
        "probe_lr_grid_mean_selected_val_softCE": probe_grid,
        "probe_lr_choice": f"{min(probe_grid, key=probe_grid.get)} (lowest mean selected val softCE; interior point of the grid)",
        "reproducibility_check": {"run": "no_num seed 42 retrained with the current train_abl.py "
                                         "(runs/revision_B/step1_repro/no_num_s42)",
                                  "max_abs_val_logit_diff_vs_original": repro, "criterion": "< 1e-5",
                                  "passed": repro < 1e-5},
        "decisions_recorded": [
            "2026-11-03: fresh holdout of never-used ArtEmis artworks (DRAFT2).",
            "2026-11-03: option (b) — primary contrast = per-seed mean inside each bootstrap resample; ensemble "
            "contrasts secondary and descriptive (DRAFT3).",
            "2026-11-03: a pHash pair classified 'uncertain' counts as a duplicate; class + one-line reason recorded "
            "per pair (DRAFT3).",
        ],
        "r3_note": "R3 retrained with softCE selection reproduces the existing R3 per-epoch val softCE exactly "
                   "(seed 42); softCE selection picks epochs 1-2, because R3's val softCE rises after epoch 1-2.",
        "evaluator_dry_run": ({"log": str(DRY_RUN_LOG.relative_to(ROOT)), "sha256": sha(DRY_RUN_LOG.relative_to(ROOT)),
                               "stand_in": "evaluation val minus its 16 confirmed duplicates (2,984 artworks), built "
                                           "with build_fresh_holdout.py --pretend-unused val (labels mechanical from "
                                           "revision_B Step 0, each with a one-line reason); secondary population "
                                           "faked (10% random artists) to exercise the code path",
                               "refusal_paths_tested": ["draft selection / empty HOLDOUT", "tampered code and "
                                                        "checkpoint hash", "existing output directory",
                                                        "builder: unclassified pair", "builder: pair without reason"],
                               "builder_uncertain_rule_tested": "relabelling one 'different' pair as 'uncertain' "
                                                                "excluded one more artwork (2,984 -> 2,983)"}
                              if DRY_RUN_LOG.exists() else "not yet performed"),
        "open_items_before_finalising": [
            "Full WikiArt images: run build_fresh_holdout.py candidates --images <dir>.",
            "Visually classify every pHash candidate pair (expected order of magnitude: ~600 pairs for ~30K "
            "artworks, from the val dry run's 61 pairs per 3,000), then run finalize.",
            "Extract holdout features with extract_holdout_features.sh.",
            "Fill HOLDOUT {manifest, features, output_dir, sha256} and set status to final in "
            "runs/final_selection_v3.txt (not this draft); record finalised_at_utc.",
            "Report the holdout's composition (size, class distribution, artist overlap) when it exists; the "
            "hypotheses do not depend on it, but power for H1/H5 (margin 0.005) does.",
            "Prior test exposure: the old test split is not used in Phase C; its earlier uses remain to be stated "
            "in the manuscript.",
            "R3_MF1 (existing macro-F1-selected R3) is descriptive only and in no hypothesis.",
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
