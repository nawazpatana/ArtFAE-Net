# ArtFAE-Net — project index

**Do not move or rename anything listed here.** `runs/final_selection*.txt` record exact relative paths and SHA-256 hashes of checkpoints, logits, manifests, features, code and holdout files. Moving a file breaks its hash record.

Paper: `paper/main.tex`, built with `tectonic` (TeX packages cached in `~/.cache/Tectonic`). Tables come from `paper/make_tables.py`, figures 1–5 from `paper/make_figures.py`, figure 6 from `paper/posthoc/attention_figure.py`, and `\val{}` numbers from `paper/check_claims.py` (which writes `paper/numbers.tex`). Python: `/workspace/.venv` (use `HF_HUB_OFFLINE=1`; CLIP weights are in `~/.cache/huggingface`).

## What each folder supports

| Path | Contents | Supports |
|---|---|---|
| `paper/` | LaTeX sources, sections, tables, figures, claims ledger, reference checks | the paper |
| `runs/final_selection_v3.txt` | pre-registered plan: models, seeds, holdout, all hashes | §5.3 Pre-registration |
| `runs/final_selection_v3_DRAFT*.txt` | the three drafts that led to v3 | §5.3 (audit trail) |
| `runs/final_eval_v3/`, `runs/final_eval_v3_posthoc/` | pre-registered holdout evaluation, and the post-hoc analyses | §6.3 holdout, §6.4 ties, §6.5 per-class/anger |
| `runs/final_selection.txt`, `_v2.txt`, `runs/final_eval/`, `runs/final_eval_v2/` | the three earlier evaluations on the reused test set (superseded) | §5.2 prior exposure, §6.1 development history |
| `runs/val_table_s42.md`, `runs/val_prior_table.md`, `runs/R*_s4*/`, `runs/L14_*/`, `runs/R0n_paper14k_*/`, `runs/m1_paper/` | per-seed training runs (B/32 and L/14 ladders) and validation tables | §6.2 validation ablations, §4.7 baselines |
| `runs/revision_A/` | supervisor revision phase A: subset provenance (`subset_construction.md`), parameter counts, near-duplicates | §3.2 splits, appendix |
| `runs/revision_B/` | phase B: dedup sensitivity, ablations, artist holdout, fresh-holdout build code, features and logs | §3.3 fresh holdout, §6.2, §6.3 |
| `runs/app_diagnosis/` | diagnosis of the demo app labelling dark paintings as anger (`REPORT.md`); `app_before/` holds the old app assets for rollback | demo app |
| `src/`, `artfae_balance/`, `scripts/`, `build_balanced_artemis.py`, `pyproject.toml` | model, training, feature extraction, dataset balancing | §4 Method |
| `artemis_official_data/` | official ArtEmis release v0 annotations | §3.1 |
| `archive/images/` (= `clean_artemis/images/`, a symlink) | 49,011 cleaned WikiArt images used by every split | §3 |
| `/workspace/wikiart_full/wikiart/` | full WikiArt (81,446 files), source of the fresh holdout | §3.3 |
| `Data/artemis_full_v1_20261031/` | cleaned annotations, quality flags, duplicate groups, exclusions (images are in `clean_artemis/images/`). **Missing since before 2026-11-08 00:00 UTC; being restored from the owner's laptop** (see 2026-11-12 below) | §3.4 known data issues |
| `data/paper_split/`, `data/artemis_balanced/` | the v2 20K val/test split used for every reported val/test number (`paper_split` image paths are relative to the v2 package) | §3.2 |
| `data/paper14k/`, `features_paper14k/` | paper-14K training set and its features (R0n) | §4.7 baselines |
| `data/artemis_balanced_30k/`, `data/artemis_balanced_locked20k/`, `data/{train,val,test}_artworks.csv`, `data/artworks_20k.csv` | superseded and v1 manifests, kept as "already used" lists for holdout eligibility | §3.3 |
| `data/images_unique_backup/` | the 4 v1 images not in `clean_artemis/images/` (see note below) | provenance |
| `/workspace/datasets/ArtEmis-20k-v2/ArtEmis-20k-GPU/` | user-reviewed v2 20K package (images and CSVs), extracted from `uploads/ArtEmis-20k.zip` | §3.2 |
| `uploads/ArtEmis-20k.zip` | the original v2 upload (2026-10-26), byte-identical CSVs to the extracted package | §3.2 provenance |
| `features/` (B/32), `features_l14/` (L/14@336), `features_l14_fresh_holdout/` | cached frozen CLIP features | §4.1, §6 |
| `app/`, `app_bundle.tar.gz` | Streamlit demo (R1 default, R3 option) and its packaged bundle (the latest bundle) | demo |
| `artfae_demo/` | older installed demo copy; its `.venv` runs the app and the tests | demo, tests |
| `results/` | October 2026 phase-0 experiments (v1 B1–B5 baselines, smoke tests); `leakage_audit_20261102` is cited by the paper; `resnet50_frozen_cpu_*` and `clip_b32_full` are read by `runs/revision_A/code/t5_params.py` | history; §3 leakage |
| `docs/`, root `*.md`, `notebooks/`, `configs/`, `tests/`, `logs/` | earlier guides, notebooks, configs, unit tests, extraction and training logs | history, reproducibility |

## Known hash mismatches in the draft selection files

Verifying every hash in `runs/final_selection*.txt` gives 858 of 861 entries matching. `final_selection.txt`, `_v2.txt` and `_v3.txt` match 100%. The 3 mismatches are all in superseded drafts and are expected: the files were edited after that draft was written and were re-hashed in later versions.

| Draft | File | Re-hashed (current hash) in |
|---|---|---|
| `final_selection_v3_DRAFT.txt` | `runs/revision_B/code/train_abl.py` (edited 2026-11-03 09:54 UTC) | DRAFT2, DRAFT3, v3 |
| `final_selection_v3_DRAFT2.txt` | `runs/revision_B/code/eval_holdout.py` (edited 2026-11-03 10:48 UTC) | DRAFT3, v3 |
| `final_selection_v3_DRAFT2.txt` | `runs/revision_B/code/build_fresh_holdout.py` (edited 2026-11-03 10:49 UTC) | DRAFT3, v3 |

Check rule: 100% for final, v2 and v3; the drafts must show exactly these 3 mismatches and no others.

## Cleanup log

**2026-11-06 — quarantine (not deleted).** 15 items were moved to `/workspace/_to_delete/` with their relative paths kept; every move is in `/workspace/_to_delete/MOVES.log` (reverse each `MOVED` line to restore). Moved:
- `Data/artemis_full_v1_20261031/images/`: a byte-identical duplicate of `archive/images/`, plus one `.rsync-partial` file
- `/workspace/wikiart_full/.wikiart.zip.hj7JDR`: a truncated partial download
- `features_locked20k/`, `artfae_app/` (outdated demo install), `data/images/`, `data/.sharing_test`
- `__pycache__/` folders outside protected code folders
- `~/.cache/pip` (moved to `/workspace/_to_delete/home_dev_cache/pip`)

After the move, these checks passed:
- hash rule above
- `check_claims.py` (315/315 claims)
- tables (identical) and figures (identical apart from PDF creation dates), regenerated into a temp folder
- `main.pdf` compiled into a temp folder: identical except the `\today` date

**2026-11-06 — `data/images/` note.** `data/images/` (the v1 seed-42 20K image copy, 20,000 files) is now in `/workspace/_to_delete/ArtFAE-Net/data/images/`. 19,996 of its files are byte-identical to `clean_artemis/images/`. The other 4 are copied, hash-verified, to `data/images_unique_backup/`, and are also in the v2 package:
- `franz-marc_sleeping-animals-1913`
- `henri-rousseau_view-of-the-bridge-at-sevres-and-the-hills-at-clamart-st-cloud-and-bellevue-1908`
- `juan-gris_glass-and-carafe-1917`
- `pablo-picasso_untitled-1920-2`, the v2 test image dropped from evaluation, hence 2,999 test artworks

`runs/revision_A/subset_construction.md` still refers to `data/images`. It was left unedited on purpose; read that reference as the location before 2026-11-06. `README.md` and `TRAINING_GUIDE.md` now point to `clean_artemis/images/`.

**2026-11-08 — disposable artifact cleanup.** Deleted 13 project Python `__pycache__/` directories and `paper/main.log` / `paper/main.blg` (52 files, 463,865 bytes). Retained the demo virtual environment, datasets, feature caches, run histories, packaged releases, paper sources/PDF, and empty output placeholders. Added ignore rules for Python/tool/notebook caches. No research or source artifacts were removed.

**2026-11-12 — audit follow-up.**
- **`Data/artemis_full_v1_20261031/` is missing.** It was not in the quarantine (that held only its `images/`), and no ZFS snapshot, git history or copy has it. Because of this, `final_selection_v3.txt` verifies 193/195: `duplicate_groups.csv` and `excluded_records.csv` are absent. The drafts miss them too, besides the 3 known mismatches. The owner is restoring the folder from a laptop. Until then, `runs/revision_A/code/t6_subset_stats.py`, `runs/revision_B/code/build_fresh_holdout.py` and `phash.py` cannot be re-run. `final_selection.txt` and `_v2.txt` remain 100%.
- **`/workspace/_to_delete/` was deleted** between 00:00 and 00:21 UTC on 2026-11-12, not by the audit session. `MOVES.log` went with it. The full quarantine is still in the ZFS snapshots up to `auto-2026-11-12-0000` (`/workspace/.zfs/snapshot/<name>/_to_delete/`).
- `paper/check_claims.py`: the bare-number prose scan now skips `sections/10_backmatter.tex` and `sections/12_biographies.tex` (`PROSE_EXCLUDE`); every other section is still scanned.
- `.release/` (not otherwise indexed) holds two Git-LFS release repos for `github.com/nawazpatana/ArtAFE`. `ArtFAE-Net/` (branch `release/artfae-raw-lfs-20261108`) was pushed on 2026-11-08 12:06 UTC through the git-lfs pre-push hook; this machine has no GitHub credentials to confirm it from the remote. `ArtFAE-Net-archive-release-20261108/` was never pushed. All 49,039 files in its tar parts have byte-identical copies outside `.release/`. Its 9 unique release documents were copied (hash-verified) to `.release/archive-release-20261108-docs/`. It was **not** moved, because the quarantine no longer exists.
- Removed: 4 non-running VS Code server builds in `~/.vscode-server/cli/servers/` (~2.6 GB), `remaining_experiments/__pycache__`, the archive release's `.pytest_cache`, and, via `git prune`, a stale `tmp_obj` and loose duplicates in the project `.git`. The two obsolete VS Code extensions were left in place because they were still running.
- Removed an embedded HF token from `/tmp/artfae-hub-repo/.git/config` and from `~/.bash_history`; the owner revoked the token. The HF repo `nawazishpatana/ArtFAE-NET` holds only 7 files (`results/*.json|csv`, 2026-11-10); the 961-file staging set exists only locally.
- `remaining_experiments/` (2026-11-08 reviewer experiments) and `paper/syscom2026_camera_ready/` are new since 2026-11-06 and are kept.
- Re-run checks: hashes as above; `check_claims.py` 315/315 claims, 0 bare numbers, PASSED; `main.pdf` compiles to a temp folder (48 pages, no undefined references); the demo app starts headless and runs without exceptions.

**2026-11-12 (later) — owner decisions carried out.**
- The quarantine is not restored; the ZFS snapshots are the safety net.
- **Deleted `.release/ArtFAE-Net-archive-release-20261108/`** (`rm -rf`, 2026-11-12 00:30 UTC, 42.6 GB, 2,016 files).
  - Before deleting, the copy in snapshot `auto-2026-11-12-0000` was confirmed: all 2,016 files have identical names and sizes.
  - Every image in its tar parts has a byte-identical copy outside `.release/`.
  - Its 9 unique docs are in `.release/archive-release-20261108-docs/`.
- **Project git untracked to 921 files (27.2 MB).**
  - `.gitignore` now excludes images (except `app/assets/examples/*.jpg`), `archive/`, `uploads/`, `Data/`, `artemis_official_data/`, the two fresh-holdout label manifests, `paper/` (its own repo; the gitlink is untracked), archives and zips, `features*/`, `*.pt`/`*.npy`/`*.npz`, `.release/` and caches.
  - Not pushed.
  - `.git` stays at 25.4 GiB, because the Codex ref `refs/codex/turn-diffs/checkpoints/…` (2026-11-08) still references the old full index.
- **Hugging Face staging:** `/workspace/hf_staging/` (1,008 files, 374 MB), rebuilt from the current project.
  - Built by `/workspace/hf_staging_reports/stage_hf_v2.py`. File list: `filelist.tsv`. Exclusions: `dropped.csv`.
  - The `/tmp` clones `artfae-hub-repo` and `artfae-final-only` were deleted.
  - Not uploaded yet.

**2026-11-14 — final cleanup and private GitHub repository.**
- Deleted (owner-approved, 1,226,948,389 bytes): 56 `/tmp/artfae*` scratch items from earlier sessions (including `/tmp/artfae_ui_browser`), `~/.cache/pip`, `remaining_experiments/analysis_first_formatting_error.log` (failed run, superseded by `analysis.log`), the empty top-level `models/` and `figures/`, and the stale `ArtFAE-Net-paper-latex.zip` (rebuild from `paper/` if a shareable LaTeX package is needed).
- Kept for now: `.release/ArtFAE-Net/` (the owner is checking `github.com/nawazpatana/ArtAFE`, which most likely received its LFS release branch), `results/features_clip_v1_20261022/`, `/workspace/hf_staging/`, `~/.cache/ms-playwright`.
- Code pushed to the private repository `github.com/nawazpatana/ArtFAE-Net` (branch `main`, SSH deploy key `/workspace/.ssh_github/github_artfae`, passed per push via `GIT_SSH_COMMAND`). `.gitignore` additionally excludes `study/stimuli/`, `study/precomputed/attention/` and `.vscode/`. No data, images (except `app/assets/examples/`), weights, features or study data are in the repository. Never push this project to `ArtAFE`.
