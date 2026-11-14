# ArtFAE-Net human-evaluation study

A study app, separate from the demo (`app/`). It reuses the demo's look (`app/ui.py`) and emotion names
(`app/i18n.py`) without changing them, in English and Traditional Chinese. Nothing in `app/`, the models or `runs/` is modified.

## Flow

1. Consent. The app shows `consent_en.md` / `consent_zh.md`, currently the **pilot** wording (version
   `pilot-1.0-2026-11-14`). In pilot mode there are two checkboxes:
   - required: "I am 18 or older, I have read this information, and I agree to take part in this pilot." The study
     cannot start without it;
   - optional, unticked by default: "If a research ethics committee later approves it, my anonymous pilot answers may
     be used in research and shared in an anonymised research dataset."
   Stored with the participant: `consent_pilot`, `future_use_optin`, `consent_version`. Production mode shows one
   checkbox and needs `CONSENT_VERSION` set. The final wording is in `consent_*_FINAL_TEMPLATE.md`, which the app
   never reads.
2. The participant code (e.g. `K7QM-3HTR`), on its own screen: "Please write it down; you need it to withdraw your
   answers." It is stored as `participant_code` and shown again on the thank-you page.
3. Instructions, then 1 practice trial.
4. 22 screens: 20 stimuli and 2 attention checks.
5. Questionnaire: demographics and the 10 SUS items.
6. Thank-you page with a completion code (and a reminder of the participant code).

Each stimulus trial has four steps, each with its response time recorded:

| step | shows | asks |
|---|---|---|
| a. blind | the artwork only | the emotion felt most (9 options, required), strength 1–5 (required), a reason (optional) |
| b. reveal | the R1 "10 viewers" graphic and the 9 bars | plausibility 1–5 |
| c. A/B | R1 and R3, unlabelled, sides randomised and recorded, fixed emotion order, shared axis | A / B / no difference |
| d. attention (optional) | the R1 attention overlay for R1's top emotion | 1–5, or skip |

Every step also has **"Skip this artwork"**. It stores the trial with `status = skipped`, `skipped = true` and
`skipped_step` (blind, reveal, ab or attn), keeps the steps submitted before it, and moves to the next trial.
`analysis.py` treats the skipped step and every later step of that trial as missing data, never as answers, and
counts a skipped attention check as not passed.

Each submitted step is saved immediately. A participant who closes the page partway keeps everything they
submitted: their row has `status = incomplete` (it becomes `complete` when the questionnaire is submitted), and a
trial left mid-way has `status = incomplete` with only the submitted steps filled in.

**The blind step never reads model outputs.** They are only reachable through `model_outputs()`, which the
blind step does not call. `tests/test_study.py` checks this on every blind screen, and fails if a read is added.

**Balanced assignment** (`assignment.py`):
- The 60 stimuli are split into 3 sets of 20, interleaving the strata.
- Participant *k* gets set *k* mod 3, so after every 3 participants each stimulus has been rated once.
- Each participant gets a shuffled order and random A/B sides.
- Two attention checks are inserted at random positions: one in screens 5–9, one in 14–19.

## Files

| file | purpose |
|---|---|
| `select_stimuli.py` | picks 60 stimuli from the fresh holdout (seed 20261112), plus a practice item and 2 attention-check images. Strata: 9 emotions × (3 high + 2 low agreement), 9 tied, 3 dark, 3 minimalist. All 60 are dated before 1950 (slug year) and use 60 different artists. Writes `stimuli.csv`, `private/votes.csv` and `stimuli/images/` |
| `artist_rights.py` | image rights and form, for all 63 shown images. Every image must pass three rules: the artist died before 1956, the work was created before 1931, and it is 2D (no sculpture, relief, architecture or ceramics). Sources are the WikiArt artist record (with a Wikidata fallback) and the WikiArt painting page (genre, media, date), cached in `artist_meta.json` and `artwork_meta.json`, plus a look at every image. It also applies owner exclusions. Failures are replaced in place (same ID and stratum, a new artist, passing every rule) and logged with the reason in `rights_replacements.csv`. Run after `select_stimuli.py` and before `precompute.py` |
| `precompute.py` | runs the demo's own pipeline (GPU optional) for R1 and R3, computes the 10-viewer counts and the R1 attention overlay; checks every stimulus against the saved pre-registered holdout logits (max difference 8e-6). Writes `precomputed/` |
| `frozen.py`, `stimuli_FROZEN.txt` | **the frozen stimulus set (2026-11-12).** SHA-256 of `stimuli.csv`, the 63 images, the precomputed outputs (`outputs.json`, `maps.npz`, the 61 attention overlays) and the 3 assignment sets. `study_app.py` checks every hash at the start of each session and refuses to start on any difference (changed, missing or extra file). Rerunning `select_stimuli.py`, `artist_rights.py` or `precompute.py` therefore stops the study until someone deliberately re-freezes with `python study/frozen.py --freeze`. Check with `python study/frozen.py` |
| `study_app.py` | the Streamlit study (`streamlit run study/study_app.py`) |
| `storage.py` | storage interface. `SQLiteStorage` for the pilot; `HFDatasetStorage` (JSONL plus `CommitScheduler`) pushes to a private HF dataset, token from the `HF_TOKEN` environment secret |
| `withdraw.py` | `python study/withdraw.py CODE` deletes every row of the participant with that code from the active backend (as the app picks it; or `--sqlite FILE`, `--hf-repo REPO`, `--jsonl FOLDER`; `--dry-run` first). SQLite is VACUUMed; on HF the JSONL files are rewritten in one commit and the history is squashed (`--keep-history` to skip). Logs `<UTC date>\t<code>`, nothing else, to `private/withdrawals.log`. Warns about earlier exports under `private/` that still contain the participant. On a running Space, restart it and run again (a live process can re-upload its local file) |
| `purge_pilot.py` | deletes pilot data (`phase = pilot`) from `private/pilot_real/pilot.sqlite`: all of it, or with `--keep-optin` everyone except participants with `future_use_optin = true`. Prints what it will delete, asks you to type `DELETE PILOT DATA`, refuses without an interactive terminal and has no bypass flag, so it never runs automatically. Production rows are never touched. VACUUMs afterwards. `--dry-run` to preview |
| `export.py` | writes `trials.csv` (long format, one row per trial) and `participants.csv` |
| `analysis.py` | analysis skeleton, **not run on real data**: exclusion rule, softCE / KL / top-1 against R1, R3 and the ArtEmis ceiling, plausibility with a bootstrap CI, R1-vs-R3 binomial test, splits by language |
| `sim.py`, `pilot_sim.py` | synthetic participants driving the real app (AppTest); `pilot_sim.py` runs 5 of them, then the export and the analysis |
| `space/` | Hugging Face Space packaging (CPU only). `space/build.py` writes the upload folder `space/dist/`. Not deployed |
| `tests/test_study.py` | `artfae_demo/.venv/bin/python -m pytest study/tests` |

`private/` (ArtEmis vote shares, pilot data) is never shown to participants, never copied into the Space and is
git-ignored.

## Pilot data

Pilot data are **kept**, not deleted automatically, and kept apart from everything else:
- **Stored** in SQLite at `study/private/pilot_real/pilot.sqlite`. The folder is 700 and the file 600, owner only; the
  app sets this. `study/run_local_pilot.sh` runs the local pilot there with `umask 077`.
- **Tagged:** every participant and trial row has `phase = "pilot"`. `analysis.py` leaves pilot data out by default.
- **Research use:** pilot answers may enter research only for participants who ticked the optional
  `future_use_optin` box, and only after a research ethics committee has approved that use. Only then run the
  analysis with `--pilot optin`. Pilot participants without the opt-in never enter research.
- **Deadline:** if no such approval has been obtained by **31 March 2027**, delete all pilot data with
  `python study/purge_pilot.py`. Once approval is in place, use `--keep-optin` to delete everyone except the
  opted-in participants. Also delete any exports made from pilot data.
- **Withdrawal:** a participant withdraws with their participant code: `python study/withdraw.py CODE`.

## Before real data collection

- Replace the pilot consent (`consent_en.md`, `consent_zh.md`) with the final templates once ethics approval is in
  place; fill the `[待填：…]` placeholders in the zh template and have a native speaker check both zh files.
- Replace the zh-TW SUS items with a validated Traditional Chinese SUS.
- Image rights: done by `artist_rights.py`. All 63 images pass: artist died before 1956, created before 1931, 2D. Visual review (2026-11-12): S25 (a sculpture) was the only 3D work; S06 (an Annunciation painted to look like statues, oil on panel), S42 (a book vignette) and S59 (Khnopff, *The Veil*) were checked and kept as 2D.
- Pre-register the exclusion rule and the analysis plan.
