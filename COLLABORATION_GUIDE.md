# Two-account collaboration on Shoal

Share the approved read-only dataset by a configurable `--data-dir`, not by copying images or changing permissions from this code. Keep account-specific results outside the common dataset. Each collaborator uses a unique run directory and records their booking, machine, config, seed, environment, and manifest hashes.

Keep emotion order and split manifests identical. Compare manifest SHA256 values recorded in validation reports and feature-cache metadata. A collaborator's `image_path` values may be stale; loaders resolve `filename` beneath the configured image directory. Never modify the source manifests to accommodate a home directory.

Use the existing environment when compatible. Compare `docs/phase0/environment.txt` and current package versions before changing packages. Checkpoint resume requires the matching configuration, cache, architecture, and normalization state; model-only weights are insufficient for continued training.

Schedule GPU use within approved bookings. SIGINT/SIGTERM preserves the last completed epoch checkpoint; a new account can resume from an authorized shared checkpoint and an identical cache. Output directories must be new. Do not assume a process survives the booking transition. GPU start/end timestamps must include a timezone offset.

Commit code, configs, tests, and documentation. Dataset images, caches, checkpoints, logs, credentials, and large outputs are ignored. Never publish artwork images or upload the dataset without explicit permission. No email, repository push, or external dataset transfer is part of the current implementation.
