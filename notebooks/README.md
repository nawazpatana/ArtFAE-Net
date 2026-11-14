# Interactive model training

## Separate notebooks by model

Each notebook runs independently with the **Python (ArtFAE .venv)** kernel. Start with setup and settings, enable and run the smoke check, then enable the reference training cell. All switches initially preview commands.

| Model | Notebook | Workflow |
| --- | --- | --- |
| B1 | [ResNet50](B1_ResNet50.ipynb) | Frozen/full backbone training, learning-rate trials, seeds, resume |
| B2 | [Frozen CLIP](B2_Frozen_CLIP.ipynb) | Validate the preserved trained head on CPU |
| B3 | [CLIP + numerical attributes](B3_CLIP_Numerical_Attributes.ipynb) | Train, tune optimizer, replicate, resume, inspect predictions |
| B4 | [CLIP + attribute text](B4_CLIP_Attribute_Text.ipynb) | Train, tune optimizer, replicate, resume, inspect predictions |
| B5 | [Simple multimodal fusion](B5_Multimodal_Fusion.ipynb) | Train, tune optimizer, replicate, resume, inspect predictions |
| M1 | [ArtFAE-Net](M1_ArtFAE_Net.ipynb) | Train, tune, replicate, ablate, resume, inspect attention and evidence |

B2 reuses the existing trained checkpoint rather than retraining it. Its evaluation switch is `RUN['evaluate']`; the other notebooks use `RUN['train']`. B3–B5 use fixed fusion-head width and tune learning rate and weight decay; M1 also supports hidden-width trials. Only M1 includes architecture ablations and evidence evaluation. M2 training is not implemented. The combined notebook remains available below.

Preview execution reports for all six notebooks are saved under `results/notebook_validation/separate_models/`. Delivered notebooks contain no execution outputs.

Open [ArtFAE_Training_and_Improvement.ipynb](ArtFAE_Training_and_Improvement.ipynb) in VS Code Remote SSH on Shoal.

1. Select **Python (ArtFAE .venv)** or `/workspace/.venv/bin/python` as the kernel.
2. Run setup and review **Run settings**: dataset/cache paths, device, booking, epochs, learning settings, and execution switches.
3. For the first GPU M1 experiment, set `RUN['smoke'] = True` and `RUN['m1'] = True`; run the smoke section before the M1 section during your booking.
4. Inspect validation metrics and per-class plots, then enable only the tuning, replication, or ablation sections you want.

All execution switches default to `False`. A default Run All previews commands, writes experiment configurations, and displays existing validation reports without training. The notebook was executed in that mode successfully. It also checks that no training checkpoint or CUDA context was created by preview execution.

The notebook uses the actual project training scripts, not a separate implementation. The completed frozen CLIP cache is reused by default. Every new run writes to a unique path under `results/notebooks/<session>/`; logs, configuration YAMLs, command status, and a run registry are retained. The original dataset and CLIP head are not overwritten.

## Improving results

Train an M1 reference before testing the small learning-rate/width search. Keep data partitions and budgets fixed and judge improvements using validation unique-top Macro-F1, per-class scores, soft cross-entropy, and calibration. Repeat the selected recipe across seeds 42, 43, and 44. Different hyperparameters are separate recipes; repeated runs of one seed are not independent seeds. Rerun the comparison section after replication or ablation jobs finish.

The notebook offers B1 frozen/full ResNet training, preserved B2 validation, B3/B4/B5 head training, M1, controlled search, AGF/ECSA/branch ablations, checkpoint resume, learning curves, inference overlays, and a prespecified validation evidence study. It does not claim that any candidate must improve the score. M2 training remains unimplemented and is explicitly excluded.

GPU commands enforce the booking loaded from `configs/shoal_booking.yaml`. Interrupt a running cell to request graceful checkpointing. If the child process is still saving after 30 seconds, inspect `ACTIVE_PROCESS.poll()` before starting anything else. Training resumes into a new output directory using the unchanged old configuration and `last.pt`.

The held-out test set is not used by the notebook. Final evaluation remains a separate locked-protocol stage described in [TRAINING_GUIDE.md](../TRAINING_GUIDE.md).

## Notebook environment

Kernel and validation support are installed in the existing environment, and the kernel is registered as `artfae`. Optional dependencies are declared in `pyproject.toml` under `notebooks`; exact versions are recorded in `docs/notebook_environment.json`.

If setting up another approved account with the existing research environment:

```bash
/workspace/.venv/bin/python -m pip install 'ipykernel>=6,<8' 'nbformat>=5,<6' 'nbclient>=0.10,<1'
/workspace/.venv/bin/python -m ipykernel install --sys-prefix --name artfae --display-name 'Python (ArtFAE .venv)'
```

Validation artifacts for this notebook are under `results/notebook_validation/`. The delivered notebook is saved without execution outputs so you can start a fresh session.
