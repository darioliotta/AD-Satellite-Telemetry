# TranAD+ Experiment Guide

## 1. Purpose

This document describes the workflow used to preprocess datasets, train TranAD+, evaluate trained models, and preserve experimental artifacts.

The project uses the original TranAD+ implementation together with external Python scripts that provide a consistent interface for experiment execution.

The original authors' code is stored in `TranADPlus/src/` and `TranADPlus/scripts/`.

The surrounding project scripts manage dataset locations, command-line arguments, device selection, reporting, and experiment orchestration.

## 2. Project Layout

```text
tranad_plus/
├── TranADPlus/
│   ├── src/
│   ├── scripts/
│   ├── Datasets/
│   ├── Checkpoints/
│   └── Pickles/
├── logs/
├── results/
├── config.py
├── preprocess.py
├── train.py
├── evaluate.py
├── report.py
├── run_native.py
├── run_experiments.py
├── requirements.txt
├── EXPERIMENT_GUIDE.md
├── PARAMETER_AUDIT.md
└── SOURCE_MANIFEST.json
```

Dataset source files and project-generated preprocessing outputs are maintained under the sibling `../data/` directory.

Some directories inside `TranADPlus/` may be links to external dataset or checkpoint locations.

## 3. Environment Preparation

From the `tranad_plus/` directory:

```bash
python -m pip install -r requirements.txt
```

The requirements file will be audited after the project Python scripts have been reviewed.

Supported execution devices depend on the installed PyTorch build and available hardware:

- `cpu`: CPU execution.
- `mps`: Apple Silicon GPU backend, where supported.
- `cuda:0`: NVIDIA GPU, where supported.

A CUDA-enabled environment is recommended for computationally demanding training runs.

Cloud runtime storage may be temporary. Persistent storage must be configured before long-running experiments.

## 4. Dataset Preprocessing

The preprocessing launcher is:

```bash
python preprocess.py --dataset DATASET
```

Replace `DATASET` with the selected dataset name.

Benchmark datasets currently considered are:

- `SMAP`
- `MSL`
- `SMD`
- `SWaT`
- `WADI`

The preprocessing launcher uses the existing project data structure and the authors' preprocessing implementation.

Generated benchmark data is expected under:

```text
../data/processed/tranad_plus/
```

For example:

```text
../data/processed/tranad_plus/SMAP/
../data/processed/tranad_plus/MSL/
```

The `TranADPlus/Datasets/Preprocessed` path is intended to provide access to these processed data files.

### SMAP Preprocessing Reference

The verified SMAP preprocessing run produced:

```text
SMAP preprocessing verified: 54 tracks; train=138004; test=435826; anomalous=55922
```

This reference applies specifically to the original SMAP data used in the project.

### Preprocessing Verification

Before training, verify:

1. The expected training and test files exist.
2. The dataset dimensions are consistent with the intended model inputs.
3. Anomaly labels and sequence boundaries have been preserved correctly.
4. The preprocessing output is accessible from the expected model directory.
5. Preprocessing artifacts have been saved in persistent storage.

Dataset-specific details will be documented after each preprocessing stage has been completed.

## 5. Experiment Configuration

Experiment configurations are managed through `config.py`.

Each configuration should define:

- Dataset and experiment identifier.
- Number of training epochs.
- Window size.
- Batch size.
- Learning rate.
- Feedforward dimension.
- Number of layers.
- POT risk parameter.
- Data loading and preprocessing-related options.

For published configurations, the parameter values should be traced to the authors' experimental logs or source code.

See `PARAMETER_AUDIT.md` for the currently documented settings.

### Dry Run

Before launching training, inspect the generated command when the selected experiment is supported:

```bash
python train.py --dataset SMAP --experiment pot --dry-run
```

A dry run displays the configured command without training the model.

The availability and naming of all non-SMAP experiment profiles will be checked during the upcoming code audit.

## 6. Training

The training launcher is:

```bash
python train.py --dataset DATASET --experiment EXPERIMENT --device DEVICE
```

Example:

```bash
python train.py --dataset SMAP --experiment pot --device cpu
```

An NVIDIA GPU may be selected as follows:

```bash
python train.py --dataset SMAP --experiment pot --device cuda:0
```

Supported experiment names and hyperparameters are defined in the current project configuration.

Published experiment settings should not be changed without documenting the deviation.

### Completed and Pending Runs

**SMAP**

The window-size-10 and window-size-50 configurations have been trained and evaluated.

**MSL**

The window-size-50, `2x` feedforward configuration completed 50 training epochs on an NVIDIA Tesla T4, but the checkpoint was not retained. It must be trained again before evaluation.

**SMD, SWaT, WADI**

Training is pending preprocessing and configuration verification.

## 7. Evaluation

The evaluation launcher is:

```bash
python evaluate.py --dataset DATASET --experiment EXPERIMENT --device DEVICE
```

Example:

```bash
python evaluate.py --dataset SMAP --experiment pot --device cpu
```

The original evaluation pipeline produces results using both POT-based thresholding and a Best F1 threshold search.

The experiment profile used for training and the evaluation method are distinct concepts: a single trained model may yield metrics for both evaluation methods.

### Important Methodological Distinction

Best F1 selects a threshold using test-set anomaly labels. It is therefore an oracle-style reference metric, not a fully unsupervised thresholding procedure.

This distinction must be preserved when reporting scientific results.

## 8. Combined Execution

The project includes a script for sequential experiment execution:

```bash
python run_experiments.py --dataset SMAP --experiment pot --device cpu
```

The exact behavior and supported options of this script will be verified during the Python code audit.

Avoid launching resource-intensive training runs concurrently on shared hardware without checking resource availability.

## 9. Artifact Management

Every completed experiment should preserve:

| Artifact | Purpose |
|---|---|
| Experiment configuration | Reproduce the training settings |
| Training log | Record training execution and loss progression |
| Model checkpoint | Preserve trained model parameters |
| Evaluation log | Preserve evaluation output |
| Structured metrics | Support comparison and reporting |
| Environment information | Record relevant software and hardware details |

Training logs are stored under `logs/`.

Structured project reports are stored under `results/`.

The original implementation may also produce checkpoint files under `TranADPlus/Checkpoints/` and native evaluation outputs under `TranADPlus/Pickles/`.

Copies of completed artifacts should be kept outside temporary cloud-runtime storage.

### SMAP WS50 Archive

The SMAP WS50 experiment has been archived locally and on Google Drive with the following files:

```text
smap_plus_ws50.ckpt
smap_plus_ws50_train.log
smap_plus_ws50_evaluate.log
best_f1.json
```

The final local archive layout should be kept consistent with future experiment folders.

## 10. Experiment Completion Criteria

An experiment is considered complete when:

1. Dataset preprocessing has been verified.
2. The experiment configuration is recorded.
3. Training finishes successfully.
4. The final checkpoint is preserved.
5. Evaluation is completed.
6. Evaluation metrics and logs are archived.
7. The results are compared against the selected reference.

A completed training run without a recoverable checkpoint or evaluation results does not count as a completed benchmark experiment.

## 11. Reproducibility Notes

- The original implementation is preserved as the experimental baseline.
- Project-specific launchers may add compatibility and orchestration functionality.
- Different PyTorch versions and hardware backends may produce different numerical results.
- Data ordering can influence optimization behavior.
- The original implementation does not provide a complete automatic training-resume mechanism.
- Persistent checkpoint storage must be prepared before using temporary cloud environments.
- Changes to experimental hyperparameters must be documented separately from reference runs.

For detailed parameters and methodological notes, see `PARAMETER_AUDIT.md`.

## 12. Next Steps

1. Finish the local organization of SMAP experiment artifacts.
2. Preprocess and verify SMD.
3. Preprocess and verify SWaT.
4. Preprocess and verify WADI.
5. Finalize and validate all benchmark experiment configurations.
6. Complete the remaining benchmark training and evaluation runs.
7. Evaluate the original model on OPS-SAT.
8. Develop and assess lightweight TranAD+ variants.