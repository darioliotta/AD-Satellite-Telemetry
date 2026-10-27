# TranAD+ Experimental Parameter Audit

## 1. Purpose

This document records the provenance of experimental hyperparameters used to reproduce and evaluate TranAD+.

Its primary goals are to distinguish published settings from project-specific choices, preserve methodological details, and support reproducible benchmark comparisons.

## 2. Evidence Sources

Three sources must be distinguished.

### A. Published Paper

*A Transformer-Based Deep Learning Approach to Anomaly Detection of High-Bandwidth Multivariate Time-Series Satellite Communications*, SpaceOps 2025.

The paper describes the model, the experimental methodology, and aggregate benchmark results. It does not provide a complete hyperparameter table for every individual training run.

### B. Authors' Experimental Logs

The authors' repository contains experimental logs with run-specific hyperparameters and evaluation results.

These logs are the primary reference for reconstructing the published experiment configurations.

### C. Original Source Code

Relevant implementation files include:

- `TranADPlus/scripts/training_loop.py`
- `TranADPlus/scripts/evaluation.py`
- `TranADPlus/src/models.py`
- `TranADPlus/src/helpers.py`
- `TranADPlus/src/datasets.py`
- `TranADPlus/src/omnianomaly.py`
- `TranADPlus/src/global_config.py`

The original source code establishes model behavior and implementation details that may not be fully described in the paper.

The project-specific wrappers and configurations must be audited separately.

## 3. SMAP: Verified Reference Configurations

Two configurations were extracted from the authors' SMAP experimental logs:

- `TranAD_SMAP_lr0-001_dff2x_ws10.txt`
- `TranAD_SMAP_lr0-001_dff2x_ws50.txt`

| Parameter | WS10 | WS50 |
|---|---:|---:|
| Dataset | SMAP | SMAP |
| Tracks | 54 | 54 |
| Input features | 25 | 25 |
| Epochs | 50 | 50 |
| Batch size | 8192 | 2048 |
| Window size | 10 | 50 |
| Learning rate | 0.001 | 0.001 |
| Feedforward dimension | `2x` = 50 | `2x` = 50 |
| Encoder layers | 1 | 1 |
| Layers per decoder | 1 | 1 |
| Attention heads | 25 | 25 |
| Transformer model dimension | 50 | 50 |
| Dropout | 0.1 | 0.1 |
| Shuffle | False | False |
| Padding | True | True |
| Data loader workers | 0 | 0 |
| Downsampling | None | None |
| POT risk `q` | 0.07 | 0.07 |

### Dataset Characteristics

Verified SMAP preprocessing summary:

| Property | Value |
|---|---:|
| Tracks | 54 |
| Training observations, before padding | 138004 |
| Test observations, before padding | 435826 |
| Anomalous observations | 55922 |

Window-based sample counts depend on the window size and padding behavior.

### Optimizer and Learning Rate Scheduler

The original training implementation uses:

- Optimizer: Adam.
- Adam betas: `(0.9, 0.999)`.
- Adam epsilon: `1e-8`.
- Weight decay: `0`, under the standard optimizer defaults.
- Scheduler: `StepLR(step_size=5, gamma=0.9)`.

### Model and Loss

The original model uses a two-stage decoder reconstruction architecture.

Training uses a weighted combination of decoder reconstruction losses, with epoch-dependent weighting.

The model is trained using single-precision floating-point values.

### Normalization

The original dataset implementation uses MinMax scaling fitted on training data and subsequently applied to test data.

The exact scaling implementation must be preserved when comparing the reference model with modified variants.

## 4. SMAP: Reference Results

The SMAP reference values below were extracted from the authors' experimental logs.

| Metric | WS10 / POT | WS50 / Best F1 |
|---|---:|---:|
| F1 | 0.842405811 | 0.849094337 |
| Precision | 0.877676789 | 0.783788282 |
| Recall | 0.809860162 | 0.926272308 |
| ROC AUC | 0.896633136 | 0.944457771 |

These reference columns use different experimental configurations and evaluation methods. They must not be treated as the results of a single training run.

## 5. SMAP: Reproduction Results

Two project runs have been completed.

| Metric | WS10 | WS50 |
|---|---:|---:|
| POT F1 | 0.843085479 | 0.847022967 |
| Best F1 | 0.849415398 | 0.841426261 |

Additional recorded WS50 metrics:

| Metric | POT | Best F1 |
|---|---:|---:|
| Precision | 0.854302708 | 0.840787759 |
| Recall | 0.839866242 | 0.842065734 |
| ROC AUC | 0.909462560 | 0.909376689 |
| True positives | 46967 | 47090 |
| False positives | 8010 | 8917 |
| False negatives | 8955 | 8832 |

The SMAP WS50 run completed 50 epochs in 658 seconds, with a final reported training loss of approximately 0.0002.

The WS50 checkpoint, training log, evaluation log, and structured Best F1 comparison have been preserved locally and on Google Drive.

### Interpretation

The reproduced F1 scores are close to the corresponding published reference values.

However, some secondary metrics differ more substantially. For example, the reproduced WS50 Best F1 ROC AUC is approximately 0.90938, compared with the reference value of approximately 0.94446.

Benchmark replication must report these differences rather than claiming exact numerical reproduction.

## 6. MSL: Recorded Experiment Configurations

The project configuration and previous execution records identify two MSL experimental profiles.

| Parameter | POT profile | Best F1 profile |
|---|---:|---:|
| Window size | 50 | 50 |
| Epochs | 50 | 50 |
| Batch size | 2048 | 2048 |
| Learning rate | 0.0001 | 0.0001 |
| Feedforward | `2x` | `16` |
| Number of layers | 1 | 1 |
| POT risk `q` | 0.01 | 0.01 |
| Shuffle | False | False |
| Padding | True | True |
| Workers | 0 | 0 |
| Downsampling | None | None |

The MSL POT profile was executed with the identifier:

`msl_plus_ws50_dff2x`

The run used 55 input features and completed all 50 epochs on an NVIDIA Tesla T4.

Recorded results:

- Final reported training loss: approximately 0.0028.
- Training time: 615 seconds.
- Evaluation: not completed.
- Final checkpoint: lost after the cloud runtime was reset.

Consequently, no project-measured MSL F1, precision, recall, or ROC AUC results are available from that run.

The MSL configuration must be checked against the complete original author logs during the final configuration audit.

## 7. Remaining Benchmark Configuration Inventory

The following settings were previously collected during project planning. They are **provisional records until cross-checked against the original experimental logs and the current `config.py` implementation**.

| Dataset | Window | Epochs | Batch size | Learning rate | Feedforward |
|---|---:|---:|---:|---:|---|
| SMD | 20 | 500 | 8192 | 0.0001 | `16` |
| SWaT | 50 | 50 | 2048 | 0.0001 | `2x` |
| SWaT | 50 | 50 | 2048 | 0.001 | `16` |
| WADI | 20 | 50 | 8192 | 0.001 | `2x` |
| WADI | 50 | 50 | 2048 | 0.0001 | `2x` |

These entries are not a substitute for an audit of:

1. Original log filenames and experiment identifiers.
2. Other parameters, including POT `q` and layer counts.
3. Reference evaluation values.
4. Availability of the corresponding profiles in `config.py`.
5. Correct mapping between training profiles and reported metrics.

No training should be represented as a published-configuration reproduction until these checks are completed.

## 8. Evaluation Methodology

### POT

Peak Over Threshold is used to determine anomaly thresholds from anomaly scores according to the original evaluation implementation.

For the verified SMAP configuration:

- POT risk parameter: `q = 0.07`.
- Initial level: `0.98`.
- Dynamic POT: disabled.

The original implementation may encounter numerical difficulties on degenerate score distributions. Its behavior should be documented rather than silently modified during reference reproduction.

### Best F1

Best F1 searches for the threshold yielding the highest F1 score using test-set labels.

The reference implementation uses a grid of 200 thresholds.

**Best F1 is an oracle-style evaluation metric**, because it relies on labels from the evaluation data.

It must not be described as a label-free thresholding technique.

### Decoder Combination

The original evaluation implementation searches for decoder-combination weights:

- `alpha` from 0 to 1 in 21 steps.
- `beta = 1 - alpha`.

The optimal values are evaluation outcomes, not fixed training hyperparameters.

Reference SMAP logs report:

- WS10 / POT: `alpha = 0.05`, `beta = 0.95`.
- WS50 / Best F1: `alpha = 0.30`, `beta = 0.70`.

The score combines decoder reconstruction outputs according to the original model evaluation code.

### Point Adjustment

The evaluation pipeline includes point adjustment of anomaly predictions.

This behavior must be retained in reference comparisons and explicitly documented when interpreting results.

## 9. Reproducibility Considerations

### Differences From Native Defaults

The original source configuration includes defaults that differ from the run-specific hyperparameters used in the authors' reported experiments.

For example, the native SMAP defaults include 5 training epochs and feedforward dimension 16, whereas the audited SMAP reference runs use 50 epochs and feedforward dimension `2x`.

Experiment launchers must explicitly pass the intended reference configuration rather than relying on native defaults.

### Data Ordering

The original data pipeline may depend on filesystem directory enumeration order.

Different platforms can therefore produce different training trajectories despite identical nominal hyperparameters.

### Hardware and Numerical Differences

Different hardware backends and software versions can produce small numerical differences.

Results obtained on CPU, Apple MPS, and NVIDIA CUDA devices should be accompanied by sufficient environment information.

### Checkpoint Recovery

The original implementation does not provide a complete automatic training-resume workflow.

Saving a final checkpoint to persistent storage is essential, particularly when using cloud environments with temporary filesystems.

### Model Provenance

The original model source files are tracked by `SOURCE_MANIFEST.json`.

The baseline reproduction should preserve the authors' implementation as a separate reference from future architectural modifications.

## 10. Current Audit Status

| Item | Status |
|---|---|
| SMAP WS10 configuration | Verified against reference documentation |
| SMAP WS50 configuration | Verified against reference documentation |
| SMAP training and evaluation | Completed |
| SMAP WS50 artifact archival | Completed |
| MSL preprocessing | Completed |
| MSL training and evaluation | Pending completion |
| MSL configuration cross-check | To finalize |
| SMD configuration cross-check | Pending |
| SWaT configuration cross-check | Pending |
| WADI configuration cross-check | Pending |
| Full Python launcher audit | Pending |
| Requirements audit | Pending |

## 11. Next Audit Tasks

The next documentation update should follow the review of the existing project Python scripts.

In particular:

1. Verify the final experiment registry in `config.py`.
2. Check the exact command-line behavior of `train.py` and `evaluate.py`.
3. Confirm dataset-specific preprocessing options.
4. Validate experiment identifiers and checkpoint naming.
5. Record the remaining authors' reference metrics.
6. Update the requirements and environment documentation.
7. Preserve a clear distinction between original implementations, external orchestration scripts, and future model modifications.