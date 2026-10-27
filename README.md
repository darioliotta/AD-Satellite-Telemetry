# Satellite Telemetry Anomaly Detection

Research project focused on anomaly detection in multivariate time-series data, with particular emphasis on satellite telemetry and the development of computationally efficient deep learning models.

The project is organized into three main stages:

1. **Benchmark reproduction:** reproduce and validate the experimental results of TranAD+ on established anomaly detection datasets.
2. **Satellite telemetry evaluation:** apply the original TranAD+ model to OPS-SAT telemetry data.
3. **Model optimization:** develop and evaluate a lighter version of TranAD+, investigating the trade-off between anomaly detection performance and computational efficiency.

## Project Structure

```text
AD-Satellite-Telemetry/
├── data/
│   ├── raw/
│   └── processed/
├── tranad_plus/
│   ├── TranADPlus/
│   ├── logs/
│   ├── results/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── report.py
│   ├── run_native.py
│   ├── run_experiments.py
│   ├── requirements.txt
│   ├── EXPERIMENT_GUIDE.md
│   ├── PARAMETER_AUDIT.md
│   └── SOURCE_MANIFEST.json
└── README.md
```

The structure above shows the main components rather than the complete contents of each directory.

### Data

The `data/` directory contains the original datasets and their preprocessed versions.

Raw data and generated preprocessing artifacts are kept separate to support reproducibility and avoid modifying the original datasets.

### TranAD+

The `tranad_plus/` directory contains:

- The original authors' implementation in `TranADPlus/`.
- Project-specific Python scripts for preprocessing, training, evaluation, and reporting.
- Experiment configurations and documentation.
- Training logs, model checkpoints, and evaluation results.

The project-specific scripts are intended to provide a consistent interface to the original implementation without unnecessarily changing the authors' model.

## Benchmark Datasets

The benchmark reproduction stage considers five datasets:

| Dataset | Description | Current status |
|---|---|---|
| SMAP | NASA spacecraft telemetry | Training and evaluation completed |
| MSL | NASA Mars rover telemetry | Preprocessed; evaluation pending |
| SMD | Server Machine Dataset | Preprocessing pending |
| SWaT | Secure Water Treatment | Preprocessing pending |
| WADI | Water Distribution | Preprocessing pending |

### Experimental Progress

**SMAP**

Two experimental configurations, with window sizes of 10 and 50, have been trained and evaluated. Their results are close to the corresponding reference F1 scores reported by the TranAD+ authors.

**MSL**

Preprocessing has been completed. A 50-epoch training run was successfully executed, but its checkpoint was lost following a temporary cloud-runtime reset. Training and evaluation must therefore be repeated.

**SMD, SWaT, and WADI**

Preprocessing, configuration verification, training, and evaluation are planned.

The detailed status of individual experiment configurations is maintained in the TranAD+ documentation.

## Experimental Workflow

Each benchmark experiment follows the same general procedure:

1. Acquire and inspect the original dataset.
2. Run and verify dataset preprocessing.
3. Select and document the experimental configuration.
4. Train the model.
5. Preserve the model checkpoint and training logs.
6. Run evaluation and save the metrics.
7. Compare the results with the authors' reference results.

An experiment is considered complete only when its relevant artifacts and metrics have been preserved.

## OPS-SAT Evaluation

After completing benchmark validation, the original TranAD+ model will be evaluated on OPS-SAT telemetry data.

This stage requires inspecting the telemetry structure, defining suitable preprocessing, preparing the experimental data, and establishing a baseline for anomaly detection performance.

The OPS-SAT evaluation is distinct from benchmark reproduction and may require dataset-specific adaptations.

## Lightweight Model Development

The final research stage will investigate modifications to TranAD+ intended to reduce computational requirements.

The original model will provide the baseline against which the optimized variants are evaluated.

Relevant comparison criteria include:

- Detection precision, recall, and F1 score.
- Number of trainable parameters.
- Memory consumption.
- Training time.
- Inference latency.

Architectural changes and their experimental results will be documented separately from the original benchmark reproduction.

## Documentation

Further information is available in:

- [`tranad_plus/EXPERIMENT_GUIDE.md`](tranad_plus/EXPERIMENT_GUIDE.md): preprocessing, training, evaluation, and artifact management.
- [`tranad_plus/PARAMETER_AUDIT.md`](tranad_plus/PARAMETER_AUDIT.md): experimental configurations, reference values, and reproducibility considerations.
- [`tranad_plus/SOURCE_MANIFEST.json`](tranad_plus/SOURCE_MANIFEST.json): provenance and checksums for the original TranAD+ Python files.

## Reproducibility

The project distinguishes between the original authors' implementation and project-specific orchestration scripts.

Experimental configurations, saved model checkpoints, preprocessing artifacts, and evaluation metrics should be preserved to support reproducibility.

Exact numerical agreement with published results is not guaranteed because of differences in software versions, hardware, numerical computation, and data ordering.

## References

- TranAD+ authors' repository: https://github.com/yyexela/TranADPlus
- TranAD+ SpaceOps 2025 paper: *A Transformer-Based Deep Learning Approach to Anomaly Detection of High-Bandwidth Multivariate Time-Series Satellite Communications*.

Additional datasets, literature, and experiment-specific references will be documented as the research progresses.

## Project Status

**Ongoing research.**

The immediate priorities are completing the preprocessing of the remaining benchmark datasets, auditing their experimental configurations, and preparing the next training and evaluation runs.