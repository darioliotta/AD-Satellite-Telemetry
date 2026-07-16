# AD-Satellite-Telemetry

This repository hosts the development of my Master's Thesis in **Physics of Data** at the University of Padova. The research is conducted under the framework of the **TOAST** (Transformers On-Board for Anomaly-detection in Satellite Telemetry) project.

## Project Goal
The ultimate goal of this project is to implement, evaluate, and optimize a compact Transformer-based anomaly detection (AD) pipeline capable of running under the strict resource constraints of satellite edge-hardware (FPGAs or embedded GPUs). 

Specifically, starting from the **TranAD** multivariate architecture and using the **OPSSAT-AD** benchmark dataset, we aim to study the trade-off between:
1. **Model Compression:** Reducing parameter size via parametric pruning and lightweight design.
2. **Detection Performance:** Measuring how well the compressed model retains anomaly detection accuracy (such as F1-score or AUC) compared to the unpruned baseline.

## Repository Structure
* `data/` : Contains the benchmark datasets (`dataset.csv` and `segments.csv`).
* `docs/` : Bibliography references (`references.bib`) and research notes.
* `src/` : Python source files for data processing, model training, and pruning experiments.
* `notebooks/` : Jupyter Notebooks for exploratory data analysis and experimental testing.