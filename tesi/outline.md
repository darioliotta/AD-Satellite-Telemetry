# Thesis outline and progress tracker

**Title:** Towards Lightweight AI Models for Anomaly Detection in Satellite Telemetry Data

The seven-chapter structure was approved on 2026-11-02. This is a writing plan,
not a claim that the scientific prose or planned experiments are complete.
The personalized title page, its metadata and the existing visual style are preserved.

## Scientific scope

The thesis studies how to simplify a transformer-based anomaly detector while
characterizing detection performance and resource requirements. It contributes
to the initial data/protocol and baseline/compact-model activities of TOAST
(WP1 and WP2 in `../docs/papers/TOAST.pdf`). Target-device deployment, radiation
robustness, fault injection and irradiation are future project activities unless
explicitly supported by additional experiments.

Dataset roles are distinct:

- **Baseline reproduction:** the six benchmarks used in the TranAD+ paper
  (SMAP, MSL, SMD, SWaT, WADI and DSN_1k).
- **Main simplification study:** OPS-SAT, SMAP and MSL.
- **Reference application:** OPS-SAT; discuss its fragment labels and single-feature
  input explicitly when interpreting results.

## Research questions

- How closely can the published reference results be reproduced after reconstructing
  their configurations and evaluation protocol from official materials?
- How do window size and feed-forward dimension affect detection performance,
  resource requirements and variability on the selected telemetry datasets?
- Which architectural simplifications offer useful detection--cost trade-offs,
  especially on OPS-SAT, within the limits of the evaluated platform and protocol?

## Chapter status

Use `[todo]`, `[draft]`, `[review]` and `[done]` for **writing status**.
An approved outline does not imply a completed chapter.

| Chapter | Title | Source | Writing status |
| --- | --- | --- | --- |
| 1 | Introduction | [chapter-introduction.tex](chapters/chapter-introduction.tex) | [draft] |
| 2 | Background and Reference Models | [chapter-background.tex](chapters/chapter-background.tex) | [draft] |
| 3 | Datasets and Experimental Protocol | [chapter-datasets.tex](chapters/chapter-datasets.tex) | [draft] |
| 4 | Baseline Reproduction and Adaptation | [chapter-replication.tex](chapters/chapter-replication.tex) | [todo] |
| 5 | Window Size and Feed-Forward Dimension Ablation | [chapter-ablation.tex](chapters/chapter-ablation.tex) | [todo] |
| 6 | Architectural Simplification | [chapter-architecture.tex](chapters/chapter-architecture.tex) | [todo] |
| 7 | Discussion and Conclusions | [conclusions.tex](chapters/conclusions.tex) | [todo] |
| A | Reproducibility Materials | [appendix.tex](appendix/appendix.tex) | [todo] |

## Detailed outline

### 1. Introduction

1.1 Satellite telemetry and anomaly detection  
1.2 Towards lightweight models for onboard inference  
1.3 The TOAST project and thesis scope  
1.4 Objectives and research questions  
1.5 Contributions and organization of the thesis

Establish the application, the scientific questions and the boundary between the
thesis and later TOAST activities. State contributions according to their actual
completion status.

### 2. Background and Reference Models

2.1 Anomaly detection in time series  
2.2 Reconstruction-based detection and anomaly scores  
2.3 Attention and transformer architectures  
2.4 TranAD  
2.5 TranAD+  
2.5.1 Reference architecture and training objective  
2.5.2 Changes relative to TranAD  
2.6 Model simplification and resource-aware inference

Provide the mathematics and verified literature needed to understand the reference
implementation and subsequent interventions. Keep the original paper descriptions
distinct from observed code behavior.

### 3. Datasets and Experimental Protocol

3.1 Dataset selection and roles in the study  
3.2 OPS-SAT telemetry  
3.2.1 Telemetry fragments and annotation granularity  
3.2.2 Data organization and split construction  
3.3 SMAP and MSL  
3.4 Preprocessing and temporal windowing  
3.5 Training, validation and test protocols  
3.5.1 Original benchmark protocol  
3.5.2 OPS-SAT validation and test protocol  
3.6 Anomaly scores, thresholds and detection metrics  
3.6.1 Decoder combination and anomaly scoring  
3.6.2 POT and the original Best F1 search  
3.6.3 Dense Best F1 search  
3.6.4 Point adjustment, label granularity and ROC-AUC  
3.7 Computational cost and runtime measurements  
3.8 Reproducibility and experimental controls

Define the protocols before presenting results. Document differences in labels,
splits, threshold selection and data ordering. Treat computational estimates and
measured runtime as separate quantities.

### 4. Baseline Reproduction and Adaptation

4.1 Initial reproduction attempts with TranAD  
4.2 TranAD+ implementation and configuration reconstruction  
4.3 Reproduction of the original benchmarks  
4.4 Sensitivity to the threshold-search protocol  
4.5 Adaptation of TranAD+ to OPS-SAT  
4.6 Reproduction assessment and remaining discrepancies

Explain the choice of reference implementation using documented attempts and
source comparisons. Map each reported benchmark result to its actual configuration;
the best POT and BF can originate from different runs. Preserve existing runs when
additional configurations are evaluated. Compare threshold searches on the same
checkpoint before interpreting score changes as model improvements.

### 5. Window Size and Feed-Forward Dimension Ablation

5.1 Research hypotheses and experimental design  
5.2 Reference configurations and controlled comparisons  
5.3 Window-size reduction at fixed feed-forward dimension  
5.4 Feed-forward reduction at fixed window size  
5.5 Joint reduction of window size and feed-forward dimension  
5.6 Variability and comparison across datasets  
5.7 Detection performance versus resource requirements  
5.8 Implications for architectural simplification

Use the existing study results after verifying evaluation compatibility between
campaigns. Distinguish study baselines from paper-best configurations. Analyze
isolated effects, interactions and seed variability without assuming monotonic
performance loss. Give OPS-SAT a central role in the interpretation.

### 6. Architectural Simplification

6.1 Design hypotheses and model variants  
6.2 Training and evaluation of the variants  
6.3 Anomaly-detection performance  
6.4 Resource requirements and measured latency  
6.5 Detection--cost trade-offs and candidate configurations  
6.6 Limitations and implications for onboard inference

The variants are implemented and the initial architectural campaign is complete;
additional seeds and single-factor controls extend the study. The chapter prose
remains unwritten. Document accompanying changes to
training objectives and evaluation. Include negative findings and retain the
distinction between exploratory test-based comparisons and validation-based selection.

### 7. Discussion and Conclusions

7.1 Answers to the research questions  
7.2 Main methodological and experimental findings  
7.3 Limitations and validity of the conclusions  
7.4 Future work within TOAST

Tie conclusions to completed experiments. Relate the model-level findings to future
deployment work without claiming onboard readiness from server measurements alone.

### Appendix A. Reproducibility Materials

A.1 Experiment identifiers and configurations  
A.2 Software versions and execution commands  
A.3 Additional benchmark details and reproduction results  
A.4 Additional tables and diagnostic plots  
A.5 Computational cost and profiling details

## Evidence and writing boundaries

- Keep **measured**, **computed**, **literature-backed** and **proposed** content distinct.
- The original BF grid and dense threshold sweep are different evaluation protocols.
  Report their results separately; test-label-optimized metrics are oracle results.
- Distinguish binary-after-point-adjustment ROC-AUC from continuous-score ROC-AUC.
- Do not treat fragment and point labels as equivalent annotations.
- Record splits, batch sizes, seeds, checkpoints and code versions for every comparison.
- Existing results and analysis under `../results/legacy/`, `../results/v10/study/`,
  `../results/v12/study/` and `../results/analysis_v10_v12/` are evidence to verify,
  not automatically approved thesis prose.
- Attribute every numerical claim to its experiment or source in a comment or the journal.
- Keep hardware cost analysis within the experimental chapters. Deployment,
  quantization and radiation studies belong in future work unless actually performed.
