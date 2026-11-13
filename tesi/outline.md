# Thesis outline and progress tracker

**Title:** Towards Lightweight AI Models for Anomaly Detection in Satellite Telemetry Data

The initial structure was approved on 2026-11-02. On 2026-11-10, the author
added a separate quantization chapter and removed the reproducibility appendix.
The resulting eight-chapter structure is a writing plan,
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
- How does quantization affect model storage and detection performance, both
  with dequantized weights and with integer matrix operations?

## Chapter status

Use `[todo]`, `[draft]`, `[review]` and `[done]` for **writing status**.
An approved outline does not imply a completed chapter.

| Chapter | Title | Source | Writing status |
| --- | --- | --- | --- |
| 1 | Introduction | [chapter-introduction.tex](chapters/chapter-introduction.tex) | [draft] |
| 2 | Reference Models | [chapter-background.tex](chapters/chapter-background.tex) | [draft] |
| 3 | Datasets and Experimental Protocol | [chapter-datasets.tex](chapters/chapter-datasets.tex) | [draft] |
| 4 | Baseline Reproduction and Adaptation | [chapter-replication.tex](chapters/chapter-replication.tex) | [draft] |
| 5 | Ablation Study | [chapter-ablation.tex](chapters/chapter-ablation.tex) | [draft] |
| 6 | Architectural Simplification | [chapter-architecture.tex](chapters/chapter-architecture.tex) | [todo] |
| 7 | Quantization | [chapter-quantization.tex](chapters/chapter-quantization.tex) | [todo] |
| 8 | Discussion and Conclusions | [conclusions.tex](chapters/conclusions.tex) | [todo] |

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

### 2. Reference Models

2.1 Anomaly detection in time series  
2.2 Reconstruction-based detection and anomaly scores  
2.3 TranAD  
2.4 TranAD+  
2.4.1 Reference architecture and training objective  
2.4.2 Changes relative to TranAD  
2.5 Model simplification and resource-aware inference

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

4.1 TranAD+ implementation and configurations  
4.2 Reproduction of the original benchmarks  
4.3 Evaluation checks and the WADI discrepancy  
4.4 Adaptation of TranAD+ to OPS-SAT  
4.5 Reproduction assessment

Start directly from TranAD+; the earlier TranAD-attempts section was removed at
the author's request. Map each selected benchmark result to its actual configuration.
The draft compares single-run values selected retrospectively for closeness to
the published F1, independently for POT and original BF200. Configuration agreement
and numerical agreement remain distinct. SMD POT is updated to the verified 68.61% from the received v16 seed-3
replication (same author configuration). Mark WADI's 53.39% NaN-mask diagnostic and
retain its finite-output control alongside it. No new bibliography entries are
needed. Preserve existing runs when additional configurations are evaluated.

### 5. Ablation Study

5.1 Experimental design  
5.2 Reference settings and comparisons  
5.3 Window-size reduction  
5.4 Feed-forward reduction  
5.5 Joint reduction and interaction  
5.6 Variability across datasets  
5.7 Detection performance and computational cost  
5.8 Implications for architectural simplification

The draft uses the 270 verified v10/v12 reports and original BF200/POT with PA
and binary AUC-PA; no dense-search results replace these metrics. Tables report
three-seed means and sample standard deviations. The main four-corner contrasts
all use v10; the extended combined grid is descriptive because matching model
hashes alone do not certify full pipeline identity. Two vector figures show
grid means/variability and isolated W/FF sweeps. OPS-SAT supports a favorable
joint reduction; SMAP improves at the joint endpoint; MSL preserves POT with
lower computed cost but has unstable BF200. Physical explanations remain
qualified. Distinguish arithmetic/weight reductions from A100 latency.

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

### 7. Quantization

7.1 Quantization methods and experimental protocol  
7.2 Quantized weight storage and floating-point calculations  
7.3 Integer matrix operations with quantized weights and activations  
7.4 Anomaly-detection performance  
7.5 Storage and computational requirements  
7.6 Limitations and implications for onboard inference

The chapter has its own outline. Use the completed precision and integer studies
under `../results/v16/`, `../results/v17/` and the no-PA analysis under
`../results/v18/`. Distinguish weight-only quantization with dequantized FP32
computation from quantized weights/activations with real integer matrix products.
Four-bit operand precision does not establish native INT4 instruction execution.
Keep stored bytes, operation counts and measured runtime distinct.

### 8. Discussion and Conclusions

8.1 Answers to the research questions  
8.2 Main methodological and experimental findings  
8.3 Limitations and validity of the conclusions  
8.4 Future work within TOAST

Tie conclusions to completed experiments. Relate the model-level findings to future
deployment work without claiming onboard readiness from server measurements alone.

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
- Keep hardware cost analysis within the experimental chapters. Quantization has
  completed experiments and a dedicated chapter. Target-device deployment and
  radiation studies remain future work unless actually performed.
