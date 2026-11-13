# SMAP/MSL: real examples and isolated controls

Open **SMAP_MSL_investigation.pdf** (18 pages). No training, changed labels, alternative threshold protocol, or source-result changes.

## Read first

- Pages 5–6: SMAP P-4, an interval of 111 positive labels containing two narrow visible peaks. Original BF200 direct hits: reference 2, smaller W 1; after PA both receive 111. The interval annotation does not assert that every flat sample is a physically abnormal measurement.
- Pages 13–14: MSL M-7. Original BF200 misses the event in the reference; POT hits it. Smaller W can receive full PA credit from a single alarm.
- Page 17: isolate threshold choice on MSL. Model, alpha=.65 and score are identical. The original POT threshold for that alpha gives test PA F1 84.97%, versus BF200 40.71%. This is a diagnostic control using an existing per-alpha threshold, not a new optimized protocol.
- Page 18: the W25 BF200 alarm at M-7 index970 is driven by encoded input coordinate20. Telemetry-coordinate error supplies 0.17% of the score at that point; the other inputs supply 99.83%. The alarm is inside a positive interval. This does not identify a command fault or prove the alarm is spurious; it shows that inspecting only telemetry reconstruction misses what creates the score.
- Pages 7–8: SMAP A-2 is a counterexample. Reference detects the annotated interval, encoder-only misses it at its original selected endpoint. This BF pattern occurs in two of the three checked seeds.
- Pages 15–16: MSL T-9 is another counterexample in the displayed seed. It is not stable across the three seeds and is explicitly marked 1/3; do not use it to claim a general architectural disadvantage.

## What was held fixed

Every plot reuses seed2026, 50 epochs and original V13 selected BF200 or POT alpha/threshold. Hyperparameter pages compare reference architecture at reference/window-only/FF-only/compact settings. Architecture pages compare four architectures at the same reference W/FF. BF and POT get separate pages because their selected alphas can differ; one curve must not be paired with a threshold selected for a different alpha.

Original PA is computed over the complete saved test stream **before** cropping. Plots show coordinate0 of the input (telemetry), while the anomaly score is MSE across all 25 SMAP or 55 MSL inputs. Other coordinates encode command-related information; their indices do not identify a physical sensor or decoded spacecraft command.

The top/score panels use the same colors, styles, shared sample axes and positive-label shading throughout. Error panels distinguish FP, FN and correct predictions, with exact counts before/after PA. Nominal crops can only contain FP/TN; positive intervals can contain TP/FN. No PA removes an FP.

## Label audit

81 processed tracks (54 SMAP, 27 MSL) match the inclusive start/end intervals used by original preprocessing. Labels are indexed by time within the long track, not one class for the whole track. The input-label matrices repeat the same temporal label across feature columns; this does not make independent feature-level annotations and those labels are not input features.

SMAP P-2 has duplicate, inconsistent metadata rows; current preprocessing uses the first, as in the original code. That policy is recorded in label_checks.csv. P-2 is excluded from the examples. This audit verifies agreement with the adopted annotation, not the physical correctness of every expert boundary.

## Why these cases

Selection is illustrative and outcome-based, not a random estimate of performance. Anchor seed2026 is fixed. Candidate categories and all candidates are saved in selection.json. Nominal examples are median lexical 160-sample blocks among declared FP-reduction candidates. Positive examples are median-duration events in the declared detection category. Selection prefers agreement in at least two of seeds2026–2028; fallbacks are explicitly marked. Cases from the preceding visual PDF are excluded.

## Key quantitative checks

matched_global_summary.csv reports original endpoint means over seeds2026–2028, preserving alpha and threshold selection for PA. Its raw F1/recall are **at the PA-selected setting**, not the separately raw-optimized V18 endpoints.

For SMAP BF200, W10→W5 at FF50 changes PA F1 82.90→84.90%, PA recall 82.82→81.15%, direct recall 2.02→1.64%, and nominal point FPR 2.49→1.46%. The improvement is a false-alarm/event-coverage tradeoff, not uniformly better point detection.

For MSL POT, reference→encoder-only at W50/FF110 changes PA F1 85.69→87.72%, PA recall 93.44→91.70%, direct recall 10.35→7.57%, and nominal point FPR 2.85→2.01%. Again, fewer false alarms coexist with fewer directly detected positive samples.

MSL BF200 has a separate threshold-grid issue. At reference seed2026, alpha=.65, the first nonminimum BF candidate is about .007488; the 99th percentile of scores is only .00008749. The original same-alpha POT threshold .00002648 lies in the large untested interval between BF's first two candidates. AUC-score stays 61.00% when only threshold changes. Rare extreme errors affect standard deviation and grid spacing; this is not evidence that changing a model's architecture alone caused the BF jump.

In SMAP P-4 the normalized peak value is 1. At fixed alpha=.5, reference and W5 reconstruct the second peak as .92376 and .93868, respectively: the shorter-window model reconstructs this anomalous peak MORE accurately and therefore assigns a smaller telemetry error. The current target is already in the input window, so nominal-only training does not guarantee failure to reconstruct anomalous values. This local observation is not a universal causal proof about all windows.

## Artifacts

- build.py: label/score audit, recorded candidate selection, original-endpoint examples and checkpoint replay.
- explain.py: matched-alpha reconstruction controls, threshold-only control and exact per-feature MSE contributions.
- example_counts.csv: every plotted TP/FP/FN/TN count.
- original_endpoint_metrics.csv / matched_global_summary.csv: original full-stream endpoint results and matched three-seed summaries.
- MSL_matched_alpha_threshold_control.csv: same scores, original BF threshold versus original POT threshold for the identical alpha.
- matched_alpha_reconstruction_controls.csv: alpha=.5 controls, checked against stored score arrays.
- MSL_M7_feature_contributions.csv: decomposition of all 55 reconstruction-error coordinates.
- provenance.json / verification.json: source hashes, all 64 reconstruction/score checks, metadata policy and counting checks.

Source: official Telemanom README, "Anomaly labels and metadata" and "Raw experiment data": https://github.com/khundman/telemanom#anomaly-labels-and-metadata
