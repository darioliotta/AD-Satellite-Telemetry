# Consistent W / FF visual comparison

Open Visual_checks.pdf. Seven real intervals, four TranAD+ models per interval: reference, smaller W, smaller FF, both smaller. All use seed2026, 50 epochs and the same other training settings within each dataset.

Three panels: observed telemetry and reconstructions; four aligned score lanes with POT (solid) and BF200 (dashed) thresholds; pointwise errors for both threshold methods, before and after PA. Orange = false positive, red = false negative, white = correct. Counts appear on the right. All panels share the same sample axis.

Alpha is fixed at .5 for every model. Thus the same reconstruction and score legitimately have both POT and BF200 thresholds. Thresholds were recalculated using full original stored scores; their PA F1 is checked against the saved alpha=.5 report. This is a controlled visual diagnostic, not the independently alpha-optimized headline benchmark. Raw predictions use the identical PA-selected threshold: no raw retuning. PA is applied to full original sequences / separate OPS fragments before cropping.

The selected cases illustrate behavior, not its frequency. Both favorable and unfavorable examples are retained. OPS cases239/1560/1281 come from prior outcome-based controls (then encoder-only, seed2028), now shown for all four reference-architecture configurations at seed2026. SMAP G-7 is a prior morphology-selected event; E-4 was selected for a prior window-related false-alarm difference. MSL P-11 is morphology-selected; M-6 deliberately illustrates the largest previously found out-of-output-range target. Physical fault causes are not inferred from these traces.

SMAP/MSL top panels show coordinate0; scores use all 25/55 inputs. OPS labels cover whole fragments, not precise physical fault locations. Shading always represents the same supplied positive labels. FP is invariant under PA; fewer FN after PA means event credit, not necessarily better time localization.

No training, checkpoint changes, original-result changes or thesis changes. Build script, compressed plotted arrays, exact counts and source hashes are included for reproducibility.
