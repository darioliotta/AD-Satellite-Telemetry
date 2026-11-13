# Chapter 5 evidence

`runs.csv` contains the 270 recorded runs, their original integrated-report paths
and SHA256 hashes, declared checkpoint hashes, BF200/POT operating points,
parameter counts, analytical MACs and recorded batch-one latency medians.
`bf_auc` and `pot_auc` are binary-after-point-adjustment AUC-PA, not continuous
score ROC-AUC. Dense-search results are not used in this chapter.

`evidence.json` records the checked design, four-corner statistics, paired
seed differences, endpoint interaction contrasts, resource reductions and source
revision hashes. Tables use mean and sample SD over seeds 2026–2028. The plotted
single-factor bars use the observed minimum and maximum, not confidence intervals.

From the project root:

```bash
python -B tesi/scripts/prepare_chapter5.py
python -B tesi/scripts/plot_chapter5.py
```

The first command validates source reports and rebuilds the three generated
LaTeX tabular files under `tesi/tables/chapter5/` and the scalar evidence JSON.
The second validates the complete grid and rebuilds vector PDFs and PNG previews
under `tesi/figures/chapter5/`. Neither command trains or evaluates a model.
Plotting requires NumPy and Matplotlib. Table preparation uses the standard library.

The full grid combines disjoint v10 and v12 cells. Matching model hashes alone
does not certify full numerical identity of their runner/evaluation revisions.
All principal four-corner contrasts in the text use v10 only. Native padding
changes the number of duplicated nominal targets and batches when W changes;
OPS-SAT preserves a fixed fragment split and target population.

Timing is A100 batch-one FP32 forward time. It is not an onboard measurement.
The source profiling uses 10 warm-ups and 50 timed calls per run; summaries are across
the 3 seed-specific medians. NASA timing windows are synthetic, while OPS-SAT uses
a real test window. Learned weight sizes exclude buffers and archive overhead.
