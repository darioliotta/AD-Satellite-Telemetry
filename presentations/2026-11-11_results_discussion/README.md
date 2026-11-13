# Benchmarks and ablation discussion

Open `Results.pdf` or `Results.pptx` (35 slides). No cover page. English, Padova red, vector PDF/SVG figures with high-resolution PowerPoint fallbacks. Heatmaps use the exact first-presentation red palette and flat vector cells.

1. Original presentation network schematic, reused unchanged.
2. Previously requested best individual benchmark values. WADI POT 53.39% is marked as the NaN-mask diagnostic; no data rows or labels are dropped. MSL POT 90.17% is the individual maximum, not the closest-to-paper run used in the thesis evidence. SMD POT is 68.61% from V16.
3–4. Window and FF sweeps, mean ± sample SD over 3 seeds, 50 epochs. The other parameter is fixed at its largest tested value. Labelled dataset-specific vertical scales show within-dataset trends.
5–6. All 30 configurations per dataset; three separately labelled colour scales, so small within-dataset trends remain visible.
7, 9, 11. Grouped bar charts replace the previous F1-only tables. Precision, recall and F1 are all shown with mean ± sample SD for both BF200 and POT. OPS-SAT compares W10/FF8 and W3/FF2; SMAP W10 and W3 at FF50; MSL W50 and W13 at FF110.
8. Real OPS-SAT magnetometer fragment 239 with zero-valued stretches; measured positive-sample population share, not channel counts alone.
10. A real SMAP trace, with actual positive-label shading, alongside the within-track equal-value fractions for nominal and positive labels. The event is selected by median length among positive runs of at least 20 samples with at least 50% identical successive values.
12. A real MSL trace, with actual positive-label shading, alongside pooled MSL 95th-percentile absolute steps (scaled by nominal-training range). The event is selected by median length among positive runs of at least 20 samples with within-event step q95 above pooled nominal q95. These are descriptive examples, not representative samples, causal explanations of F1, or expert fault classes.
13. A simplified version of the 12-sample PA example from the second presentation: true labels, direct alarms and adjusted alarms. Added positives and an unchanged false alarm are explicitly distinguished; TP/FP/FN counts are shown before and after PA.
14. Raw false-positive rate and recall for the matched window comparison at each model's saved PA-selected BF200 alpha and threshold. FP/TN counts are verified identical before/after PA.
15–16. BF200 and POT with PA, without PA at fixed selection, and without PA after raw-F1 selection. These use exactly the same trained FP32 reference models within each comparison: W10/W5 for OPS-SAT and SMAP, W50/W25 for MSL, FF fixed, seeds 2026–2028, 50 epochs. The PA results numerically match the corresponding original grid runs. POT chooses alpha among its per-alpha SPOT thresholds; it does not freely sweep thresholds like BF200. Status/metric/manifest hashes are verified. No architectural variant or quantized model is included.
17. Short interpretation distinguishing false alarms, event credit and direct pointwise detection.
18–19. Three architecture diagrams and the four W/FF settings; 240 trained models, five seeds per cell, 50 epochs.
20–21. All four architectures at reference W/FF: F1 with PA and separately optimized F1 without PA, mean ± sample SD over five seeds.
22–24. Per-dataset precision, recall, F1 and binary AUC-PA for all four architectures and BF200/POT, at reference W/FF.
25. Unique parameters and analytical matrix MACs relative to TranAD+ at reference W/FF.
26–27. BF200/POT architecture-by-W/FF grids; five seeds per cell, all 240 trained models included.
28. Weight-only quantization results: quantized storage with FP32 calculations.
29. Integer matrix-product quantization results, W8A8 and W4A4. W4A4 uses INT8 instructions, not native INT4.
30–31. Integer precision by architecture, BF200/POT with PA; each cell averages four W/FF settings and five seeds.
32. Integer precision without PA, optimized for raw F1.
33. Stored tensor payloads in KiB at reference W/FF, including scales/biases/buffers, for every architecture and dataset.
34–35. Integer precision/recall/F1/AUC-PA, averaged over the declared 80 parents per dataset (four architectures × four settings × five seeds).

The appended architecture and precision section contains results only, with no new causal explanations. Selected fixed-setting comparisons and pooled-over-setting comparisons are explicitly separated. V16/V17 metrics and V18 raw-optimized metrics are hash-checked against complete status artifacts; exact run tables and source manifests are under data/. No new training or inference is performed.

The first ablation plots use point adjustment. The added controls explicitly separate PA and no-PA metrics and their optimization criteria. All results use ordinary BF200 and POT, not the dense grid. Ablation results are means, whereas the benchmark table shows selected individual runs. The disjoint V10/V12 cells remain a descriptive combined grid; source-revision differences are recorded in provenance rather than certified numerically equivalent here. The newly added no-PA window comparisons have explicitly labelled window sizes, which need not equal the extreme configurations contrasted on slides 7/9/11.

`data/benchmarks.csv`, `data/ablation_cells.csv`, and `data/provenance.json` retain exact displayed values, source paths and hashes. The build checks the 270 original report hashes and recomputes plotted means/SDs. `native_signal_summary.csv`, `opssat_positive_population.csv`, `opssat_fragment_239.csv`, `smap_real_example.csv`, `msl_real_example.csv`, `example_candidates.csv` and `pa_no_pa_runs.csv` retain the measurements, selection population and paired controls. Native descriptive statistics are pooled/sample-weighted, may reflect channel composition and are not claimed to identify physical fault causes. All signal source hashes are recorded. Original presentations, thesis files, source results and training code are preserved.

Rebuild: `python -B presentations/2026-11-11_results_discussion/build.py` from the repository root. Dependencies: NumPy, matplotlib, python-pptx, lxml (the existing local presentation dependency directory is supported).
