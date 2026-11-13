# Ablation preview — original BF200 and POT

Open **Ablation_preview.pdf** (5 pages). PNG previews, vector PDFs and SVGs are under `figures/`.

- 270 runs: 3 datasets × 6 windows × 5 feed-forward widths × 3 seeds; 50 epochs.
- Every plotted scalar is verified against its original report and report SHA-256.
- Main 1D slices fix the other hyperparameter at the largest width/window. Lines connect recorded means, not fitted trend models. Shading and error bars show sample SD across 3 seeds, not confidence intervals.
- Heatmaps show all 30 cells per dataset. Their colour scales are local to each dataset; numeric labels permit direct comparison. The outlined cell is the largest W/FF reference, not a selected best result. A fifth page displays BF200 seed variability on a common scale.
- Only original Best F1 with 200 thresholds and POT, both point-adjusted. No dense search. Original per-run test-selected alphas/thresholds are retained, not recalibrated.
- BF200 can be lower than POT because its finite threshold grid can miss useful operating points. The marked MSL variability must not automatically be attributed to reconstruction quality; score distribution and threshold resolution are plausible contributors. No new training or score evaluation is performed here.
- The combined grid uses disjoint V10/V12 cells. This is a descriptive presentation of saved measurements, not certification that their source revisions are numerically identical. Four endpoint comparisons (largest/smallest W and FF) are all V10.
- Native padding changes the number of duplicated targets with W; OPS-SAT retains its fixed test fragments. Cross-dataset raw point metrics are not interchangeable with independent fragment classification.

`source_runs.csv` preserves all 270 rows. `cell_summary.csv`, `slice_summary.csv`, and `endpoint_comparisons.csv` contain exact plotted statistics. `provenance.json` records source/report hashes and plot definitions.

Rebuild from the repository root:

```bash
python -B presentations/2026-11-11_ablation_preview/build.py
```

Existing thesis text, prior presentations, research results and training code are not changed.
