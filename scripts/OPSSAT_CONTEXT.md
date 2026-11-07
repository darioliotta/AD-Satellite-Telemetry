# OPS-SAT context analysis

Run from the local project root:

```bash
python -B scripts/opssat_context.py
```

Dependencies: Python 3.10+, NumPy, Matplotlib and tqdm. No PyTorch, CUDA or network
access is needed. Progress bars show the existing endpoints and frozen controls.
Use `--no-figures` for CSV/Markdown output only.

Results are written to `results/analysis_opssat_context/`. Start with `REPORT.md`
and `figures/channel_comparisons.pdf`. Source experiments are read-only; their
score hashes are verified, their global metrics are reproduced, and the analysis
refuses to write inside the original campaigns or the data directory.

## Questions addressed

1. Does the W10-to-W5 change at fixed FF affect magnetic and solar-angle channels
   differently, at both 50 and 100 epochs?
2. Does it depend on actual fragment duration, sampling, large timestamp gaps,
   repeated timestamps, local departures or endpoint changes?
3. Does the same encoder-only checkpoint depend on the order of its history, or
   on having varying past samples at all?

The analysis reads 80 V13 architectural runs and 30 OPS-SAT V14 continuation
endpoints. Ten existing encoder-only context controls supply the original,
reversed-past and repeated-current conditions. They are not new training runs.
The two-output frozen context exports use fixed alpha=.5, which is generally
different from the saved BF/POT winning alpha; those exports are deliberately
excluded from the threshold-based sensitivity tables rather than mixing score
definitions. The trained-model tables include all four architectures.

## Evaluation policy

- Only **original Best F1 with 200 thresholds** and **POT** are used.
- The saved global threshold and alpha of each run are retained for every
  channel/subgroup. A separate subgroup winner is never selected.
- For frozen encoder-only controls, the original checkpoint's BF/POT thresholds
  remain fixed across all interventions. Its unperturbed global metrics must
  reproduce the original report before comparisons proceed.
- All reported AUC values are **AUC-PA**, computed from adjusted binary
  predictions. Continuous-score AUC and dense threshold searches are not used.
- Class labels belong to whole fragments. Descriptors such as local departure
  are observable morphology, not expert annotations of spikes, drift or faults.
- Duration and morphology cutoffs use only nominal optimization fragment IDs.
  The same test fragments are compared across windows and seeds.

## Interpretation limits

PA can favor longer positive fragments, and the main threshold optimizes global
pointwise F1, not equal weight per channel or fragment. Always read class counts
and fragment recall alongside the main metrics. A single-class channel has no
defined AUC; sparse channels provide weak evidence even with five model seeds.

Source timestamps sometimes repeat for different values. The analysis preserves
all original rows. Actual window spans use timestamps; local shape descriptors
use sample order and do not pretend to be physical time derivatives. No unit,
attitude history or cause of failure is inferred from these descriptors.

Separately trained W5/W10 models test the combined effect of training and window
size. Frozen interventions hold weights and targets fixed but can be out of
distribution. Together they provide evidence about sensitivity, not a complete
physical explanation. A future W1 model trained from scratch would be a cleaner
no-history training control if these results justify another campaign.

Verify the diagnostic invariants:

```bash
python -B -m unittest discover -s scripts -p test_opssat_context.py -v
```

## Follow-up: matched signal, score and threshold examples

```bash
python -B scripts/opssat_window_cases.py
```

Output: `results/analysis_opssat_window_cases/`. This focuses on FF8, W5/W10,
TranAD+ and encoder-only, 50/100 epochs, and all five seeds: 40 saved endpoints,
40 method/seed comparisons and the same 529 test fragments throughout.

Start with `REPORT.md`, `figures/signal_score_examples.pdf` and
`figures/outcomes_and_decomposition.pdf`. Complete populations are retained in
CSV files. A fragment outcome is called stable when four of five seeds agree;
this is a descriptive rule, not a significance test. Examples are median-duration
cases within a declared outcome category; no category is silently substituted
when empty. Example channels require at least five samples of each fragment class.

The analysis crosses the two models' saved score grids, alpha choices and
thresholds. It reproduces actual endpoint metrics before examining hybrid
settings. The numerical factor allocation is exact and symmetric but depends on
score scale. It is diagnostic bookkeeping, not a causal attribution to physics.
No new thresholds are selected and no dense search is performed.

`FINDINGS.md` in the output folder is a manually reviewed interpretation of the
current results and examples. Recheck it if source experiments change. The
automatically generated `REPORT.md`, CSVs and figures always reflect the current
inputs. `peak_transition_alignment.csv` reports error-peak proximity to large
sample-to-sample transitions at all three declared tolerances (0, 2, 5), rather
than selecting the most favorable tolerance. This is not expert onset localization.

```bash
python -B -m unittest discover -s scripts -p 'test_opssat*.py' -v
```
