# V18 — BF200 and POT selected without point adjustment

One current implementation under `tranad_plus/`; this is an evaluation campaign,
not another source copy. Output: `results/v18/no_pa/`.

## Scope

- OPS-SAT, SMAP and MSL.
- TranAD+, one-decoder, one-pass and encoder-only.
- The same reference / window-only / FF-only / compact configurations as V17.
- All five model seeds, 2026–2030.
- FP32, INT8 and INT4: 240 existing models, **720 evaluation cases**.
- No training, QAT, new weight quantization, FP16 or dense threshold search.

The existing V17 exported integer models and V13 FP32 checkpoints are reused
without modification. Test inference runs with the actual score batch recorded
for each V17 precision case. This also preserves the corrected MSL FP32 batching.
Data splits, normalization, windows and OPS-SAT fragment boundaries are unchanged.

## Selection and comparisons

BF200 searches the original 200 linearly spaced thresholds from minimum score
to mean + 3 standard deviations. For two-output models it selects alpha from the
original 21 values in [0,1]. The objective is ordinary pointwise F1 **without PA**.
Single-output architectures have one score row and no alpha parameter.

POT does **not** search arbitrary thresholds by F1. V17 already stored the SPOT
threshold for each alpha. PA is applied after POT fitting, so those thresholds
can be reused. V18 keeps q, fitting and the per-alpha thresholds unchanged and
selects the alpha with the highest **pointwise** F1. Only test inference is needed;
training scores do not need regenerating. All 7,920 V17 per-alpha POT thresholds
were present in the local source results when preparing this campaign.

For each method and precision, the report contains this crossed comparison:

| Selection | Reported without PA | Reported with PA |
|---|---|---|
| Original V17 PA selection | `no_pa_at_pa_selection` | `pa_at_pa_selection` |
| Newly optimized without PA | `no_pa_optimized` | `pa_at_no_pa_selection` |

PA never participates in the new selection. It is applied separately afterward
only for these comparison views, respecting OPS-SAT fragment boundaries.
The original V17 results are not edited or relabelled.

Metrics: F1, precision, recall, TP/TN/FP/FN, alpha and threshold. Binary ROC-AUC
is called **AUC-bin** without PA and **AUC-PA** after PA; both equal balanced
accuracy on the relevant predictions. These are not continuous-score ROC-AUC.
An all-positive-classifier baseline is retained so a high-recall trivial
solution is recognizable. All selection remains a declared test-label oracle,
as for Best-F1; it is not a deployment-calibrated estimate.
BF200 is the best on that prescribed grid, not an exact all-score threshold
maximum. The coarse-grid limitation on MSL is not removed by this experiment.

## Validation and provenance

The read-only preflight verifies source V17 completion, every V17 artifact hash,
source-model hashes, numerical code compatibility, input hashes/order, original
alpha lists and all per-alpha POT thresholds. The runtime also checks the server
execution environment and regenerates the same test input layout and score batch.
Original numerical model/scoring/quantization implementations remain frozen;
only new campaign selection/report code is introduced.

Regenerated per-alpha test score statistics are compared against V17 using the
original rtol=5e-4, atol=1e-7. These are aggregate checks, not a claim of exact
pointwise equality: V17 did not archive full pointwise score arrays. Confusion
counts at the old alpha/threshold are compared explicitly with the V17 values.
Small threshold-boundary count changes are recorded, with the old reference and
current comparison distinction retained; no data are masked. Large distribution
changes fail the campaign. Missing inputs, unknown numerical code changes,
unsupported CUDA or absent integer instructions are errors.

The existing integer runtime still uses true INT8 x INT8 -> INT32 instructions.
W4A4 uses four-bit values and packed weights on that INT8 execution kernel;
it is not native INT4 Tensor Core execution. No FP32 GEMM fallback is added.

## Launch

From `tranad_plus/`, after activating the server environment:

```bash
python -B run_campaign.py thresholds --dry-run
bash run_no_pa.sh --device cuda:0
```

The launcher runs read-only checks before starting inference. Progress shows
case / 720, dataset, architecture, W, FF, seed, precision, test batch and alpha
search. There are no epochs because there is no training.

For a short first case:

```bash
bash run_no_pa.sh --device cuda:0 \
  --only OPS-SAT/encoder_only/compact/seed_2026/fp32
```

Then launch the full command: completed cases are verified and skipped.
The same full command resumes after interruption; only an unfinished precision
case repeats inference. Code/data/environment changes are rejected rather than
mixed with old results.

```bash
python -B run_campaign.py thresholds --status
python -B run_campaign.py thresholds --verify
```

The terminal total is **720 precision evaluations**, not 720 newly trained
models. Parent model count remains 240. Default order is OPS-SAT, SMAP, MSL;
compact configuration first and simpler architectures first within each block.

## Saved output

Each case has a source/data/model manifest, effective profile, scalar metrics,
all per-alpha candidates, score-statistics parity checks, timing, process memory
telemetry, progress, logs and integer instruction audit where relevant. Models
and pointwise arrays are not duplicated. No repeated latency profiling is done:
V17's resource measurements remain the resource evidence.

Tables are rebuilt every 12 cases and at completion:

- `tables/per_seed.csv`: all four comparison views, both methods, all metrics.
- `tables/mean_std.csv`: mean and sample SD across five seeds per configuration.
- `tables/selection_comparison.csv`: benefit of raw selection over PA selection.
- `tables/precision_comparison.csv`: matched INT8/INT4 minus FP32 differences.

The output ceiling is **1 GiB**, not a claimed archive-size measurement. Only
compact reports are saved, so this avoids the previous large score archives.
Model artifacts remain in the existing V13/V17 result trees. No files under
those source trees are changed.

Tests from the project root:

```bash
python -B -m unittest tranad_plus.campaigns.thresholds.test_thresholds
```
