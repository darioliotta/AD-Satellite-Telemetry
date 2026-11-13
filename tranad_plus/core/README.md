# TOAST v13: architectural simplification

This sibling of `tranad_plus/` contains the second experimental phase. It uses
the existing datasets and audited baseline implementations; it does not modify
the baseline or write into legacy/v10/v12 results. Keep the two source folders
together when copying the project to a server.

Start with the [server guide](SERVER_GUIDE.md). The existing
`ad-satellite-telemetry` environment supplies the dependencies listed in
`../tranad_plus/requirements.txt`; no additional package is needed by this runner.
CPU unit tests use Python's standard-library `unittest`.

For the extension of the completed 72-run study to five seeds and four W/FF
settings, use the [additional-run guide](ADDITIONAL_RUNS.md):
`bash run_additional.sh --device cuda:0`. The `additional` campaign schedules
168 new jobs, excluding every original experiment. The historical `all`
campaign remains the original 72-job plan.

## Models and hypotheses

| Model | Intervention | Objective | Scoring |
|---|---|---|---|
| `reference` | Unmodified original model | Original epoch-weighted two-output reconstruction loss | Original 21-alpha mix |
| `shared_decoder` | Tie the two decoder modules; retain both encoder/decoder passes and self-conditioning | Same two-output loss | Same 21-alpha mix |
| `single_pass` | Remove decoder 2 and the self-conditioned second pass; retain original zero-context first pass and 2F embedding | Single-output MSE | One reconstruction score; alpha not applicable |
| `encoder_only` | Remove both transformer decoders; apply the original linear-sigmoid head to the final encoder token | Single-output MSE | One reconstruction score; alpha not applicable |

All variants first initialize the reference architecture and then tie/remove
modules. Retained weights therefore have the same initialization for a matched
seed. This is not pretrained weight transfer. The encoder was already shared in
the original model. Sharing decoder weights reduces unique parameters but does
not remove either forward pass. `single_pass` and `encoder_only` change both
architecture and training objective; report these changes together.

The native SMAP/MSL objective uses the original `MSELoss(reduction='none')`
kernel; OPS-SAT retains its existing squared-error expression. Adam, StepLR
(step 5, gamma 0.9), 50 epochs, FP32, TF32 disabled, ordered batches, padding,
and per-dataset batch/LR values match the hyperparameter-ablation protocol.

## Controlled configurations

These are the v10/v12 **study baselines**, not the six-dataset paper-best runs.
FF dimensions below are effective integers.

| Dataset | Reference W / FF | Compact W / FF | Batch | LR | POT q |
|---|---|---|---|---|---|
| OPS-SAT | 10 / 8 | 5 / 4 | 256 | .001 | .01 |
| SMAP | 10 / 50 | 5 / 25 | 1024 | .001 | .07 |
| MSL | 50 / 110 | 25 / 55 | 128 | .0001 | .01 |

Compact settings are predeclared half-window/half-FF settings already represented
in the previous grid. They are not selected by maximizing test scores.

## Campaigns

| Campaign | Runs in plan | New runs after preceding campaigns |
|---|---:|---:|
| `pilot` | 9 | 9: reference/shared_decoder/single_pass, reference setting, seed 2026 |
| `main` | 27 | 18: same comparisons, seeds 2026/2027/2028; pilot reused |
| `compact` | 27 | 27: same three models and seeds at compact settings |
| `encoder_only` | 18 | 18: encoder-only at both settings; controls come from main/compact |
| `all` | 72 | Union of all unique experiments; pilot is not counted twice |

Run the pilot and inspect correctness before spending resources on the other
campaigns. One seed gives exploratory results, not a variability estimate. The
main analysis uses matched-seed differences and mean/sample SD over three seeds.
Test-label-selected metrics are exploratory oracle comparisons. For OPS-SAT,
use held-out validation evidence for model/configuration selection; do not choose
the winning architecture or seed by its test F1.

```bash
cd tranad_plus/core
python run_experiments.py --campaign pilot --dry-run
python run_experiments.py --campaign pilot --check
python -u run_experiments.py --campaign pilot --device cuda:0
```

Every job runs **training immediately followed by evaluation**. `--stage train`
and `--stage evaluate` are available separately. Use repeated `--dataset` or
`--variant` to filter; `--only` selects one exact printed identifier. Default
failure handling records the error and continues; the final exit status is 1
if any selected job failed. `--fail-fast` stops at the first failure. `--dry-run`
and `--check` do not write result artifacts. The latter validates full input
hashes, feature dimensions and dataset splits.

## Data and evaluation

OPS-SAT reuses `opssat_pipeline.prepare_data` and `FragmentWindows`: one shared
feature, per-fragment windows, nominal official-train optimization/scaling,
20% nominal holdout plus anomalous official-train fragments for validation,
fixed split seed 2026 for every training seed, and untouched official test.
Labels are weak whole-fragment labels, not measured pointwise anomaly onset.
Scoring batches retain the preceding study's sizes: OPS-SAT 1024, SMAP 1024,
MSL 128. Training batch sizes remain those in the table above.

SMAP/MSL reuse the existing `TSDataset_tracks`, sorted file pairing, train-fitted
MinMax scaling and official train/test splits. The upstream concatenation and
boundary-window behavior are deliberately preserved. These datasets have no
new validation split in this phase. Inputs are the existing
`data/processed/tranad_plus/{SMAP,MSL}/{Train,Test}/{Tracks,Labels}/*.npy` and
`data/raw/OPSSAT/segments.csv`.

Each successful evaluation records:

- POT with point adjustment and bounded initialization retries (maximum 35;
  otherwise explicitly unavailable, never an infinite retry loop).
- Original BF: 200 linear thresholds, original PA semantics, 21 alphas for
  two-output models; dense BF separately over all distinct thresholds.
- Precision, recall, F1, confusion counts and **binary-after-PA ROC-AUC** for
  each selected method, plus continuous-score AUC/AP per alpha under distinct names.
- Fragment-aware PA for OPS-SAT, original global PA for SMAP/MSL.
- Separate OPS-SAT validation-selected fragment AUC and raw point metrics.
  The existing two-output validation-selection routine is reused unchanged.
  Single-output selection considers calibration/aggregation only, without alpha.

POT and BF choose alpha independently. Alpha/BF test-label search is oracle, not
deployment-time performance. Single-output variants have one score row and
`alpha=null`, not duplicate decoder outputs. NaN/Inf outputs or scores fail the
evaluation and create a diagnostic rather than a valid metric report.

## Results and recovery

```text
results/v13/architectures/
  OPS-SAT/shared_decoder/reference/seed_2026/
    config.json                 # complete effective configuration
    manifest.json               # code/input hashes and execution identity
    data_inventory.json         # actual split/window inventory
    checkpoints/final.ckpt
    checkpoints/resume.ckpt     # durable after every epoch, retained
    train.log
    epochs.jsonl
    evaluation.log
    validation_selection.json
    scores.npz
    metrics.json                # written only after successful evaluation
    status.json
  _campaign_logs/
  all_experiments.csv
  aggregate.json
```

The runner verifies configuration, inputs, source digest and execution identity
before resuming. Completed experiments are reused only after checking their
artifact hashes. Per-run OS locks prevent two launchers writing the same job.
Changing a source/configuration/data/environment under an existing experiment
fails closed; preserve it and use a new output root for a revised study.
`--output-root` is a parent directory and always appends `v13/architectures`.

`BASELINE_SOURCES.json` pins numerical model/data/metric implementations. Path,
catalog and telemetry support modules are recorded at runtime rather than pinned,
so the server's optional WADI catalog can coexist with this runner. Their actual
hashes still enter experiment provenance. No baseline file is edited.

`scores.npz` stores train/test/(OPS-SAT validation) score matrices and labels;
two-output rows follow `alpha_values`. A single-output model has one row and an
empty alpha list. OPS-SAT also stores per-point fragment IDs and channels.
These arrays support reanalysis without rerunning inference. Checkpoints and
optimizer/RNG/scheduler states support exact epoch recovery on the tested CPU
path; CUDA numerical reproducibility must be assessed on the actual platform.

Resources include unique trainable parameters, FP32 parameter bytes, recorded
training memory/time, and batch-one forward latency on the declared CPU/GPU.
Analytical matmul MACs are labeled estimates excluding nonlinearities and I/O.
None of these are satellite-device latency, power or radiation measurements.

## Reports and checks

```bash
python reports/summarize.py
python reports/plot_tradeoffs.py --metric bf_dense_f1 --cost parameters
python reports/plot_tradeoffs.py --metric pot_f1 --cost latency_median_ms
```

Summary tables are updated after each job. Aggregates preserve the available
seed count, report sample SD only for n>1, and keep execution environments
separate. Figures require selecting one execution if multiple environments are
present. Paired deltas require the matching reference dataset/setting/seed and
execution; absent controls are marked unavailable.

From the project root:

```bash
python -m unittest discover -s tranad_plus/core/tests -v
```

Tests cover original-reference numerical identity, first-stage equivalence,
actual weight sharing and parameter reduction, original native loss gradients,
all-variant interruption/recovery, OPS-SAT split/window parity, single-output
alpha semantics, campaign overlap, validation/test separation, and complete
training/evaluation/reuse/corruption rejection on synthetic data. Synthetic tests
write only into temporary directories; they are not thesis results.
