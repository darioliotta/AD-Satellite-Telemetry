# V20 — What changes when the detector is simplified?

Current code: `tranad_plus/`. Results: `results/v20/mechanisms/`.
This is a mechanism investigation, not another attempt to select the largest F1
or prove that an original author overfit. Earlier experiments remain immutable.

## Three blocks on OPS-SAT, SMAP and MSL

| Block | Work | Count |
|---|---|---:|
| `analysis` | Reevaluate existing V13/V17 parents, FP32/W8A8/W4A4 | 432 evaluations, no training |
| `baselines` | Persistence and past-median detectors | 6 deterministic evaluations |
| `train` | Observed vs hidden current sample, matched retraining | 72 training trajectories |

The first block uses four architectures × four W/FF profiles × three seeds ×
three datasets = **144 existing parents**, each evaluated in three precisions.
Seeds are 2026, 2027, 2028. The four profiles are reference, window-only, FF-only
and compact from the established V13 grid. All original checkpoints and integer
exports are reused read-only. FP32 is reevaluated with the same scoring batch as
the integer formats, providing a paired reference for this campaign. No model
weights are duplicated in this block.

The training block uses **all four architectures**, two input conditions and
three seeds on **all three datasets**: 4 × 2 × 3 × 3 = **72 new trainings**.
It deliberately fixes W and FF to one profile per dataset to isolate current
sample availability. It does not repeat the entire architectural/W/FF grid:

| Dataset | W | FF | Layers | Training batch | LR | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| OPS-SAT | 10 | 8 | 1 | 256 | 0.001 | 50 |
| SMAP | 10 | 50 | 1 | 1024 | 0.001 | 50 |
| MSL | 50 | 110 | 1 | 128 | 0.0001 | 50 |

These are study profiles, not the separate V19 benchmark anchors. Retain
original ordered Adam, weight decay 0, dropout 0.1, epoch-dependent reconstruction
objective and StepLR every 5 epochs with gamma 0.9. Both conditions have identical
eligible samples, split, normalization, initialization seed and duration.

No early stopping or retrospective checkpoint choice is applied here. Fixed
50-epoch endpoints isolate the input intervention from changes in training
length or checkpoint-selection criteria. Nominal train/validation curves are
still saved every epoch using fixed-alpha score MSE in evaluation mode.
Diagnostics preserve RNG states and do not change subsequent dropout draws.

## What the three blocks test

1. **Score separation vs event coverage.** Compare class score distributions,
   continuous-score ROC-AUC and average precision with raw/PA decisions and
   event coverage. Improvements confined to PA/event coverage support an
   interval-alert interpretation, not necessarily better point localization.
2. **Simple local detection.** Compare persistence and the median of past samples
   under the same controlled data/calibration protocol. Competitive simple
   baselines suggest that local changes can explain part of the task; they do
   not establish a particular sensor's physical fault mechanism.
3. **Access to the current sample.** Retrain with and without that value in the
   model input. A systematic change in the architecture gap supports a role for
   this reconstruction path. Current-sample reconstruction is a legitimate
   task; this control does not establish train/test contamination.

An inconclusive or negative result is retained. The experiments distinguish
plausible statistical/algorithmic mechanisms; anonymized telemetry and weak
labels do not automatically permit a physical diagnosis.

## Data families must remain separate

### Historical analysis

Reproduce inference inputs for the existing V13/V17 models. Verify completed
V17 artifacts, input hashes, source compatibility and checkpoint/export hashes.
Exclude artificial padded target copies from all V20 score/decision summaries;
keep every real sample. Preserve raw track/fragment boundaries for event
statistics. Exact excluded counts are reported.

OPS-SAT has its original held-out validation, so nominal validation scores
calibrate the primary threshold. Native V13 models trained on the complete
official training set: their nominal training-score calibration is explicitly
labelled **in-sample**, not unseen validation. This historical block cannot
retroactively create an independent holdout for those models.

Historical BF/POT endpoints from V17 are retained as context metadata. Their
alpha/threshold optimization and padding conventions differ from the V20
controls; do not treat the new diagnostic numbers as exact paper replication.

### Matched controls and simple baselines

OPS-SAT retains the original fragment split and channel scalers, fitted on
nominal optimization fragments. Native data use the first 80% of each official
training track for optimization and the last 20% for nominal validation;
MinMax scales fit optimization points only. Official test tracks are the source
of test data; they are not split into a labelled validation set.

Construct **complete, unpadded windows within each raw unit**. Discard W-1
warmup targets per unit; report any fragments shorter than W that are excluded.
No window crosses a track, a fragment, or an optimization/validation boundary.
The two training conditions and the simple baselines use exactly the same
remaining targets. These cropped complete-window results are a new controlled
family and must not be directly pooled with historical full-test results.

Raw series and target indices are stored in RAM, with one window batch gathered
on demand. CUDA uses a resident series/index buffer when affordable; overlapping
windows are not materialized into a large N×W×F disk or GPU archive.

## Exactly what is hidden

- `observed_current`: the standard window includes its current sample; decoder
  queries receive that sample, following the reference reconstruction task.
- `hidden_current`: replace **every feature** of the last input row with the
  fixed normalized value zero and replace decoder queries with zeros. Retain
  the true current sample only outside the model, for the loss and final score.

Masking applies during **training, monitoring and test**. Both retain the same
W-1 real past samples and the same positional indices. Self-conditioning uses
only the masked input and model predictions. In encoder-only the final-position
representation is reconstructed from the past plus the fixed placeholder.
The absence of padding prevents early windows from copying the hidden target
into a supposed past position. This is an input-availability experiment, not a
radiation-fault or sensor-corruption simulation.

Simple baselines use the same past samples:

- `persistence`: reconstruction is the previous sample;
- `past_median`: reconstruction is the per-feature median of W-1 past samples.

Their score is mean squared error against the current observation. They have
no learned network weights, but still use input buffers and the shared fitted
scalers. They are evaluated once each, not duplicated across seeds.

## Fixed scoring and explicit metric names

All two-output models use **alpha=0.5**, fixed before examining test results;
one-output models have no alpha. Combine reconstruction outputs first, then
calculate mean squared error across features, following the existing score
formula. This removes alpha optimization as a confound in this diagnostic.

Primary calibration is the **99th percentile of nominal calibration scores**,
aiming for a 1% nominal point alarm rate on the calibration set. Apply strict
`score > threshold` and retain the actually achieved rate (ties can prevent an
exact 1%). Freeze the threshold on test; no test labels enter this calibration.
The target is a comparison setting, not a claimed mission-operational requirement.

Save:

- `AUC_score`: ROC-AUC of continuous scores, computed exactly from all samples;
- `AP_score`: average precision, explicitly not trapezoidal PR area;
- raw F1/precision/recall/confusion counts and AUC-bin;
- PA F1/precision/recall/confusion counts and AUC-PA, from the same direct alarms;
- nominal test false-positive rate, event coverage, direct-hit fraction and
  first-alarm position inside detected annotated runs.

PA is bounded by each raw track/fragment. This diagnostic fixes the event
boundary explicitly and does not silently claim the upstream index-zero/global
padding convention is reproduced. OPS-SAT labels are weak fragment labels:
first-alarm position inside a labelled fragment is not measured physical fault
onset or physical detection latency.

Supplementary endpoints are also saved:

- **BF200 raw oracle**, threshold chosen by test pointwise F1, alpha still fixed;
- **BF200 PA oracle**, threshold chosen by test PA F1, alpha still fixed;
- **POT fixed alpha**, fitting nominal training scores and a nominal calibration
  stream, with the original dataset q and bounded original SPOT initialization.

The BF endpoints are declared test-label oracles. Both use exactly 200 linear
thresholds from minimum score to mean+3×standard deviation, not a dense search.
Unsuccessful POT fitting is explicitly unavailable, never silently replaced.
An all-positive baseline is saved to identify trivial high-recall solutions.

### OPS-SAT fragment classification

Also produce **one prediction per fragment, without PA**. Its score is the
fixed 95th percentile of point scores. Choose its threshold by BF200 F1 on the
labelled validation fragments and freeze it on test fragments. Alpha and the
95th percentile are not searched. Save fragment-level F1/precision/recall,
AUC-bin and continuous score separation. Native whole tracks are not arbitrarily
assigned binary anomaly labels: this endpoint is OPS-SAT-specific.

## Archive and progress

Each case saves manifests, exact source/data/model hashes, data policy, scalar
results, class histograms/quantiles, short plotting curves, timing and class MSE
contributions by feature **index**. Physical sensor identities are not inferred
from indices. Only plotting coordinates are subsampled; AUC calculations use all
scores. OPS-SAT additionally keeps small fragment-score NPZ files, not full
pointwise arrays.

Only the 72 new trajectories save weights: final weights and one overwritten
Adam/RNG/scheduler recovery file. Per-epoch curves contain scalars only. No
per-epoch checkpoint archive, copied historical weights, reconstructions,
predictions, full score grids or duplicated raw data are stored.

Full preflight on the actual local models gives a conservative allowance of
approximately **421 MiB**. The configured budget is **1 GiB**; usage is checked
before each case and work stops if it has exceeded the budget, preserving all
existing files. This is a stopping guard, not a filesystem quota. No scientific
artifacts are deleted to enforce it.

The terminal shows case identity, dataset/variant/condition/precision, training
epoch, batch and timing. Verbose loader messages remain in `run.log`. Each new
model is evaluated immediately after training, before the next trajectory.
Failures have a dedicated `error.json`; `campaign.json` records interrupted or
failed cases. Re-run the identical command to verify completed cases and resume
an unfinished training from the last durable epoch.

Tables are regenerated incrementally:
`per_case.csv`, `mean_std.csv`, `paired_comparisons.csv`,
`simple_detector_comparisons.csv`. Metrics are fractions. Seed variation is
computed within the same dataset/profile/format/condition; deterministic
baselines have one case and no invented seed uncertainty. Matrix-operation
counts are analytic inference estimates, not measured backward FLOPs or energy.
Integer formats have measured representation/buffer sizes; no native INT4 speed
or power advantage is asserted.

## Transfer and launch

From the local repository root, transfer **code only**:

```bash
rsync -avh --checksum --partial --progress \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  tranad_plus/ \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/tranad_plus/
```

The server needs its existing `data/`, `results/v13/architectures/` and
`results/v17/integer/`. Preflight catches missing artifacts before training.
Do not update source files during an unfinished source-bound campaign.

On the server:

```bash
tmux new -s v20
```

Inside tmux:

```bash
source /home/dliotta/AD-Satellite-Telemetry/ad-satellite-telemetry/bin/activate
cd /home/dliotta/AD-Satellite-Telemetry/tranad_plus
bash run_mechanisms.sh
```

This runs all three blocks in order. The 510 cases contain **72 trainings**;
432 reuse existing models, and 6 are deterministic detectors. There is no SMD,
WADI or additional benchmark retraining. CUDA/Triton preflight verifies genuine
integer instructions before the integer block. FP32, W8A8 and W4A4 are all scored
with the same per-dataset batch size. W4A4 uses INT8 instructions and INT32
accumulation, not native INT4 execution; documented nonlinearities remain FP32.

Status, from `tranad_plus/`:

```bash
python -B run_campaign.py mechanisms --status
python -B run_campaign.py mechanisms --verify
```

Optional separate blocks (same output root, verified completed cases are skipped):

```bash
bash run_mechanisms.sh --stage analysis
bash run_mechanisms.sh --stage baselines
bash run_mechanisms.sh --stage train
```

`--dataset`, `--variant`, `--seed` and exact `--only` filters are available.
`--format` filters historical evaluations only. Training conditions remain paired.
Use `--dry-run` to inspect the exact cases; `--check --device cpu` performs a
read-only archive/data preflight. CPU FP32-only evaluation is possible, but the
integer production path requires CUDA and never substitutes floating arithmetic.

Afterward, from the local repository root:

```bash
mkdir -p results/v20/mechanisms
rsync -avh --checksum --partial --progress \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/results/v20/mechanisms/ \
  results/v20/mechanisms/
```

## Verification

From the repository root:

```bash
python -B -m unittest tranad_plus.campaigns.mechanisms.test_mechanisms -v
```

Tests exercise all four hidden-current architectures (including decoder query
and zero-gradient access), complete-window boundaries, nominal-only scaler fit,
historical real-target masks, PA event boundaries/BF200 exhaustive parity,
calibration independence from test labels, past-only simple predictors, exact
recovery and training independence from test values. Synthetic CLI integration
covers paired controls, baselines, reports and verified skipping. CPU reference
integer kernels exercise score streaming for every architecture. A real saved
OPS-SAT FP32 parent has also been reevaluated locally in a scratch destination.
These checks do not claim the new V20 campaign has already run on the A100.
