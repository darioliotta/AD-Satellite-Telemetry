# V19 — Controlled generalization and compression study

One current source tree: `tranad_plus/`. Default results:
`results/v19/controlled_study/`. The earlier OPS-only V19 audit and all historical
campaigns remain untouched. This revision uses schema 2 and a new result root;
never overwrite an unfinished schema-1 experiment with these sources.

The hypothesis is that excess capacity or excessive training can contribute to
some apparent gains from compression. This study can support or weaken that
hypothesis. It cannot guarantee better F1, a monotonic compression trade-off,
or establish how much the authors' own archived runs overfit.

## Three phases, launched separately

### 1. Diagnostic controls — default, 60 training trajectories

| Dataset | Architecture/configuration | Seeds × arms | Trajectories |
|---|---|---|---:|
| OPS-SAT | All four architectures; reference W10/FF8 and compact W5/FF4 | 3 × 2 | 48 |
| SMAP | TranAD+ reference; W10/FF50/L1/BS8192/LR0.001 | 3 × 2 | 6 |
| MSL | TranAD+ reference; W50/FF880/L2/BS2048/LR0.0001 | 3 × 2 | 6 |

Seeds are fixed in advance: 2026, 2027, 2028. Arms are original Adam with
weight decay 0 and identical Adam with weight decay 1e-4 (coupled L2, not AdamW).
The existing dropout 0.1 is unchanged. Preserve original loss, ordered batches
and StepLR (every 5 epochs, gamma 0.9). Use FP32, TF32 disabled.

OPS-SAT settings: BS256, LR0.001, L1, q0.01, original horizon 50, audit horizon
100. SMAP q0.07, original horizon 50, audit horizon 100. The MSL anchor has q0.01,
original horizon 100 and audit horizon 150. Do not relabel its settings as the
older L1/FF110 configuration.

SMAP parameter provenance is the archived
`results/legacy/SMAP/smap_plus_ws10_training_20261021T074354533693Z_d783b339.json`.
MSL matches `verify_author_checkpoints.py`, case `msl50`, author checkpoint
`Aug-20-24_03-34-10-129705.ckpt`. The plan records both sources. These are
benchmark-parameter anchors retrained with an explicit holdout, not an unchanged
replication of the authors' training protocol.

### 2. Matched compression study — after reviewing phase 1

`--stage compression` selects **144 trajectories**: four architectures, two
profiles, three seeds, two arms, on all three datasets. The 48 OPS-SAT jobs are
identical to phase 1 and verified/skipped, leaving **96 new native trajectories**.

These use the established V13 study grid, explicitly labelled `study_grid`:

| Dataset | Reference W / FF | Compact W / FF | BS | LR |
|---|---|---|---:|---:|
| OPS-SAT | 10 / 8 | 5 / 4 | 256 | 0.001 |
| SMAP | 10 / 50 | 5 / 25 | 1024 | 0.001 |
| MSL | 50 / 110 | 25 / 55 | 128 | 0.0001 |

All have L1 and fixed/audit horizons 50/100. This family intentionally differs
from the benchmark anchors. Never pool these families when attributing a change
to regularization. Primary comparisons pair the same dataset/profile/seed.

`--stage all` is the **156 unique trajectories** in phases 1 and 2, not 204.
The default launches phase 1 only; decide whether to extend after examining it.
`--all-configs` additionally includes window-only and FF-only profiles, allowing
one-factor tests under the new split. It is optional, not the default: expanding
all phases would select 300 trajectories. Existing historical W/FF results
remain available, but their full-training protocol is not silently pooled with
new holdout results.

### 3. Integer PTQ of the controlled checkpoints — no training

After completing a selected stage, use `--quantize-only` with the same stage.
The existing V17 integer runtime evaluates W8A8 and W4A4 for each retained
checkpoint. CUDA/Triton preflight verifies actual INT8×INT8→INT32 instructions.
W4A4 uses operands restricted to four-bit values on INT8 instructions; it is
not native INT4 execution. Softmax, sigmoid, residuals, scaling and other
explicitly documented operations remain floating point.

The FP32 parent checkpoints are sufficient to regenerate integer models.
Do not archive duplicate integer weight models: retain metrics, resident buffer
bytes, kernel audit and hashes of parents. This permits checking whether PTQ
changes after early stopping/regularization without duplicating model archives.
No QAT, FP16, new training or silently missing parents. Missing FP32 jobs are
reported as pending and the command exits nonzero. `--with-integer` can evaluate
integers immediately after each completed trajectory, but is optional.

## Data: common nominal validation criterion, dataset-specific labels

OPS-SAT retains the exact V13 fragment split: 1,019 nominal optimization
fragments, 575 validation fragments (254 nominal and 321 anomalous), and 529
official test fragments. Windows remain inside fragments; per-channel MinMax
scalers fit only optimization fragments. The split seed stays 2026 for every
model seed. Validation anomalies never enter the optimizer or scaler.

SMAP/MSL reserve the last 20% of **each official training track** as nominal
validation. Split the raw processed series before fitting scalers, padding or
constructing windows. Training labels must all be nominal; otherwise fail.
Per-feature MinMax fits only the first 80%; no validation/test fitting or
clipping. Exact half-open point ranges and fitted minima/maxima are saved.

The original global concatenation/per-track padding convention is applied
separately inside optimization and validation. Thus no window crosses their
boundary. The official test retains all tracks, points, labels and upstream
padding/alignment convention. Test normalization uses the new optimization-only
scaler; normalized numbers can differ from the historical full-training run.

This changes the training protocol even though the official test is untouched.
Compare new regularized controls against the new baseline trained on the same
80%, not directly against a historical full-training result. Existing native
checkpoints trained on 100% cannot be reused as an unseen-validation baseline.

## Checkpoint selection

The **common primary monitor** is nominal-validation MSE of the last output,
evaluated with dropout off: second output for two-output models, only output
otherwise. Both decoder errors are logged separately. Record the same fixed
metric on optimization data. Preserve all RNG states around diagnostics, so
monitoring does not alter subsequent dropout or batch-order randomness.

The original training objective changes its two-output weighting with epoch.
It is logged but is not directly comparable to this fixed validation monitor.

Predeclared early stopping: patience 10, earliest stopping epoch 20, minimum
relative improvement 0.1%. Freeze the selected checkpoint at the first stopping
decision. Default **audit mode** then continues to the fixed full horizon to
observe what would happen without stopping. This does not save compute time;
it gives early/fixed/final comparisons from one trajectory.
`--stop-on-patience` actually halts and therefore cannot provide the complete
post-stop trajectory. Changing mode requires a separate output root.

OPS-SAT has one additional checkpoint, `validation_detection`: maximum BF200
F1 with fragment-aware PA on **validation**, checked at epoch 1, every 5 epochs
and the final epoch. Alpha is fixed at 0.5 for two-output models, inapplicable
for one output. This secondary criterion is declared in advance; it is not the
same as early stopping on nominal MSE. Record its raw no-PA F1 and continuous
ROC-AUC as interpretation controls. Test labels never select this checkpoint.

Evidence consistent with overfitting: evaluation-mode training MSE keeps falling
while nominal-validation MSE rises, and the predeclared earlier checkpoint or
regularization improves held-out detection across paired seeds. A final gap
alone is insufficient, because tracks/fragments can have different distributions.
Lower nominal MSE also need not improve detection: anomaly reconstruction can
improve too. Report negative findings and seed dispersion; do not select only
the best seed, architecture or threshold protocol to support the hypothesis.

## Detection metrics and calibration

For every retained checkpoint:

- `test_paper_oracle`: original BF200 and POT plus 21-alpha search where
  applicable, explicitly using test labels for selection, as in the benchmark.
  No dense threshold grid. This preserves metric-protocol continuity, not an
  identical training-data protocol.
- `test_validation_calibrated`: on OPS-SAT, alpha/threshold chosen on labelled
  validation and frozen on test. Native validation has no anomalies, so
  supervised BF/F1 calibration is **not available**; it is not fabricated.
  A separate native POT endpoint uses predeclared alpha=0.5 (or no alpha for
  one output), nominal training fit and nominal validation stream, frozen on
  test. That endpoint does not optimize F1 on validation.

F1/precision/recall, confusion counts, alpha and threshold are retained.
AUC-PA means binary ROC-AUC after PA, following the authors. Continuous-score
ROC-AUC is explicitly named `roc_auc_continuous_scores`. Raw/no-PA F1 is recorded
at the same selected alpha/threshold, not from another dense/raw optimization.
OPS-SAT labels remain weak segment labels expanded to points.

## Compact artifacts and recovery

Retain only:

- early-stopped weights, weights at the original horizon (`fixed_50` or
  `fixed_100`), final weights, plus the small OPS detection-selected snapshot;
- one rolling recovery file containing current/best weights, Adam, scheduler,
  RNG and selection state; no per-epoch weight archive;
- per-epoch scalar train/validation curves, losses, LR, times and GPU peak memory;
- input/source hashes, split ranges/IDs, configuration, environment and logs;
- endpoint metrics, parameter/state bytes and analytic inference MAC counts;
- optional integer endpoint reports and resident buffer sizes, regenerated from
  FP32 checkpoints. No duplicate integer weight exports.

No reconstructions, all-alpha score arrays, predictions or duplicate labels are
written. Inference timing includes transfer and is not a pure kernel-latency
benchmark. MAC estimates exclude optimizer/backward/activations and are not
measured training FLOPs. GPU memory is explicitly measured only on CUDA.

Preflight computes a conservative storage allowance from the actual selected
model sizes. Default phase 1 is approximately **440 MiB**, not the earlier
OPS-only 34 MiB estimate. The declared planned-storage guard is 1.5 GiB; a plan
above that allowance fails before training. Estimates are not a hard filesystem
quota. Read the check output before selecting optional large expansions.

Resume with the identical command, source, plan and environment. Atomic recovery
retains best weights consistently across interrupted writes; epoch telemetry is
reconciled to the committed checkpoint. Verified completed jobs are skipped.
Never run two processes into this output root concurrently.

Incremental tables include per-seed metrics and means/sample standard deviations,
paired early-vs-fixed/final differences, paired weight-decay-vs-baseline
differences, and optional integer endpoints. Fractions are used, not percentages.
Missing POT fits remain unavailable, with available-seed counts in aggregates.

## Local-to-server transfer

From the local repository root:

```bash
rsync -avh --checksum --partial --progress \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  tranad_plus/ \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/tranad_plus/
```

No datasets/results are transferred or deleted. If an old source-bound V19 job
is still running, finish it before replacing sources. Its result root is
preserved and separate from the new `controlled_study` root.

On the server:

```bash
tmux new -s generalization
```

Inside tmux:

```bash
source /home/dliotta/AD-Satellite-Telemetry/ad-satellite-telemetry/bin/activate
cd /home/dliotta/AD-Satellite-Telemetry/tranad_plus
bash run_regularization.sh --stage diagnostic
```

The launcher checks data, source hashes, configurations and storage before
training. Use Ctrl-B then D to detach; `tmux attach -t generalization` to return.
From `tranad_plus/`:

```bash
python -B run_campaign.py regularization --stage diagnostic --status
python -B run_campaign.py regularization --stage diagnostic --dry-run
```

After reviewing phase 1, optionally:

```bash
bash run_regularization.sh --stage compression
```

Integer evaluation of phase 1, no training:

```bash
bash run_regularization.sh --stage diagnostic --quantize-only
```

Use `--stage compression --quantize-only` after phase 2. Dataset/seed/arm filters
are available. These do not change individual job provenance. Re-running a
filter that includes completed jobs verifies and skips them.

Return results, from the local repository root:

```bash
mkdir -p results/v19/controlled_study
rsync -avh --checksum --partial --progress \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/results/v19/controlled_study/ \
  results/v19/controlled_study/
```

## Verification

From the repository root:

```bash
python -B -m unittest tranad_plus.campaigns.regularization.test_regularization -v
```

Tests cover first-stop freezing, nominal-only monitoring, original-trajectory
parity, exact interruption/resume, training/test independence, validation
calibration independence from test labels, no native supervised F1 calibration
on one-class validation, native split/scaler leakage and official-test alignment,
and full synthetic campaign restart/evaluation/table/hash-verified skipping.
CPU tests do not claim that this new campaign has already run on an A100.

Method references:
https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html
https://docs.pytorch.org/docs/2.8/generated/torch.nn.Dropout.html
https://github.com/yyexela/TranADPlus
