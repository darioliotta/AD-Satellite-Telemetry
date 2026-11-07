# V15 — History controls and SMD replications

This campaign asks whether a model uses the values in its past, their temporal
order, or mainly the current value. OPS-SAT is the main investigation; SMAP and
MSL are frozen-model comparisons. The two SMD training runs are last.
The plan is declared in `plan.json`; original sources and v10–v14 results are
read, never rewritten. No new architecture is introduced.

## Plan and execution order

| Order | Work | Jobs | Training |
|---|---|---:|---|
| 1 | OPS-SAT frozen history controls | 40 | Existing v13 weights |
| 2 | OPS-SAT controlled training | 100 | 50 epochs per model |
| 3 | SMAP frozen history controls | 40 | Existing v13 weights |
| 4 | MSL frozen history controls | 40 | Existing v13 weights |
| 5 | SMD benchmark replicas | 2 | 500 epochs per model |

**222 jobs, 102 new trained models.** Every new model is evaluated immediately
after its training. The campaign does not wait until all training has finished.

All frozen and training context comparisons include **all four architectures**:

| Presentation name | Code identifier |
|---|---|
| TranAD+ | `reference` |
| One-decoder (one decoder reused in the two passes) | `shared_decoder` |
| One-pass | `single_pass` |
| Encoder-only | `encoder_only` |

They use five model seeds **2026–2030**, and two window lengths: OPS-SAT/SMAP **5 and 10**, MSL
**25 and 50**. FF, batch size, learning rate and splits follow v13. Only OPS-SAT
receives additional training; the larger datasets reuse their existing weights.

### Frozen controls: 120 checkpoints × 6 inputs = 720 evaluations

For each checkpoint, evaluate:

1. `original`: original window, with exact baseline prediction-count checks.
2. `shuffle_1101`: randomly permute the past of each individual window.
3. `shuffle_2202`: second independent permutation draw.
4. `shuffle_3303`: third independent permutation draw.
5. `reverse_past`: reverse only the past.
6. `repeat_current`: replace the past with repetitions of the current value.

The current sample and reconstruction target are always identical. A permutation
moves a full multivariate row, preserving simultaneous feature associations.
Values never move between windows. OPS-SAT windows remain inside their original
fragments. Native SMAP/MSL keep the audited author window/concatenation protocol;
the saved real-point mask identifies extra padded target rows for later analysis.

**Both alpha and threshold stay fixed at the parent checkpoint's BF200 or POT
selection.** A perturbed window cannot get a better classification merely by
selecting another alpha or threshold. V14 used fixed alpha=0.5 and a training
quantile for these interventions; V15 measures the BF200/POT endpoints instead.
The three permutation draws are repeated perturbations, **not three additional
training seeds**.

### OPS-SAT training: 100 models

| Training input | Windows | Variants | Model seeds | Models |
|---|---|---:|---:|---:|
| One sample, no history | 1 | 4 | 5 | 20 |
| Shuffled past | 5, 10 | 4 | 5 | 40 |
| Current value repeated through the window | 5, 10 | 4 | 5 | 40 |

FF=8, one layer, batch=256, learning rate=0.001, 50 epochs. Training fragments,
validation fragments, test fragments, channel scalers and split seed=2026 are
unchanged from v13. Optimization still uses nominal training fragments only.

Each shuffled window receives one deterministic permutation (private seed 1101),
reused throughout training. This tests learning without consistent past order;
it is **not fresh shuffling at every epoch**. The same permutation realization
is used across the five model seeds and all four architectures.

Models trained on shuffled/repeated inputs are tested twice: with the matched
input transformation and with ordered test inputs. Their POT training reference
scores always come from the actual transformed training input. The one-sample
models have one test condition. This gives **180 OPS-SAT training evaluations**.

These controls separate reliance on history in an already trained model from
its ability to learn under a modified input protocol. Frozen perturbations can
be out of distribution; they alone do not establish a physical cause. W=1 still
reconstructs the observed current value, as the original model does: it is not
a next-step forecasting experiment.

### SMD: two predeclared seed replicas

Use seeds **2 and 3**, **500 epochs**, W=20, **effective FF=16**, **one layer**,
batch=8192, learning rate=0.0001 and q=0.001. Keep scaling and padding enabled,
batch shuffling disabled, zero data-loader workers and no downsampling.
This copies the existing benchmark **`smd_plus_ws20_dff16`** in `results/legacy/SMD/`.
FF=16 is the absolute width, not `16x` (which would be 608 for 38 features).

The profile is copied from the archived training metadata and cross-checked
against checkpoint `hp_dict` and the actual encoder FF weight shape `[16, 76]`.
`smd_benchmark.json` records that profile, the artifact paths and their SHA256
hashes. The runner rejects an SMD plan that changes those hyperparameters.
Identifiers, output locations and random seeds distinguish the replicas.

| Existing comparison | POT F1 |
|---|---:|
| Authors' checkpoint log | 71.2906% |
| Author weights evaluated locally | 71.7068% |
| Our archived 500-epoch W20/FF16/L1 benchmark | 64.0088% |
| Our separate 100-epoch W20/FF76/L5 training, seed 1 | 67.0386% |

This already supports numerical consistency of evaluation for the author
weights; it does not prove that the retraining gap is entirely seed variation.
The two replicas repeat the archived 500-epoch configuration, without changing
POT or its q parameter. The 100-epoch/FF76/L5 configuration is a separate
experiment and is not the configuration scheduled here.
Report both replicas and their spread, rather than picking a favorable seed.
The author BF200 result for this specific W20 checkpoint is 82.1742%; the
dataset-level best BF reported across configurations is a different comparison.

## Metrics and saved data

Primary methods: **Best F1 with 200 linear thresholds**, **POT**, and the
authors' **AUC-PA** on binary predictions after point adjustment. No dense
threshold search is executed. Two-output models have the original 21-alpha
grid; one-pass and encoder-only each have one output and no alpha sweep.

Training evaluations use the same test-label oracle selection convention as
the existing benchmark. Frozen evaluations lock that selection to the original
parent result. These protocols are distinguished explicitly in the tables.
Unadjusted metrics at the same thresholds are also saved for inspection.

All output is under `results/v15/`:

```text
v15/
  campaign_latest.json
  _campaign_logs/
  tables/per_seed.csv
  tables/mean_std.csv
  frozen/{OPS-SAT,SMAP,MSL}/{variant}/{configuration}/seed_*/
    manifest.json, config.json, status.json, progress.json, run.log
    data_inventory.json, metadata.npz, metrics.json
    original.{json,npz}, shuffle_1101.{json,npz}, ...
  train/OPS-SAT/{variant}/{configuration}/{input_condition}/seed_*/
    checkpoints/resume.ckpt, checkpoints/final.ckpt
    epochs.jsonl, manifest.json, config.json, data_inventory.json
    status.json, progress.json, run.log, metrics.json
    outputs_matched.npz, scores_matched.npz
    outputs_ordered_test.npz, scores_ordered_test.npz  # if applicable
  smd/legacy_ws20_ff16_l1_e500/seed_{2,3}/
    checkpoints/, epochs.jsonl, manifests, metrics, train/test outputs and scores
```

Frozen arrays include BF/POT continuous scores, direct alarms and per-feature
squared reconstruction errors. Shared alpha/output errors are stored once;
each method names its array in the condition JSON. Metadata records labels,
OPS-SAT segment IDs/channels and native real-point masks. This supports later
sensor, anomaly/nominal error, feature-contribution and PA analyses without
running the models again.

New training runs save **both reconstruction outputs** for reference and
one-decoder, and **one output** for one-pass and encoder-only,
targets, all 21 score rows where applicable, labels and validation outputs for
OPS-SAT. Arrays use numeric/string dtypes, not pickled Python objects. Metrics
are JSON, with CSV summaries. Tables show **seed means and sample standard
deviations**, not maxima. Failed POT fits remain unavailable and are counted.

Expect several GB of compressed arrays; leave approximately **40 GB free**
for the expanded campaign. SMD has 500 epochs per replica. The earlier timing
of 56 seconds per epoch belongs to the different five-layer/FF76 model and
does not provide a reliable duration for this one-layer/FF16 configuration.
Epoch bars estimate remaining training time from the current run. A complete
overnight finish is not guaranteed; interruption and restart are supported.

## Copy from the Mac

From the local project root:

```bash
rsync -avh --partial --progress \
  --exclude='__pycache__/' \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  tranad_v15 \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/
```

Only the new directory is copied. Existing `tranad_plus/`, `tranad_variants/`
and `tranad_v14/` must remain alongside it on the server. The preflight verifies
their numerical sources, parent result hashes and data before training begins.
Required inputs are OPS-SAT `segments.csv`, processed SMAP/MSL/SMD, and the
120 completed v13 parent result folders (including their final checkpoints,
scores and metadata). All are already present in the local project.

## Start on the server

Connect and create the tmux session:

```bash
ssh -J atuscano@gate.pd.infn.it atuscano@lxa100
tmux new -s v15
```

Inside tmux:

```bash
cd /home/dliotta/AD-Satellite-Telemetry
source ad-satellite-telemetry/bin/activate
bash tranad_v15/run_v15.sh
```

The launcher performs a read-only preflight, then runs the complete plan.
Bars display the current configuration, seed, epoch, batch, alpha and local ETA.
The global job bar shows elapsed time: a uniform-job ETA would be misleading
because SMD is much slower than the small OPS-SAT controls.

Detach with **Ctrl+B, then D**. Reattach with `tmux attach -t v15`.
To resume after interruption, run **the same launch command**. Checkpoints save
model, Adam, scheduler and RNG state after every completed epoch. An interrupted
epoch is repeated from the previous durable checkpoint. Verified completed jobs
are skipped; frozen jobs also resume from completed condition files.

Optional commands, from the project root in the activated environment:

```bash
python -B -m tranad_v15.run --dry-run
python -B -m tranad_v15.status
python -B -m tranad_v15.summarize
```

Subsets use the same launcher and output folders, for example
`bash tranad_v15/run_v15.sh --stage train` or `--stage smd`.
If SMD input is missing, `--stage frozen` and `--stage train` can still be run
separately; the default full preflight deliberately reports missing requirements
before consuming GPU time. Default device is `cuda:0`, FP32, TF32 disabled,
standard Adam, automatic GPU-resident batches. There is no mixed precision or
batch-size substitution. Eight CPU threads avoid overhead in the small matrix
operations; this is an execution setting, not a new hyperparameter grid.

## Copy the results back

From the Mac project root:

```bash
mkdir -p results/v15
rsync -avh --partial --progress \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/results/v15/ \
  results/v15/
```

This copies checkpoint files, arrays, tables and logs together. Existing local
v13 parent weights remain the weights for frozen evaluations. No intermediate
compressed archive is created.

## Local checks

```bash
python -B -m unittest tranad_v15.test_controls -v
python -B -m tranad_v15.run --check --device cpu --cpu-threads 1 --no-progress
```

The tests include unchanged targets, within-window/feature permutation integrity,
independent RNGs, native window alignment, archived OPS-SAT BF200 parity,
BF200/POT-only evaluation, exact interrupted-versus-continuous training, artifact
checks and a complete synthetic CLI train/evaluate/resume/skip sequence.
Preflight covers all 222 local jobs. A full six-condition frozen
evaluation has also passed on a real OPS-SAT encoder-only checkpoint on CPU,
including exact baseline confusion-count parity. A100 execution of the new
campaign must be checked by the server run; local tests are not a GPU timing
measurement.
