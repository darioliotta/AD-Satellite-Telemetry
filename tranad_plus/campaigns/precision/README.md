# V16 compact — PTQ, then the original-author SMD benchmark

This campaign closes the precision study without another large training grid.
The default launcher runs **240 PTQ jobs**, followed by **two SMD training and
evaluation jobs**. It does **not** run QAT or additional FP32 fine-tuning.
Existing datasets, v13 checkpoints and all earlier results remain unchanged.

## Scientific comparison

The 240 parents are the already completed v13 50-epoch checkpoints:

- OPS-SAT, SMAP and MSL;
- TranAD+, one-decoder (`shared_decoder`), one-pass (`single_pass`), encoder-only;
- four W/FF combinations per dataset;
- all five seeds, 2026–2030. No best-seed selection.

| Dataset | Profile | W | Effective FF | Layers | Batch | Base LR | POT q |
|---|---|---:|---:|---:|---:|---:|---:|
| OPS-SAT | reference | 10 | 8 | 1 | 256 | .001 | .01 |
| OPS-SAT | window_only | 5 | 8 | 1 | 256 | .001 | .01 |
| OPS-SAT | ff_only | 10 | 4 | 1 | 256 | .001 | .01 |
| OPS-SAT | compact | 5 | 4 | 1 | 256 | .001 | .01 |
| SMAP | reference | 10 | 50 | 1 | 1024 | .001 | .07 |
| SMAP | window_only | 5 | 50 | 1 | 1024 | .001 | .07 |
| SMAP | ff_only | 10 | 25 | 1 | 1024 | .001 | .07 |
| SMAP | compact | 5 | 25 | 1 | 1024 | .001 | .07 |
| MSL | reference | 50 | 110 | 1 | 128 | .0001 | .01 |
| MSL | window_only | 25 | 110 | 1 | 128 | .0001 | .01 |
| MSL | ff_only | 50 | 55 | 1 | 128 | .0001 | .01 |
| MSL | compact | 25 | 55 | 1 | 128 | .0001 | .01 |

Each PTQ job checks its FP32 baseline, then evaluates FP16, INT8 and INT4:
**960 format evaluations**, without training. The two SMD evaluations bring
the total to 962 format evaluations inside **242 jobs**, not 962 training jobs.
Within PTQ the fixed order is OPS-SAT, SMAP, MSL; compact/window_only/ff_only/
reference; encoder-only/one-pass/one-decoder/TranAD+; then seeds 2026–2030.
SMD starts only after the PTQ block. Each SMD seed is evaluated before the next
seed is trained. The launcher stops on unexpected errors, so a failed PTQ
experiment is not silently passed over to begin SMD.

The original data splits, normalization, channel handling, padding and targets
are unchanged. OPS-SAT boundaries remain fragment-aware. No test data fit the
quantizers, and no new downsampling or augmentation is introduced.

## Precision and evaluation definitions

FP16 uses genuinely half-precision weights/inputs and model execution. Outputs
are cast back to FP32 and scored against the original FP32 targets.
INT8 and INT4 are **weight-only**, symmetric, per-tensor quantization, with
scale `max(abs(parent_weight)) / (2**(bits-1)-1)`. All unique learned matrices,
including attention QKV/output projections and the reconstruction head, are
covered. Biases/buffers remain FP32. Shared weights are stored once with aliases.
INT8 codes are actual int8 arrays; INT4 packs two codes per byte.
Exported files are reloaded before evaluation.

**INT8/INT4 inference is simulated using dequantized FP32 weights.** This tests
accuracy and packed storage, not integer-kernel speed or resident integer-model
memory. FP16 is a distinct actual execution comparison. A100 results do not
establish satellite-hardware performance.

Every format reports original **BF200 and POT**, with point adjustment and
**AUC-PA**: the ROC-AUC of binary adjusted predictions, equal to balanced
accuracy. BF200 uses 200 thresholds; two-output models test 21 alpha values.
Selection uses test labels as in the authors' benchmark and is not deployment
calibration. There is **no dense BF search**. We also report raw metrics before
PA and a paired control with the parent alpha/threshold locked. FP32 baseline
scores must match the parent within the declared numerical tolerance. The
original confusion counts must be reproduced exactly from the stored parent
scores, which guards against a changed metric protocol. CPU/GPU rounding can
move a score across a fixed threshold: those boundary flips and resulting count
changes are explicitly recorded, not hidden or assigned to quantization.
All precision deltas use the newly evaluated FP32 baseline on the same device.

## Compact storage: what is retained

- Saved FP32/FP16/packed INT8/packed INT4 models and their loading schema.
- BF/POT precision, recall, F1, AUC-PA, raw metrics, confusion counts, selected
  alpha/threshold, all-alpha scalar search results, and score distribution summaries.
- Parent-locked metrics and score/alarm drift relative to the v13 parent.
- Tensor/file/resident model sizes and per-tensor quantization error/coverage.
- Raw 100-forward latency samples after 10 warmups; median, mean, std, p95;
  inference/threshold/export/load times and throughput.
- Analytical matmul MAC/FLOP counts and per-operator profiler JSON.
- Process/CUDA memory information, GPU power/temperature snapshots, environment,
  source/data hashes, configuration, per-job logs and live progress.
- For SMD: epoch losses/LR/times/resources, small batch-loss files, one latest
  recovery checkpoint overwritten atomically each epoch, and one final checkpoint.
  Recovery retains optimizer, scheduler and RNG states.
- A small 45-window OPS-SAT synthetic response probe (kilobytes), not physical
  fault labels. This is a model-sensitivity check, not proof of a failure mechanism.

**Not archived:** full decoder reconstructions, duplicated targets/labels,
pointwise predictions, or pointwise score grids for every split/alpha. These
are computed in memory, summarized, then released. They can be regenerated by
rerunning evaluation from the exported model and the pinned data/source files.
If evaluation is interrupted, completed formats are verified/reused; the
incomplete format is recomputed. There is no need to save large arrays to resume.

MAC/FLOP counts have a declared scope excluding elementwise/softmax, backward
and optimizer work. Profiler FLOPs cover supported operators only. GPU power
snapshots are not integrated energy measurements.

## Space checks

The target is a small archive, with **3 GiB as a stop budget**, not a predicted
3 GiB payload. `--check` measures the actual serialized model sizes in memory
for every selected parent (without temporary disk writes), estimates SMD recovery/final checkpoint
sizes, and adds a generous 512 KiB per format plus 512 KiB per job and 32 MiB
for diagnostics/logs/tables. The conservative total is printed before launch.
It excludes existing datasets/v13 parents and temporary RAM usage.
Native input arrays are checked in bounded memory-mapped chunks during
preflight; full normalized training datasets are created only for execution.
OPS-SAT input hashes must match the already validated parent manifest.

The runner checks output size before and after each job and stops if it has
crossed the budget; it never deletes results. This is a between-job guard,
not an OS disk quota: the last job can take output slightly above the limit.
Free space is also checked before additional work. A later resume repeats
these checks. Restarting the same command reuses verified completed work.
Do not run this compact revision over results produced by the earlier heavy
V16 code; differing signatures are intentionally rejected.

Measured preflight on 7 November 2026, using all 240 local parents and both
SMD replicas (decimal MB/GB below):

| Component | Size |
|---|---:|
| Actual exported model files, all 962 formats | about 175 MB |
| Estimated SMD recovery and final checkpoints | about 38 MB |
| Generous allowance for diagnostics, manifests, logs and tables | about 665 MB |
| Conservative complete archive estimate | **about 0.88 GB** |

This is a measured payload plus an overhead estimate, not a promised exact
final file size. The server preflight recomputes it. A real OPS-SAT encoder-only
PTQ job completed all four formats and occupied about 0.37 MB of logical file
content; larger models/datasets have different diagnostic sizes.

## SMD: exact author POT configuration

`author_smd.json` pins the metadata and SHA256 of
`Aug-20-24_00-59-24-881901.ckpt`, associated with paper POT F1 **71.2906%**.

| Epochs | W | Effective FF | Stack layers | Batch | LR | POT q | Seeds |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 20 | 76 (=2×38) | 5 | 8192 | .0001 | .001 | 2, 3 |

These are two seed replicas of **one configuration**. Scaling/padding enabled,
shuffle false, zero workers, no downsampling, faithful FP32/standard Adam.
It is the configuration already used for
`papercheck_SMD_pot_w20_ff76_l5_bs8192_s1`. The old local 500-epoch/FF16 run
and the authors' 500-epoch/FF608 run are different experiments.
SMD results remain separate under `results/v16/smd/`; no legacy file is replaced.

Author sources: [training log](https://github.com/yyexela/TranADPlus/blob/main/Results/TranADPlus/Aug-20-24_00-59-24-881901_train.txt),
[evaluation log](https://github.com/yyexela/TranADPlus/blob/main/Results/TranADPlus/Aug-20-24_00-59-24-881901_eval.txt).
Implementation references: [fake quantization](https://docs.pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html),
[profiler](https://docs.pytorch.org/docs/2.8/profiler.html).

## Copy and launch

From the Mac project root:

```bash
rsync -avh --partial --progress \
  --exclude='__pycache__/' \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  tranad_plus/campaigns/precision \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/
```

On the server:

```bash
ssh -J atuscano@gate.pd.infn.it atuscano@lxa100
tmux new -s v16
```

Inside tmux:

```bash
cd /home/dliotta/AD-Satellite-Telemetry
source ad-satellite-telemetry/bin/activate
bash tranad_plus/campaigns/precision/run_v16.sh
```

This single command checks all prerequisites and storage, runs PTQ, then trains
and evaluates SMD seed 2, then trains and evaluates seed 3. No new dependency
or PyTorch upgrade is required. Existing `tranad_plus/`, `tranad_plus/core/`,
`tranad_plus/campaigns/exploration/`, `tranad_plus/campaigns/temporal/`, datasets and v13 parents must remain available.
QAT is not accepted as a stage, so it cannot accidentally launch.

Terminal headers show job index, phase, dataset, architecture, configuration,
seed, W, FF, layers and batch. Three rows show campaign progress, current
split/alpha/epoch, and current inference/training batch. Local bars show ETA;
the global bar shows elapsed time because PTQ and SMD costs differ greatly.
Verbose loader/profiler diagnostics stay in per-job logs. Detach with
Ctrl+B then D; reattach using `tmux attach -t v16`.

After interruption, rerun the identical command. SMD resumes from the latest
completed epoch, including optimizer/scheduler/RNG; an unfinished epoch repeats.
Do not edit source files or change software versions while a campaign is active.

Useful commands in the activated environment:

```bash
python -B -m tranad_plus.campaigns.precision.run --dry-run
python -B -m tranad_plus.campaigns.precision.run --check --device cpu --cpu-threads 1 --no-progress
python -B -m tranad_plus.campaigns.precision.status
python -B -m tranad_plus.campaigns.precision.summarize
```

`--stage ptq` or `--stage smd` restricts the launch. For SMD use `--dataset SMD`
if filtering; `--seed` filters v13 parent seeds only. SMD seeds are declared in
`plan.json`. `--only` takes an exact identifier printed by `--dry-run`.

## Results and transfer

```text
results/v16/
  campaign_latest.json
  environment.json
  _campaign_logs/
  tables/
    per_seed.csv
    mean_std.csv
    paired_deltas.csv
    paired_mean_std.csv
    unavailable_formats.csv
  ptq/{dataset}/{variant}/{configuration}/seed_*/
    config.json, manifest.json, metrics.json, status.json, run.log
    models/{fp32,fp16,int8,int4}.npz
    evaluation/{format}/result.json, operators.json, latency_samples.npz
  smd/author_pot/seed_{2,3}/
    checkpoints/{resume,final}.ckpt
    models/fp32.npz
    epochs.jsonl, epoch_resources.jsonl, batch_losses/, evaluation/fp32/
```

Tables report seed means and sample standard deviations. Paired PTQ deltas
compare the same parent/seed/configuration against FP32, including compression
ratios. Unavailable precision kernels/nonfinite converted outputs are explicit
missing formats; remaining formats continue. FP32 parity failures and unexpected
exceptions stop the default launcher. POT fitting failures remain unavailable,
not replaced with a different threshold method.
For SMD POT, per-seed rows also include the published reference F1 and the
difference in percentage points. The source reference is pinned in each SMD
report, so its configuration and comparison remain inspectable offline.

From the Mac project root, copy the complete compact archive:

```bash
mkdir -p results/v16
rsync -avh --partial --progress \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/results/v16/ \
  results/v16/
```

The same rsync command can be repeated after an interrupted transfer.

## Validation

```bash
python -B -m unittest tranad_plus.campaigns.precision.test_precision -v
```

Regression tests cover all architectures/attention matrices, packed INT4 round
trips and aliases, unchanged FP32 parents, FP16 storage, compact outputs,
original SMD configuration, native loss capture, storage guards, seed summaries,
numerical boundary-flip versus metric-protocol checks, and the existing internal
fine-tuning helper (not called by this campaign).
GPU execution/timings must be checked on the A100; local validation uses CPU.

Local validation completed on 7 November 2026: **17 regression tests passed**;
all **242 jobs passed preflight**, with conservative storage estimate 0.8177
GiB (about 0.88 GB). Real OPS-SAT encoder-only and two-output TranAD+ PTQ jobs
completed all four formats on CPU, with compact outputs and parent verification.
Completed-job reuse was also checked. These are implementation checks, not a
replacement for the full A100 precision study.
