# TOAST v12 — Complete v10 architectural grids to 6 × 5 (126 new runs)

## Current source tree

All current implementation lives in this directory. Campaign versions label
results, not sibling code copies. See [CURRENT_CODE.md](CURRENT_CODE.md) for
the consolidated layout, server installation and the integer campaign.

For the current V19 validation/early-stopping/regularization study, use
[campaigns/regularization/README.md](campaigns/regularization/README.md).
The default diagnostic stage selects 60 trajectories across OPS-SAT, SMAP and
MSL; later compression and integer reevaluation stages are separate commands.

For V20 score-separation, simple-detector and current-sample controls on all
three datasets, see [campaigns/mechanisms/README.md](campaigns/mechanisms/README.md).
Launch `bash run_mechanisms.sh`: 432 existing-model evaluations, six baselines
and 72 new paired trainings, with compact artifacts under `results/v20/mechanisms/`.


A **single replacement `tranad_plus/` source tree**: extract the ZIP into
`/home/dliotta/AD-Satellite-Telemetry/`. Dataset files and existing v10/v11
results live outside this folder and are not included in this archive.

## Experimental design

The v10 campaign contains **16 architectures per dataset**, 4 windows × 4 FF
widths, with 3 seeds (144 runs). The v12 campaign **adds exactly the 14 missing
cells per dataset** to make the complete 6 × 5 grid. The old 16 cells are
**NOT rerun**. Thus 14 × 3 datasets × 3 seeds = **126 new runs**, 50 epochs each;
270 study runs when combined with v10.

| Dataset | Existing W (v10) | New W (v12) | Existing FF (v10) | New FF (v12) |
|---|---|---|---|---|
| OPS-SAT | 3,5,8,10 | **4,6** | 2,4,6,8 | **3** |
| SMAP | 3,5,8,10 | **4,6** | 13,25,38,50 | **32** |
| MSL | 13,25,38,50 | **32,44** | 28,55,83,110 | **69** |

For each dataset the new cells are exactly:
`new_W × existing_FF` (8) + `existing_W × new_FF` (4) +
`new_W × new_FF` (2) = **14**. There are no duplicate v10 cells.
All dimensions are **effective integers**, not rounded percentages.

## Scientific training/evaluation compatibility

- **Keep v10 hyperparameters**: OPS-SAT BS256, LR0.001;
  SMAP BS1024, LR0.001; MSL BS128, LR0.0001.
- 50 epochs, Adam standard, CUDA precision `faithful` (FP32, no TF32),
  seeds **2026/2027/2028**, unchanged model, two-decoder loss, scheduler.
- OPS-SAT **restores the exact v10 plan split**: `validation_fraction=0.2`,
  `split_seed=2026` (normal-fragment holdout + official-train anomalous
  fragments for validation; official test untouched). **The prior 45-run v12
  incorrectly used `0.0`; do not use that ZIP.**
- SMAP/MSL use their original official training/test track splits, as in v10.
- Dataset-specific labeling differs (OPS-SAT weak fragment labels vs
  SMAP/MSL point labels). Cross-dataset F1 values are not directly equivalent.
- Original 21-alpha selection, 200-threshold paper Best F1, POT, dense Best-F1
  diagnostic are preserved. **Test-label-optimized alpha/Best F1/POT are oracle**
  and not validation-calibrated deployment performance. OPS-SAT also records
  its separately validation-selected metrics; report those apart from oracle.
- Exact *numeric* metric comparability with v10 must still be audited:
  the current scoring implementation and old v10 implementation may differ.
  We have preserved the training hyperparameters/splits, not certified
  a byte-for-byte identical evaluation implementation.
- Do not interpret differing batch sizes as an architecture-only effect.
  These are unchanged relative to v10, not the larger v11 profiling batches.

## Unified console and progress

- One `tqdm` campaign bar for **126 training** runs and then **126 evaluation**
  runs; explicit `[V12 TRAIN 001/126 | GLOBAL 001/252]` and
  `[V12 EVALUATE 001/126 | GLOBAL 127/252]` markers remain readable in `tee` logs.
- Dataset-independent `[TRAIN] DATASET | ID | epoch/50 | loss | lr | time | batches | throughput | GPU peak`
  every epoch for OPS-SAT, SMAP and MSL.
- Dataset-independent `[EVAL] DATASET | ID | alpha n/21` for all three,
  with `tqdm` on train/test scoring batches.
- Underlying evaluation is **once after training**; alpha sweep is still
  complete. A progress bar is observational, not an optimizer/metric change.
- `TOAST_NO_TQDM=1` disables animated bars while keeping text progress.
- Native v10 checkpoint IDs and v10 results are never overwritten;
  v12 writes under `../results/v12/study/`.

## VM installation (when SSH access is restored)

Do not replace the folder while a previous campaign is still running.
On the VM, in `/home/dliotta/AD-Satellite-Telemetry/`, after uploading the ZIP:

```bash
cd /home/dliotta/AD-Satellite-Telemetry/
source ad-satellite-telemetry/bin/activate
mv tranad_plus "tranad_plus_backup_$(date +%Y%m%d_%H%M%S)"
unzip TOAST_v12_COMPLETE_GRID_126_RUNS.zip
cd tranad_plus
python -m pytest -q tests/
python launch_v12.py --dry-run --device cuda:0
python launch_v12.py --check --audit-splits --device cuda:0
```

Inside **tmux**, only **one launch command**:

```bash
set -o pipefail
python -u launch_v12.py --stage all --device cuda:0 --precision faithful --adam standard --gpu-cache auto 2>&1 | tee v12_campaign.log
```

`--stage all` trains 126 models, verifies all 126 checkpoints, then evaluates
all 126. `--stage train` can be resumed after interruption using recovery
checkpoints; `--stage evaluate` starts only after all checkpoints exist.
Rerunning after a successful completion skips verified results. **Never
remove `results/v12/` to restart**; it holds the durable checkpoints.

Generate theory-only structural figures:
`python reports/make_plots.py --structural-only`.

## Operational notes

- The existing SSH `Your account has expired` error is an authentication
  problem; delivering this ZIP cannot repair access to `lxa100`.
- CPU unit tests and dry-run do not certify a full A100 training/evaluation.
- Do not install within the still-running original project folder.

Once all 126 evaluations finish, validate the **entire** completed 6 × 5
architectural grid against the existing v10 results (no reruns):

```bash
python tools/merge_v10_v12.py
```

This requires `../results/v10/study/all_experiments.csv` and
`../results/v12/study/all_experiments.csv`; it fails closed if any of the
270 runs, three seeds per cell, or required metric values are missing.
Output: `../results/v12/combined_grid/`. Inspect scoring parity **before**
drawing cross-version numerical claims.
