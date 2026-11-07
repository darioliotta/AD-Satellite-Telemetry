# V14 — exploration campaign

V14 investigates three questions with one launcher, a declared plan, separate
outputs, progress bars and recovery. Existing v10/v12/v13 models and results are
read-only. Numerical model/training code in `tranad_plus/` and `tranad_variants/`
is reused unchanged.

## What runs, in order

| Stage | Question | Inputs | Work |
|---|---|---|---|
| `windows` | Why did shorter windows improve some first-ablation metrics? | All 270 v10/v12 score exports/checkpoints | CPU analysis, no training or new inference |
| `context` | How much do the trained architectures depend on their past input? | 60 v13 checkpoints | Three test-inference conditions per checkpoint, no training |
| `continue` | Does the 50-epoch training budget affect the comparison? | 30 v13 recovery checkpoints | Epochs 51–100, followed immediately by full evaluation for each model |

`plan.json` declares the datasets, configurations, models and seeds. Defaults:

- Window study: OPS-SAT, SMAP, MSL; six windows x five FF widths x three original
  seeds (2026–2028), exactly 270 saved runs.
- Context: all three datasets; reference and encoder-only; reference/window-only
  configurations; five seeds (2026–2030), exactly 60 checkpoints.
- Continuation: SMAP, OPS-SAT, then MSL; reference and encoder-only; reference
  configuration; five seeds, exactly 30 continuations. SMAP is scheduled first
  because its recorded epochs were faster on the A100.

## Window study: an explicit control of the first ablation

The study compares window lengths **at the same effective FF width and seed**.
It records both the original reported metrics and newly computed controls:

1. Original POT, BF200 and dense PA-oracle F1, preserving their original alpha
   selections. These columns reproduce the saved study rather than replacing it.
2. New diagnostics at fixed alpha=0.5, without test-based alpha selection.
3. Actual test points only: remove extra padded target copies in native SMAP/MSL
   exports, retaining identical real points for every window length.
4. A second common population excluding the initial `max(W)-1` real points of
   every track/fragment. This checks whether the window effect depends on padded
   historical context. The mask is identical across windows.
5. Continuous AUC/AP and raw Best F1, plus dense PA F1 at the same fixed alpha.
6. Direct point recall versus interval coverage at the same 0.5% test-normal FPR.
7. OPS-SAT fixed mean/max fragment-score AUC, with fragment boundaries retained.

Only threshold-oracle measures use optimized test thresholds. The matched-FPR
comparison also uses normal test labels and is explicitly diagnostic. The
train-quantile threshold remains a separate train-only calibration. Point labels
in OPS-SAT are weak fragment labels and do not locate an exact anomaly onset.

`paired_window_effects.csv` compares each shorter window with the longest at the
same FF and seed. The original campaign version and source hashes remain recorded;
matching seed/FF is not a proof that every historical campaign implementation was
identical. No additional training is required for this stage.

## Context interventions

Every checkpoint is evaluated with:

- `original`: unchanged windows; scores must reproduce the saved v13 scores within
  declared numerical tolerances before any intervention is interpreted.
- `reverse_past`: reverse all points before the current point; keep the current
  point and reconstruction target unchanged.
- `repeat_current`: replace all past points with the current value; keep the
  current target unchanged.

Alpha stays at 0.5 for reference; encoder-only has one output. The original train
99.5th-percentile threshold is held fixed across interventions. There is no PA.
OPS-SAT reports mean/max fragment AUC, and native results also report common real
points without padded target copies. Feature-specific reconstruction errors and
AUC associations with global labels are saved, to investigate score averaging;
these associations are not a feature-localization accuracy measure.

An artificial history intervention may cause distribution shift. Sensitivity is
therefore evidence about the frozen model's response, not a complete proof of why
shorter-window training performs differently.

## Continued training and recovery

Continuation requires the recovery file from the end of epoch 50:
`checkpoints/resume.ckpt`. It imports model weights, Adam moments, StepLR state and
Python/NumPy/CPU/CUDA RNG state into a **new** result directory. Final/recovery model,
optimizer and scheduler states must agree. The source checkpoint is never changed.

The original trainer performs epochs 51–100, with absolute epoch-dependent loss
weights `1/51 ... 1/100`, unchanged batch order, batch size and decaying LR. Each
added epoch writes durable epoch telemetry and a recovery checkpoint. Evaluation
starts immediately when that model reaches epoch 100, before the next training.
Original POT/BF200/dense PA metrics and OPS-SAT validation-selected evaluation are
retained; `comparison.json` also contains fixed-alpha continuous AUC/AP and raw F1.
The test epoch is fixed in advance: no best test epoch is selected.

Rerun the identical command after interruption: completed jobs are verified/skipped,
and incomplete training resumes from its saved epoch. Changing the plan, source,
data or numerical environment is rejected rather than silently mixed into a run.

## Progress and timing

`tqdm` is enabled by default:

- first-ablation runs processed / total, with elapsed time and ETA;
- checkpoint jobs completed / total;
- context conditions / three, plus batch progress;
- additional training epochs / 50, plus batch progress, loss and epoch seconds;
- evaluation batch progress and alpha search / 21 or / 1.

Full text is saved to each job's `run.log`; repetitive training/alpha lines are
kept in the log while the terminal shows progress bars. `progress.json` records
phase, epoch/condition, elapsed time and available remaining-time estimates.
ETAs are estimates; mixed evaluation/training job durations differ. Use
`--no-progress` for plain console output; timing/log files are still written.

## Layout

```text
tranad_v14/
  plan.json                   # Predeclared study settings
  run_v14.sh                  # Preflight, then all stages
  run.py                      # CLI, provenance and job dispatch
  windows.py                  # v10/v12 window controls and figures
  context.py                  # Frozen-checkpoint input interventions
  continuation.py             # Fork recovery state, train, evaluate
  metrics.py                  # Explicit diagnostic metric definitions
  progress.py                 # tqdm, text logging and timing state
  summarize.py                # Matched run tables and seed means/SD
  test_followups.py
  test_windows.py

results/v14/
  _campaign_logs/              # One timestamped report per launch
  campaign_latest.json         # Latest campaign status
  window_study/
    runs.csv                  # 270 individual runs
    mean_std.csv              # 90 W/FF cells; original three seeds
    paired_window_effects.csv # 225 shorter-vs-longest comparisons
    window_effects.pdf        # Three pages, fixed baseline FF
    *_window_effects.png
    provenance.json
  context/<dataset>/<model>/<setting>/seed_<seed>/
    metrics.json
    scores.npz
    manifest.json
    diagnostic_status.json
    progress.json
    run.log
  continuation/<dataset>/<model>/reference/seed_<seed>/
    checkpoints/{resume,final}.ckpt
    epochs.jsonl              # Parent epochs 1–50 + new epochs 51–100
    comparison.json           # Paired epoch-50 / epoch-100 results
    metrics.json
    scores.npz
    validation_selection.json
    manifest.json
    diagnostic_status.json
    progress.json
    run.log
  summary/
    context_runs.csv
    continuation_runs.csv
    mean_std.csv              # Means/sample SD; n explicitly reported
    index.json
```

Historical times for copied epochs 1–50 are not new training time. No raw result is
overwritten by a mean or by a maximum over seeds.

## Copy and start

From the **local project root**:

```bash
rsync -avh --partial --progress --exclude='__pycache__/' \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  tranad_v14/ \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/tranad_v14/
```

The server must retain the original full `results/v10/study/`,
`results/v12/study/` and `results/v13/architectures/` trees. Preflight verifies their
required reports, score exports, checkpoints, source/data hashes and the numerical
continuation environment before the campaign starts. Existing A100 training used
48 CPU threads, FP32, TF32 disabled and the original optimizer configuration.

On the **server**:

```bash
tmux new -s v14
cd /home/dliotta/AD-Satellite-Telemetry
source ad-satellite-telemetry/bin/activate
bash tranad_v14/run_v14.sh --device cuda:0
```

This uses the project-local virtualenv shown in the server tracebacks. If the
existing environment has since moved into Conda, use its usual activation command.
Detach with Ctrl-b, then d; reattach with `tmux attach -t v14`.

Preview the plan (no results written):

```bash
python -B -m tranad_v14.run --dry-run
```

Only first-ablation diagnostics (CPU; no torch needed):

```bash
python -u -B -m tranad_v14.run --stage windows
```

Only context evaluations, or only the ten MSL continuations:

```bash
bash tranad_v14/run_v14.sh --stage context --device cuda:0
bash tranad_v14/run_v14.sh --stage continue --dataset MSL --device cuda:0
```

`--dataset` and `--seed` can be repeated. The window stage accepts its original
seeds 2026–2028; use `--stage context` or `--stage continue` to select only a new
v13 seed such as 2029. `--stage both` retains the old context+continuation-only
selection. The default is all three stages.

Transfer results back from the **local project root**:

```bash
mkdir -p results/v14
rsync -avh --partial --progress \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/results/v14/ \
  results/v14/
```

## Checks

```bash
python -B -m unittest tranad_v14.test_followups tranad_v14.test_windows -v
```

Checks cover target preservation, exact CPU continuation with progress hooks,
interrupted recovery, immutable parents, real/interior test populations, PA parity
with original code, corrupt-input rejection, exported window tables/figures and
end-to-end 50->100 training/evaluation with verified safe reruns.
