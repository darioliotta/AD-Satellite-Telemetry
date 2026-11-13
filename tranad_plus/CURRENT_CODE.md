# One current implementation

`tranad_plus/` is the only current source root. Do not create versioned sibling
copies. Campaign versions identify plans and immutable results, not branches
of the implementation.

| Location | Purpose |
|---|---|
| `TranADPlus/`, existing root scripts | Original model and benchmark/native pipeline |
| `core/` | Shared architecture variants, data adapters, training, evaluation |
| `campaigns/exploration/` | V14 context/window/continuation tools |
| `campaigns/temporal/` | V15 temporal interventions |
| `campaigns/precision/` | V16 weight-only sensitivity and author-configured SMD |
| `campaigns/integer/` | V17 integer weight/activation inference |
| `campaigns/thresholds/` | V18 BF200/POT selection without PA |
| `campaigns/regularization/` | V19 controlled holdout, early stopping, L2 and optional integer reevaluation |
| `campaigns/mechanisms/` | V20 score separation, simple detectors and matched current-sample controls |
| `run_campaign.py` | Common entry point, usable from `tranad_plus/` |

Examples:

```bash
python -B run_campaign.py --help
python -B run_campaign.py architectures --help
python -B run_campaign.py integer --dry-run
bash run_integer.sh --device cuda:0
```

V20 is documented in `campaigns/mechanisms/README.md`; launch with
`bash run_mechanisms.sh`. It covers OPS-SAT, SMAP and MSL, all four architectures,
432 historical-model precision evaluations, six deterministic baselines and
72 matched observed/hidden-current trainings. Output: `results/v20/mechanisms/`.
Historical and new complete-window protocols are kept separate. Final and rolling
recovery checkpoints are retained only for new trainings; historical weights
are read-only. Source/data/model hashes and case-level resume checks are required.

V19 overfitting controls are documented in `campaigns/regularization/README.md`.
Use `bash run_regularization.sh --stage diagnostic` for 60 controlled
trajectories: OPS-SAT architecture controls plus SMAP/MSL benchmark-parameter
anchors. The optional compression stage adds 96 native trajectories, with
156 unique trajectories across both stages. Results go to
`results/v19/controlled_study/`; earlier OPS-only V19 artifacts are preserved.
Native validation is a nominal chronological training holdout; official tests
are unchanged. OPS-SAT also has a detection-selected validation checkpoint.
The full-horizon audit freezes the first nominal early-stopping decision and
retains original-horizon/final endpoints. Optional `--quantize-only` evaluates
integer PTQ without retraining or duplicate integer weight archives.

The former sibling directories `tranad_variants`, `tranad_v14`, `tranad_v15`
and `tranad_v16` were moved into this tree. They are not needed on the Python
path. Dataset splits, model objectives, floating-point model implementations,
metric computations and historical result files were not changed by the move.
`relocation.json` records actual before/after SHA-256 values. Core numerical
modules were checked for AST equivalence after canonicalizing import paths;
bootstrap and entry-point filesystem adjustments are recorded separately.
Historical parents still require matching numerical sources, matching input
hashes and verified artifacts. Only explicitly recorded hash transitions are
accepted; arbitrary mismatches are rejected. All 240 v13 parent source
manifests passed this check locally.

Completed historical campaigns remain under their original result paths.
Source-bound campaigns are immutable: do not rerun a historical command into
the same directory after changing its source bindings. Existing final/recovery
checkpoints remain readable; use a new campaign destination for new experiments.
The native benchmark resume mechanism is unchanged. V17 can resume its own
interrupted work with the identical launch command and verifies saved files.

## Install on the server

From the local project root:

```bash
rsync -avh --partial --progress \
  --exclude='__pycache__/' --exclude='*.pyc' \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  tranad_plus/ \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/tranad_plus/
```

This copies current source only. It does not transfer or replace result data.
Stop any process still importing the obsolete source roots before removing them.
On the server, from `/home/dliotta/AD-Satellite-Telemetry`:

```bash
tmux new -s integer_v17
source /home/dliotta/AD-Satellite-Telemetry/ad-satellite-telemetry/bin/activate
cd /home/dliotta/AD-Satellite-Telemetry/tranad_plus
python -B retire_old_sources.py --apply
bash run_integer.sh --device cuda:0
```

`retire_old_sources.py` removes only the four obsolete sibling source roots.
It refuses directories containing checkpoint/data/archive artifacts or
symlinks. It never touches `data/`, `results/`, unrelated backups or archives.
Use without `--apply` to inspect the list first. Do not use rsync `--delete`
on the entire project or on research outputs.

The A100 backend requires CUDA PyTorch and Triton (normally bundled with the
Linux CUDA PyTorch wheel). The launcher validates availability and actual
integer instructions on that machine before evaluation. See
`campaigns/integer/README.md` for exact numerical and hardware limitations.

Results can later be copied from `results/v17/integer/`. Create the local
parent first with `mkdir -p results/v17/integer`; rerunning the same rsync
command safely resumes an interrupted transfer.


### V17 FP32 recovery update

After the original FP32 parity interruption, copy this same current source
tree again and relaunch `bash run_integer.sh --device cuda:0`. The runner
now matches the parent's score batch for pending FP32 evaluations, keeps
integer batches and score tolerances unchanged, and recognizes the explicitly
reviewed original source hashes. Completed jobs retain their original
manifest/signature and pass artifact verification before being skipped.
The incomplete attempt is archived locally within its result directory
before the corrected run starts. See `campaigns/integer/README.md`.


### V18 without-PA selection

The current dispatcher adds `python -B run_campaign.py thresholds` and the
launcher `bash run_no_pa.sh --device cuda:0`. This evaluates all 240 V17 parent
models in FP32/INT8/INT4 (720 cases), selecting BF200 and POT alpha by pointwise
F1 without PA. POT reuses the V17 per-alpha SPOT thresholds; BF200 retains the
original 200-threshold range. It performs test inference only, without training
or model copies. Output is results/v18/no_pa/. See campaigns/thresholds/README.md.

The V17 full-source fingerprints remain historical and are not rewritten to
include V18 files. The corresponding recovery regression test now checks the
unchanged numerical dependencies instead of assuming the entire source tree
can never grow. Historical result integrity verification remains available
through `run_campaign.py integer --verify`. Completed V17 results are immutable.
