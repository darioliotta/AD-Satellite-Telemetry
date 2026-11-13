# V28 — Completion campaign (protocol fixed before running, 2026-11-13)

Code: `tranad_plus/campaigns/completion/run.py`. Results: `results/v28/`.
Training, evaluation, models and data adapters are the unchanged v13 implementations
(`tranad_plus/core`, pinned by `BASELINE_SOURCES.json`). Every run saves `scores.npz`
with the v13 layout (train/test[/validation] scores for 21 alpha values, labels; OPS-SAT
segment ids and channels), so threshold-free and equal-false-alarm metrics can be
computed afterwards for every run.

| Study | Jobs | What |
|---|---|---|
| `grid_rescore` | 270 | Inference only on the existing v10/v12 window × FF grid checkpoints (3 seeds). Recomputes the authors' BF200+PA F1 and stores it next to the original report value (`agreement_with_source_report`). |
| `grid_seeds` | 180 | Same 30 window × FF cells per dataset (OPS-SAT, SMAP, MSL), reference model, seeds 2029 and 2030, 50 epochs, baseline training settings of `core/configs/baselines.json`. Brings the grid to 5 seeds. |
| `univariate` | 20 | TranAD+ reference and encoder-only on SMAP/MSL using only the telemetry coordinate (input coordinate 0), seeds 2026–2030, reference window (SMAP 10, MSL 50), FF = 8 (the univariate OPS-SAT reference value; the multivariate FF 50/110 has no meaning for one input). Same input as the KNN/MLP detectors of V21–V27. |

Univariate inputs are sliced after the standard per-coordinate min–max scaling, which is
therefore identical to the scaling of coordinate 0 in the multivariate runs.

Local verification before the server run (CPU, scratch output, not kept):
- rescoring OPS-SAT W=3, FF=2, seed 2026 and SMAP W=3, FF=13, seed 2026:
  recomputed BF200+PA F1 identical to the original reports (difference 0.0);
- one-epoch smoke runs of a univariate MSL encoder-only job and an OPS-SAT grid-seed job:
  training, evaluation and `scores.npz` written with the expected shapes
  (MSL univariate windows 50 × 1; OPS-SAT 21 alpha rows with validation split).

Server run (4 shards in parallel on one A100; each job is locked, completed jobs are skipped on restart):

    python -B -m tranad_plus.campaigns.completion.run --check --limit 3
    python -B -m tranad_plus.campaigns.completion.run --shard 0/4   # and 1/4, 2/4, 3/4

Transfer size. Each run also writes `scores_compact.npz` (alpha rows 0, 0.5, 1 only; about
2–3 MB instead of 15–20 MB). The continuous AUC for all 21 alpha values is in `metrics.json`
(`paper_oracle.alpha_grid`). Only compact scores, JSON records and final checkpoints are copied
back (about 1.5 GB); full `scores.npz`, resume checkpoints and logs stay on the server.

The compact file determines every alpha: the score is a quadratic in alpha,
s(a) = a² s(1) + (1−a)² s(0) + 2a(1−a) C with C = 2 s(0.5) − (s(0)+s(1))/2.
Checked locally on two runs against the full 21-row files: AUC differences ≤ 1e-5 and
relative differences of the nominal 99% quantile ≤ 2e-6 over all 21 alphas (float32
cancellation affects only scores near zero, not rankings or thresholds).
