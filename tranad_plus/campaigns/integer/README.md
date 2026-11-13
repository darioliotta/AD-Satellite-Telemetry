# V17 — integer weight and activation inference

The current code lives only in `tranad_plus/`. V17 is a campaign identifier,
not a copy of the implementation. The default output is
`results/v17/integer/`; historical outputs remain unchanged.

## Study

| Axis | Values |
|---|---|
| Datasets | OPS-SAT, SMAP, MSL |
| Architectures | TranAD+, one-decoder, one-pass, encoder-only |
| W/FF configurations | reference, window-only, FF-only, compact; exactly the v13 parents |
| Model seeds | 2026–2030, all five already trained seeds |
| Integer precisions | W8A8, W4A4 |
| Paired reference | FP32 from the same parent checkpoint |

240 parent models; 480 integer evaluations plus 240 FP32 evaluations. No new
training, QAT or FP16. OPS-SAT is first, then SMAP, then MSL; compact profiles
and simpler architectures come first within each configuration block. This is
a pragmatic ordering, not a measured wall-time ranking.

BF uses the original 200-threshold grid and the original 21-alpha grid for
two-output models. POT, precision, recall, F1 and AUC-PA are retained, with
point adjustment and additional raw metrics. Alpha/split/OPS fragment handling
are inherited unchanged. Test labels are used for the declared benchmark
selection, not to set quantization scales. Scales are dynamic per-vector
activation absmax and fixed per-output-channel weight absmax. These scales
change the quantization scheme relative to v16's weight-only per-tensor study;
the two experiments must be identified separately.

## What actually runs

All learned matrix projections (Q/K/V, attention output, FFN and reconstruction
head), **QK and AV attention products** execute signed INT8 operands with INT32
accumulators in a Triton CUDA kernel. No learned floating-point matrix is left
in the converted runtime. Biases, scales, residuals, positional encoding,
LeakyReLU, softmax, sigmoid and self-conditioning remain FP32, as agreed for
standard integer quantization. Floating point is used to rescale the result
**after** integer accumulation, never instead of the integer matrix product.

W4A4 restricts both operands to [-7,7]. Its weights are packed as two signed
nibbles per byte in the saved model and resident buffers. They are unpacked to
INT8 temporaries for execution. **This is genuine integer arithmetic at 4-bit
numerical precision on INT8 instructions; it is not native INT4 Tensor Core
execution.** It does not establish a native INT4 throughput or memory-peak
advantage. INT8 uses [-127,127]. There is no FP16/BF16 fallback and no silent
substitution of FP32 GEMMs.

The implementation uses the documented integer types of
[Triton `dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).
[TensorRT INT4](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
is weight-only, so it was not substituted for W4A4 activation quantization.

## Run on A100

Activate `ad-satellite-telemetry`, then from `tranad_plus/`:

```bash
python -B run_campaign.py integer --dry-run
bash run_integer.sh --device cuda:0
```

The launcher first compiles and inspects INT8/INT32 PTX instructions, verifies
exact integer products against CPU arithmetic for padded/irregular shapes,
and checks every architecture on CUDA. It then validates all 240 original
parents, source compatibility, data hashes and checkpoint configurations.
Only after this succeeds does evaluation start. Missing Triton, unsupported
GPU/instructions or a numerical test failure stop the launcher.

Local development has no CUDA GPU. CPU reference tests verify arithmetic,
attention layout, all architectures, weight sharing, sample-independent
scales and lossless export/reload. The CUDA tests are intentionally pending
until this server preflight passes; do not describe the A100 backend as
experimentally validated before that.

Filters select a balanced smaller subset without changing output identities:

```bash
bash run_integer.sh --dataset OPS-SAT --seed 2026
python -B run_campaign.py integer --status
python -B run_campaign.py integer --verify
```

`--only OPS-SAT/encoder_only/compact/seed_2026` selects one parent. Rerun the
same command after an interruption. Complete models are skipped after hash
verification; completed precision cases are reused after artifact checks;
an interrupted precision case restarts its evaluation, not its training.
Different source/data/hardware bindings are rejected rather than mixed.

## Saved artifacts and size

Each model gets a manifest, parent checkpoint hashes, effective config,
packed INT8 and INT4 runtime models, per-precision metrics, BF/POT alpha
search progress, fixed-parent-threshold metrics, raw metrics, score summaries,
100 latency samples, analytical MACs, operator profiles, CPU/GPU memory
telemetry, kernel instruction audit, logs and timing. FP32 references reuse
the existing parent checkpoint. No optimizer copies, full reconstructions,
pointwise score grids, predictions or duplicated datasets are archived.
OPS-SAT retains its small controlled-window probes.

The runner checks a **3 GiB safety ceiling**; this is a limit, not a size
forecast. A schema-based model-only estimate is 31.84 MiB of tensor payload
and about 35.06 MiB of compressed exports; reports/logs add space. See
`VALIDATION.md` for the scope of this estimate. `tables/per_seed.csv`, `mean_std.csv` and `paired_deltas.csv` contain
all results available so far. File compression ratios differ from tensor
payload ratios; use the latter to compare numerical representations.

Latency includes dynamic scaling, packing/unpacking temporaries, padding,
launches and transfers within forward, with JIT warmed up. Small attention
heads and windows can make integer inference slower. Report measurements;
never infer speedup from bit width. Python/Triton profiling has partial FLOP
coverage; analytical MACs are supplied separately. The dispatch audit runs
outside the timed loop.


## Resume after the FP32 parity stop

The initial MSL stop at `shared_decoder/ff_only/seed_2026` exceeded the
unchanged FP32 score tolerance at one of 1,575,063 score values. Parent MSL
inference used batch 128; the initial integer campaign used batch 64 even
for FP32. Different GPU batching is a plausible rounding cause, not a
confirmed diagnosis until the corrected server comparison succeeds.

The corrected runner uses the parent report's score batch for FP32 (128 for
MSL), and retains the plan's score batches for INT8 and INT4 (64 for MSL).
Alpha search, BF200, POT, PA, AUC-PA and score tolerances are unchanged.
Latency measurements remain batch 1. `storage.evaluation_policy` and actual
inference telemetry record which batch was used. Completed runs retain
their original source signatures, score batch and results; newly completed
runs record the corrected policy. Do not describe all 240 FP32 score passes
as having used the same batch policy.

`fp32_batch_recovery.json` explicitly pins the original and corrected source
hashes, including two known metadata-only predecessor variants. Recovery
accepts only that reviewed source change with identical plan, data, parent
checkpoint, effective profile and execution environment. Unknown mismatches
remain errors. The preflight checks this compatibility and all completed
artifact hashes before any result-directory changes.

Finished jobs are verified and skipped without changing their result files.
An incomplete predecessor is archived under its own
`recovery/<previous_signature>/`, including its old manifest, status, error,
any packed models and partial evaluation files. It is then bound explicitly
to the corrected code and batch policy; partial format evaluations from the
old binding are rerun. No training is performed. Recovery decisions for
completed and incomplete jobs are recorded in the new campaign report.
A further score-parity failure remains an error; no tolerance is relaxed.

Copy the current `tranad_plus/` from the Mac with the rsync command in
`../../CURRENT_CODE.md`. Then inside the server environment:

```bash
cd /home/dliotta/AD-Satellite-Telemetry/tranad_plus
bash run_integer.sh --device cuda:0
```

The initial check should report 210 completed models verified/skipped and
30 pending for the reported interruption. The actual run prints
`VERIFIED / SKIP` for completed models and resumes the pending work.
The results root stays `results/v17/integer/`; no manual copying, deletion
or editing of manifests/checkpoints is required.
