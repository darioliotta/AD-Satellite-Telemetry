# V18 validation — 2026-11-09

Local macOS, Python 3.10 / PyTorch 2.2.2 CPU / NumPy 1.26.4. No local CUDA.

- Read-only preflight passed against all **240** transferred V17 source models
  and **720** precision cases: complete status, all source artifact hashes,
  numerical source compatibility, data/order, model files, 21-or-1 alpha grids
  and the existence of every stored per-alpha SPOT threshold.
- Eight initial V18 tests plus ten existing recovery tests passed (18 tests).
  They cover direct BF200 equivalence including score ties, correct pointwise
  counts and binary AUC, alpha selection without any PA calls, unchanged POT
  thresholds, single-output models, missing/invalid thresholds, finite checks,
  balanced 720-case planning and filters, protected output paths, score-statistic
  parity checks, complete/corrupt result verification and five-seed summaries.
- An additional CPU integration test passed through actual model inference,
  score reconstruction, threshold selection and crossed PA/raw reports for
  **all four architectures and all three precision representations**. Integer
  fixtures use the explicitly named integer CPU reference backend; exported
  quantized models are reloaded before testing. Synthetic pre-fitted thresholds
  test reuse/selection and do not constitute a fresh SPOT fitting validation.
- V17 integrity verification still reports **240/240 complete**, with no source
  result files modified. V17's historical recovery record is not rewritten.
- All new Python modules parse and the shell launcher passes `bash -n`.

The CUDA execution remains a server-side step. The launcher checks source
integrity before inference, performs exact integer kernel preflight on CUDA,
and uses the already validated unchanged V17 numerical runtime. Per-alpha
score-statistic parity and old-endpoint confusion-count checks are retained.
Statistics matching is not asserted to establish exact pointwise equality.

V17's saved test-inference times sum to 10,090.68 seconds (about 2h48m), across
all 720 cases. A 3–4 hour V18 runtime is a rough expectation allowing for
selection, PA comparison, checking and I/O overhead, not a measurement or
completion guarantee. The archive is report-only, with a 1 GiB stop limit.
No new GPU performance result or no-PA effectiveness result is claimed before
this campaign actually runs.
