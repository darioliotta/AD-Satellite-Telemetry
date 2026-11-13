# Validation before A100 execution

2026-11-08, local macOS CPU, PyTorch 2.2.2. No CUDA device available.

- 56 existing tests passed after consolidation: architecture equivalence,
  training/resume, pipeline, exploration, temporal controls and precision tools.
- Integer suite: 8 tests, 7 passed and 1 CUDA test skipped as expected.
  Tests cover exact integer accumulation, odd-size packed INT4 round-trip,
  attention layout against native MHA, all four architecture contracts,
  all three datasets' feature/window sizes, sharing, absence of FP GEMMs,
  batch-independent activation scales, exact exported-model reload,
  compact BF200/POT evaluation and cached-result reuse, campaign balance
  and protected output paths.
- All 240 historical v13 parent source manifests passed the explicit
  relocation compatibility check. Deliberately mismatched source hashes
  were rejected. All 59 relocated Python source hashes match relocation.json.
- Original models.py, OPS adapter and paper_metrics.py are byte-identical to
  the tracked versions. No result/data/checkpoint file was modified.
- New plan: 240 distinct parent identities, 80 per dataset, 60 per
  architecture and 60 per W/FF configuration, across five seeds. 480 integer
  evaluations plus 240 newly measured FP32 references; no training.
- Schema-based size estimate with initialized weights: total INT8+INT4
  tensor payload 31.84 MiB, compressed model containers about 35.06 MiB.
  These are model-only estimates, not observed campaign archive sizes;
  trained-weight entropy and reports/logs add bytes. A 3 GiB stop limit
  protects the output directory. Full pointwise arrays are not archived.

**Pending:** CUDA compilation, PTX instruction verification, exact device
integer products, CPU-vs-CUDA architecture comparison and measured A100
performance. The launcher performs these checks before processing datasets.
Native INT4 instructions are not implemented; W4A4 executes with INT8 MMA.
No A100 speedup or end-to-end detection result is claimed here.

Commands:

```bash
python -B -m unittest \
  tranad_plus.core.tests.test_architectures \
  tranad_plus.core.tests.test_pipeline \
  tranad_plus.campaigns.exploration.test_followups \
  tranad_plus.campaigns.exploration.test_windows \
  tranad_plus.campaigns.temporal.test_controls \
  tranad_plus.campaigns.precision.test_precision \
  tranad_plus.campaigns.integer.test_integer
```

Run from the project root, using the project environment. On the A100,
`bash tranad_plus/run_integer.sh --device cuda:0` runs the required CUDA
preflight and parent validation automatically.


## FP32 batch recovery correction (2026-11-08)

Ten targeted CPU recovery tests passed: per-format batch selection for all
three datasets, plan immutability, completed-job byte preservation, known
predecessor acceptance, rejection of unrelated source/data/profile/hardware
changes and artifact tampering, archival/rebinding of incomplete work, and
recovery from interruptions on either side of the atomic manifest update.
The source-fingerprint test checks the exact installed corrected Python tree.
A 240-job fixture reproduced the reported 210-complete / 1-interrupted /
29-new situation, verified skip decisions, and checked byte-for-byte
preservation of existing files during the read-only recovery check.
The shared FP32 score comparisons remain rtol=5e-4 and atol=1e-7.

The failing MSL GPU comparison cannot be reproduced locally without that
server GPU and its outputs. The corrected comparison must still pass on the
server; matching the parent batch is not presented as a guaranteed remedy.
Previously completed outputs are not retrospectively relabelled as results
from the corrected batch policy.
