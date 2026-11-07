# Local validation — 2026-11-03

## Extension validation — 2026-11-04

The `additional` campaign was prepared without launching scientific training.
All 18 CPU tests passed (the original 16 plus balanced-grid/disjointness and
single-factor-control regression checks). The CLI dry-run schedules exactly
168 unique new experiments: 56 per dataset, with no overlap with the original
72 completed identifiers. All 72 historical effective profiles still match
their saved `config.json` exactly. Their union covers the complete 240-run grid.

The full `additional --check` passed for all 12 dataset/setting pairs on the
real local datasets. `window_only` has the same window inventory as `compact`;
`ff_only` has the same inventory as `reference`. Data/model/loss/scoring code
remains byte-identical to that recorded in the first campaign; only
`planning.py` changed among the recorded Python implementation files.
The scientific results directory still contains exactly 72 evaluations.
The shell launcher passed `bash -n`; all configuration JSON files parsed.

The following sections record the original implementation checks.

No scientific campaign or server job was launched. The existing baseline,
datasets, thesis sources and legacy/v10/v12 artifacts were not edited.

## Automated CPU checks

The 16 standard-library unit/integration tests passed with Python 3.11 and
PyTorch 2.2.2 CPU. They cover reference forward identity, first-stage identity,
weight sharing, reduced parameter counts, gradients of the original native
loss, exact interrupted/resumed CPU weights for all four variants, dataset
split/window parity, native/fragment-aware BF/PA equivalence, absence of
single-output alpha search, rejection of nonfinite predictions, full synthetic
single-output and two-output training/evaluation, validation selection,
artifact corruption detection, completed-run reuse, campaign continuation after
a failure, campaign overlap, and real PNG/PDF export.

Synthetic integration runs and figures live in temporary directories and are
removed by the tests. They are not included in `results/v13/`.

```bash
python -m unittest discover -s tranad_variants/tests -v
```

## Existing real-data checks

The full `all --check` preflight succeeded for all six dataset/settings pairs,
including input hashes and normalized dataset construction:

| Dataset / setting | Train windows | Test windows | Validation windows |
|---|---:|---:|---:|
| OPS-SAT / reference | 123313 | 78315 | 101865 |
| OPS-SAT / compact | 123313 | 78315 | 101865 |
| SMAP / reference | 138481 | 436303 | — |
| SMAP / compact | 138216 | 436038 | — |
| MSL / reference | 59591 | 75003 | — |
| MSL / compact | 58941 | 74353 | — |

A four-window real training batch was also passed through each of the four
architectures for each dataset: loss and gradients were finite in all 12 cases.
No optimizer step or scientific result artifact was written by this smoke check.

Reference-setting unique trainable parameter counts:

| Dataset | Reference | Shared decoder | Single pass | Encoder only |
|---|---:|---:|---:|---:|
| OPS-SAT | 249 | 159 | 159 | 69 |
| SMAP | 67575 | 42075 | 42075 | 16575 |
| MSL | 323565 | 201465 | 201465 | 79365 |

## Limits

This validates the local CPU implementation and data adapters, not completed
detection results or A100 memory, runtime, numerical stability or performance.
The server environment uses a different PyTorch version and must run the
preflight and pilot before the larger campaigns. Hardware/environment identity
is recorded and used to keep measurements separate. Exact CPU recovery does
not establish bitwise reproducibility across CUDA devices or software versions.
