---
name: thesis-draft-review
description: Review thesis prose for factual support, clarity, placement, equations, figures and citations. Enforce author approval before applying scientific rewrites.
---

# Draft-review workflow

## Default: A — REVIEW (no file writes)
1. Read the named passage and minimal related context from `outline.md` or surrounding section.
2. Check scientific correctness **against supplied evidence**; mark unsupported numbers and statements explicitly, do not fill gaps from intuition.
3. Check fairness of experiments: splits, preprocessing, thresholds, configuration changes, code and checkpoint provenance.
4. Check terminology: satellite telemetry, multivariate time series, point/sequence anomalies, TranAD/TranAD+, POT, ablations, quantization and HLS only where documented.
5. Check displayed equations (symbols/units), figure caption→label→reference, tables and `\cite{}` keys.
6. Return the compact verdict below and a complete, ready-to-paste LaTeX revision. Do **not** edit files yet.

## Verdict format
- **CORRETTEZZA:** major/minor issues, tied to sources.
- **COLLOCAZIONE:** right section/chapter?
- **REGISTRO:** concrete style or clarity improvements.
- **CITAZIONI:** verified entries, missing primary sources, unsupported assertions.
- **PROPOSTA:** complete LaTeX text for the selected passage.
- **DOMANDE:** blockers that need real experiment outputs or author's decisions.

## B — APPLY (after explicit approval)
Apply precisely the approved passage; keep unapproved content unchanged; run `latexmk -pdf main.tex`, then report changed files and warnings. Update `outline.md`/`docs/journal.md` when the decision affects them.

## C — MAINTENANCE (no scientific rewrite)
May fix safe LaTeX syntax, labels, file paths, typos and formatting; report every edit. References may be added only from verified bibliographic metadata, never from fabricated keys.

## Evidence rules
- Do not compare F1 numbers produced by different scoring or thresholding protocols as an apples-to-apples ablation.
- Report changes in batch size, and never call two optimizer trajectories identical because they share epochs.
- Keep training, evaluation and deployment claims separate. A proposed hls4ml/HGQ2 flow is not a proven FPGA implementation.
- Cite measured values from the exact evaluation JSON or experiment log; never recover them from a vague session summary when primary artifacts exist.
