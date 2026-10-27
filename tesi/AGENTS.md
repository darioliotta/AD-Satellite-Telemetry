# Thesis collaboration instructions

## Scope
University of Padua / Physics of Data master's thesis starter. Topic: satellite telemetry anomaly detection, TranAD+ baseline and architecture variants, and possible resource-aware deployment. All specifics must be confirmed against research artifacts.

## Source of truth
- `metadata.tex`: title-page metadata (editable placeholders).
- `outline.md`: provisional plan and section statuses.
- `chapters/*.tex`: approved thesis prose.
- `references.bib`: bibliographic metadata only after verification.
- `docs/journal.md`: session decisions and changes.
- Research data, experiment logs, checkpoint metadata and PDF sources are external until added by the author. Never assume access or invent values.

## Collaboration rules
1. Distinguish **measured**, **computed**, **literature-backed** and **proposed** content.
2. Never invent performance results, citations, dates, model configurations or hardware claims.
3. For conceptual or scientific rewrites, propose the text first and apply only after explicit approval.
4. Preserve author voice; keep diffs small; state each file touched.
5. For every number, note the supporting experiment/source in a comment or in `docs/journal.md`.
6. Compare models fairly: document dataset split, preprocessing, metric, thresholds, seed, batch size, checkpoint and code version.
7. Run `latexmk -pdf main.tex` after modifying compilable LaTeX, and report errors instead of declaring success.
8. Do not import unrelated content from the source repo (RDMA, Kubernetes, GPUDirect or its experimental measurements).
9. The three skills in `.opencode/skills/` are guidance for tools that support the OpenCode skill format, not automatically installed ChatGPT connectors.
