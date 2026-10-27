---
name: thesis-writing
description: Write and maintain the LaTeX thesis on anomaly detection in satellite telemetry; organize chapters, check sources, bibliography, figures, units and reproducibility.
---

# Thesis writing: satellite telemetry anomaly detection

## Mission
Help develop a Physics of Data master's thesis using `main.tex`, `metadata.tex`, `chapters/`, `frontmatter/`, `appendix/`, `outline.md` and `references.bib`. Support the author's choices instead of replacing them. For review of scientific prose, load `thesis-draft-review` first.

## Reliable context map
- `AGENTS.md`: shared rules and available sources.
- `outline.md`: chapter scope and status.
- `chapters/*.tex`: the approved narrative and equations.
- `docs/journal.md`: decisions and provenance log.
- Dataset reports, training code, run logs, and papers: use only when actually present or supplied. Search exactly the needed evidence.

## Hard rules
1. Never invent anomaly labels, dataset properties, F1, precision, recall, AUC, inference latency, parameter counts or deployment feasibility.
2. Label all empirical claims with their source (dataset/run ID/report, paper citation). Report uncertainty and incomparable protocols.
3. Distinguish the original TranAD, the TranAD+ implementation being used, and experimentally verified architecture variants.
4. For ablations, record **window**, **batch**, **epochs**, **learning rate**, **seed**, **threshold strategy**, **train/test split**, **checkpoint origin** and **code version**.
5. Use `\label{chap:...}`, `\label{sec:...}`, `\label{fig:...}`, `\label{tab:...}`, `\label{eq:...}` consistently, `\eqref{}` for equations and `\cite{}` with verified keys.
6. Use `siunitx` for quantities with units. For equations, define symbols and units in the accompanying prose.
7. Use inline `\todo[inline]{...}` only for unresolved work; do not leave unsupported claims as factual prose.
8. Never modify scientific conclusions or bibliography silently.

## Workflow
1. Determine the chapter and smallest relevant context.
2. Identify which statements are already supported and which need a source.
3. Propose structural or scientific rewrites and await approval before inserting them.
4. Apply approved changes with minimal diff and update `outline.md` if status changes.
5. Run `latexmk -pdf main.tex`; check undefined citations/references and report build state.
6. Record important decisions in `docs/journal.md`.

## Voice
Formal academic English unless the author selects Italian. Explain concepts to a Physics of Data graduate unfamiliar with operational telemetry. Describe problem, method, evidence and limitations; avoid promotional language. Keep equations and figures tied to discussion in prose.
