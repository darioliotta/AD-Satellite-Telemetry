# Master's thesis — satellite telemetry anomaly detection

Adapted from the *structure and layout* of a shared Physics of Data thesis repository. The unrelated author's research prose, experimental figures, notes, references, and personal metadata are **not included**. Institutional Padua logo files and the original JHEP `.bst` are retained to preserve the usable style.

## Writing workflow

1. Use [outline.md](outline.md) as the agreed chapter plan and writing-status tracker.
   Preserve the personalized title page and its metadata unless the author requests changes.
2. Develop one section at a time in `chapters/`: propose scientific prose for review
   before applying it. Replace draft markers only with verified thesis content.
3. Run `latexmk -pdf main.tex` in this directory. Output: `main.pdf`. Requires a reasonably complete TeX Live distribution (`acro`, `tikz/pgfplots`, `siunitx`, etc.).

## Agreed chapter structure

1. Introduction — `chapters/chapter-introduction.tex`
2. Reference Models — `chapters/chapter-background.tex`
3. Datasets and Experimental Protocol — `chapters/chapter-datasets.tex`
4. Baseline Reproduction and Adaptation — `chapters/chapter-replication.tex`
5. Ablation Study — `chapters/chapter-ablation.tex`
6. Architectural Simplification — `chapters/chapter-architecture.tex`
7. Quantization — `chapters/chapter-quantization.tex`
8. Discussion and Conclusions — `chapters/conclusions.tex`

The first five chapters have drafts. The remaining chapters
retain the agreed section outline. Experimental completion and writing status
are tracked separately.

`latexmk -c` cleans auxiliary build files. Line numbering is disabled throughout
the document. The draft to-do list and visible work-in-progress boxes have been
removed; writing notes remain only in source comments and the outline.

## Folder map

- `main.tex`: document entry point, chapter includes, bibliography.
- `metadata.tex`: personalized titlepage fields; preserved by the outline update.
- `preamble/`: typographic configuration and abbreviations.
- `frontmatter/`: title page, abstract, acknowledgements.
- `chapters/`: Chapters 1--5 have first drafts; architecture, quantization and conclusions have section outlines.
- `data/chapter4/benchmark-evidence.json`: selected result paths/hashes, effective configurations, published values and the recomputed WADI diagnostic supporting Chapter 4.
- `data/chapter5/`: verified per-run BF200/POT metrics and evidence for the hyperparameter study.
- `tables/chapter5/` and `figures/chapter5/`: generated tables and vector plots for Chapter 5; rebuild with `python -B tesi/scripts/prepare_chapter5.py` and `python -B tesi/scripts/plot_chapter5.py` from the project root.
- `figures/`: institutional art and verified scientific figures; simple diagrams may also be drawn directly in the chapter sources.
- `references.bib`: verified sources cited in the drafted chapters. Bibliography output is enabled in `main.tex`, preserving the personalized `metadata.tex`.
- `outline.md`, `AGENTS.md`, `docs/journal.md`: plan and collaboration rules.
- `.opencode/skills/` and `.opencode/command/`: adapted skills and command prompts. These are plain project files: they do not install anything in ChatGPT by themselves.

## Reproducibility notes

When importing real experiments, preserve exact run identifiers and logs. Different
batch sizes, layer counts or other configuration changes are distinct experiments;
never merge their provenance under one ID. The six-dataset reproduction benchmark
and the main OPS-SAT/SMAP/MSL simplification study have different roles. Separate
the original BF search from the dense oracle evaluation, and verify evaluation
compatibility before combining results from different campaigns.

## Notes

The initial outline was approved on 2026-11-02 and extended to eight chapters
on 2026-11-10; writing and experimental
completion remain separate statuses. The thesis addresses the initial TOAST
data/protocol and compact-model activities. Hardware cost analysis is included
within the experimental chapters; target-device implementation and radiation
validation remain future work unless supported by additional experiments.
Verify final submission requirements with your department before submission.
