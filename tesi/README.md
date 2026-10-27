# LaTeX master's-thesis starter — satellite telemetry anomaly detection

Adapted from the *structure and layout* of a shared Physics of Data thesis repository. The unrelated author's research prose, experimental figures, notes, references, and personal metadata are **not included**. Institutional Padua logo files and the original JHEP `.bst` are retained to preserve the usable style.

## Start in 3 steps

1. Edit `metadata.tex` (candidate, supervisor, title, academic year; all placeholders).
2. Edit `outline.md` and fill individual files in `chapters/`. Replace the `\todo[inline]{...}` markers with verified thesis content.
3. Run `latexmk -pdf main.tex` in this directory. Output: `main.pdf`. Requires a reasonably complete TeX Live distribution (`acro`, `tikz/pgfplots`, `todonotes`, `siunitx`, etc.).

`latexmk -c` cleans auxiliary build files. To print without draft comments, comment `\linenumbers` and `\listoftodos` in `main.tex` and remove/disable the `todonotes` entries.

## Folder map

- `main.tex`: document entry point, chapter includes, bibliography.
- `metadata.tex`: only titlepage personal/institutional fields.
- `preamble/`: typographic configuration and abbreviations.
- `frontmatter/`: title page, abstract, acknowledgements.
- `chapters/`: editable section skeleton; **not** completed scientific prose.
- `appendix/`: reproducibility materials.
- `figures/`: retained institutional art; add verified scientific figures later.
- `references.bib`: currently empty to avoid copying irrelevant citations; set `\thesisbibliographytrue` in `metadata.tex` after adding real citations.
- `outline.md`, `AGENTS.md`, `docs/journal.md`: plan and collaboration rules.
- `.opencode/skills/` and `.opencode/command/`: adapted skills and command prompts. These are plain project files: they do not install anything in ChatGPT by themselves.

## Reproducibility notes

When importing real experiments, preserve exact run identifiers and logs. The original attempted WADI configuration (`batch 8192`) and any successful alternative (`batch 4096`, if completed) are **distinct experiment configurations**. Never merge their provenance under one ID.

## Notes

The chapter plan and provisional English title are editorial suggestions to be confirmed, not a statement that the work or results have been completed. The university branding is inherited from a University of Padua template; verify final submission requirements with your department before submission.
