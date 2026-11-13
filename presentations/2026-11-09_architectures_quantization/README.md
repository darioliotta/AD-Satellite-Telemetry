# Results II — architecture and quantization

English · 20 slides · Padova red #9B0014 · no title page.

- Results_II.pdf: vector slides.
- Results_II.pptx: embedded SVG with 300-dpi fallback and speaker notes.
- figures/: standalone Python-generated PDF/SVG figures.
- data/: displayed values, selected per-seed data, source hashes and speaker notes.

Rebuild from the project root: `python -B presentations/2026-11-09_architectures_quantization/build.py`.
Dependencies: numpy, matplotlib, python-pptx, Pillow and lxml. The original
Results.pdf and all research result artifacts are preserved.

Architectural figures use the reference W/FF configuration and all five seeds.
Continuation uses the reference configuration, two architectures and five seeds.
Frozen context uses all four architectures at reference W/FF; three shuffles
are averaged within each model before five-seed summaries. Matched retraining
uses OPS-SAT W10/FF8 plus the W1 control. Two POT control cells have four
available evaluations with availability recorded in speaker notes; all five trained
seeds remain in the provenance. Quantization aggregate tables use all four
architectures and W/FF configurations, 80 parents per dataset. All scopes are
stated on the relevant slide and detailed in speaker notes.

Only completed results through V17 are included. V18 and temporary exploratory
no-PA reoptimizations are excluded. To add V18 later, add a verified reader and
a new page factory; do not change the meaning of the existing raw-at-PA-selected
comparisons. The figures are regenerated from numerical sources, not screenshots.
