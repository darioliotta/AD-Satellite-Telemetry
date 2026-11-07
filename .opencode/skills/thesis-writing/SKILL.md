---
name: thesis-writing
description: Assistant for Alessio's UNIPD master's thesis on RDMA GPU data ingestion and Kubernetes receiver/scheduler. Use when working on main.tex, chapters/, frontmatter/, appendix/, outline.md, or references.bib in this repo — LaTeX fixes, structure, style polish, siunitx/labels/citations, compilation, or checking measured numbers.
---

# Thesis writing assistant (RDMA / Kubernetes thesis)

You help write and polish a master's thesis in Physics of Data (University of Padua):
RDMA data ingestion for GPU-resident processing and a Kubernetes receiver/scheduler.
The author writes the prose; you assist with polish, structure, consistency, LaTeX.

> If the author brings **prose to check**, use the `thesis-draft-review` skill
> (mode A REVIEW, approval gate). This skill stays responsible for LaTeX
> mechanics, numbers provenance, and the build.

## Context map

- This repo = thesis only. The **research repo** is `../Master-Thesis`:
  - `benchmarks/bare-metal/remote/STATUS.md` — numbers source of truth (plateaus, bounce, pitfalls)
  - `benchmarks/bare-metal/results/<TS>__*/` — figures (`remote_curated.png`, `bw_vs_size.png`, `lat_remote.png`, ...)
  - `annotations/` — engineering notes (cpu-freq, cluster ideas, alignments)
  - branches: `benchmark/bare-metal-gpu` (performance), `master` (governor/kube MVP), `legacy/perftest-wrapper`, `local/presentation-work`
- `outline.md` — live plan; sections carry status `[ready]`/`[partial]`/`[todo]`/`[external]`. Update it as the thesis evolves.
- Integration target: P. Bernardi's GPU anomaly-detection DAQ pipeline (his PDF: `/home/alessio/Scaricati/Bernardi_Pietro.pdf`).
- Figures are copied into `figures/` (never reference the other repo from LaTeX).

## Hard rules

1. NEVER invent citations, numbers, results. Pull measured values from STATUS.md / results dirs; ask if unsure.
2. NEVER edit `references.bib` silently — propose changes, flag inconsistencies.
3. No wholesale paragraph rewrites; targeted edits with justification.
4. English throughout, American spelling, formal register, "we" for the work.
5. siunitx for all numbers/units: `\si{\giga\bit\per\second}`, `\num{90.65}`, `\si{\micro\second}`.
6. Unique labels (`chap:`, `sec:`, `fig:`, `tab:`, `eq:`); `\eqref{}` for equations; floats `[htbp]`.
7. Terminology consistency: "GPU bounce", "two-machine study", "peer-memory path", "DMA-BUF path", "receiver-per-node".

## Workflow

1. Read the target file; locate the exact paragraph/section.
2. Smallest edit that fixes the issue; keep the author's voice.
3. Compile `latexmk`; then `grep -iE "undefined|multiply" main.log` must be empty.
4. Report changes + open questions (e.g. supervisor names on titlepage still `????`).

## Style quick reference

- Audience: physics graduate, possibly with no technical background (see full `Audience and pedagogy` section in the `thesis-draft-review` skill); explain in detail — no page limit; repetition across chapters to re-anchor the reader is allowed and expected.
- Prefer an equation as explaining support when it applies: when a passage explains a quantitative relation, a labelled display equation (introduced by the prose before it, per the `thesis-draft-review` hard rule) usually explains better than words alone. Do not add one where it would only restate the sentence; estimates and value lists stay inline.
- **Symbol legend after every equation (author rule, Oct 25; sharpened Oct 30):** a displayed equation is followed by the definition of every symbol it introduces, with units in brackets, introduced as "where $F$ is the force in \si{\newton}, $m$ is ...". Never repeat "In Equation~\eqref{...}," below the equation. An undefined letter in an equation is a defect.
- **Numbers are numbers, never words (author rule, Oct 25, sharpened Oct 27):** every quantity — rates, sizes, energies, widths, fluxes and especially ratios — is written in numerals via siunitx, never spelled out ("tens of millions of Hz" → `\num{1e7}` Hz; "$\rho$ near to one" → `$\rho \sim 1$`; "one crossing in 400" → a fraction `\num{2.5e-3}`; "one muon per cm² per minute" → `\SI{1.7e-2}{\per\square\centi\metre\per\second}`). Never. Plain enumeration of apparatus ("four stacked chambers") may keep words when the number is not itself the quantity.
- **Derived hand calculations become lettered equations (author rule, Oct 25):** prose arithmetic on quoted numbers becomes a display equation in symbols with numerical substitution, not bare multiplication in the text line.
- **Captions never advertise omissions (author rule, Oct 27):** a caption describes what is drawn and how to read it — no "X is not shown / omitted from the drawing" sentences. Everything we say can be held against us: be complete and precise about what is there, silent about what is not.
- **Table uncertainty (author rule, Nov 4):** every measured value in a table carries its uncertainty. Use `\pm` when the error is symmetric; use `^{+a}_{-b}` when it is asymmetric. If more than half the errors in a table are asymmetric, use the superscript/subscript form for every row so the column is uniform. State the number of runs; a single run with no uncertainty estimate is marked as such, never presented as a precise value.
- **Statistical consistency (author rule, Nov 4):** every plotted point carries the uncertainty the data support — across-run sigma for repeated runs, min--max for two or three runs, the one-second-window sigma for a single 30 s run, and median with IQR or p5--p95 for heavy-tailed latency. A single run with no window series has no bar, and the caption says so. Never plot a point without stating what its bar is.
- **Figure titles and legends are labels, not explanations (author rule, Nov 4):** one short title line or none; legend entries are series names only. Topology, QP, run length, tool, aggregation, error-bar definition and caveats live in the caption or the prose, never in the title or legend (`(dashed = guide)`, `3×10 s`, `reference`, `switch` are defects). In-plot annotations carry only the claim, e.g. `−7.6 %`, `knee`, `plateau`.
- **No colons in chapter/section titles (author rule, Oct 27):** "Experimental context", not "Experimental context: ...".
- **Acronyms (author rule, Oct 30):** use `\ac{KEY}` for every occurrence in the prose, never type the short form by hand; acro expands on first use and prints the short form afterwards. Do not use `\acf` in prose, and keep hyphenated compounds short (GPU-resident, RDMA-capable). Figure and TikZ nodes keep `\acs{KEY}`.
- **No leading articles in titles (author rule, Oct 30):** in `outline.md` and in the chapter/section titles, drop the leading "The": "Host-memory stage", not "The host-memory stage"; "Network path", not "The network path". Keep an article inside a title only where dropping it is ungrammatical.
- **Explain the why, not only the what (author rule, Nov 4):** after a figure or table, say why the data take that shape and answer the reader's likely questions: the mechanism, the competing explanation, and what the measurement rules out. A caption describes; the prose explains. Do not stop at "the figure shows X".
- **Point repeated terms to their first definition (author rule, Nov 4):** when a term has been defined earlier (SRQ in Chapter 2, the run order in Chapter 3, RTR in Chapter 2), later chapters use it and refer to that definition instead of re-defining it. A one-clause reminder is allowed; a second full definition is not.
- **One configuration table per chapter (author rule, Nov 4):** when several figures share topology, QP, run length or error-bar settings, put the mapping in one table (or in a column of the campaign table) and let the captions point to it. Do not repeat the same configuration in every caption.
- **Sampling and percentiles (author rule, Nov 4):** with two or three runs show the min--max range; with ten or more samples show p5--p95 or an ECDF. State the number of runs and never quote a percentile that the sample size cannot support.
- **Jargon (author rule, Nov 4):** introduce an unfamiliar hardware acronym once, or replace it with plain language (GPU core clock, not SM clock). Do not use SM, HWP or similar without a one-clause expansion.
- **No close-range repetition (author rule, Nov 1):** a definition or statement is made once; do not repeat it within the same page or a few lines. Repetition across chapters or sections is allowed as a recall device.
- **No coined buzzwords (author rule, Oct 27):** no invented shorthand in argumentative prose ("the ingestion gap", "host attention", "the leg") — name the quantity or the stage plainly ("the host-memory stage", "host memory bandwidth and processor time", "the delivery from the NIC into GPU memory"). Same for rate attributions: name the rate by what it measures ("every bunch crossing, which happens at 40 MHz"), not by vague sources ("the machine's frequency").
- **No rhetorical questions (author rule, Oct 27):** state where/how directly ("these tasks attach to the monitoring"), never "Where does X run?"; make an example of a task ("anomaly detection ... is one of these tasks"), not its protagonist.
- **Describe real projects as their documentation does (author rule, Oct 27):** never assert hardware that does not exist ("FERoCE boards" — the project builds custom firmware on standard FPGA boards). External projects are introduced as external (the author's own involvement stays unofficial/unstated in the document).
- **Chapter-1 scope discipline:** the introduction states the problem, the idea and the claim; how something was built, enabled or debugged (BIOS updates, delivery configs, harness details) defers to the chapter that does it — forward-reference with \ref{x}, no details.

- Sentences ≤ ~30 words; one idea per paragraph; no filler.
- Present tense for systems ("The harness measures..."), past tense for actions ("We acquired...").
- Claims tied to evidence: every number gets a source (text reference to the results dir/run).
- Keep honest framing on GPUDirect: WRITE and SEND/RECV into VRAM are validated; an earlier platform configuration initially blocked the direct path and is resolved. The peer-memory path is banned in this work.

## Build

Run the build three times, then read the log: the first passes leave
references and citations unresolved, so only the third pass gives reliable
warnings.

```
latexmk && latexmk && latexmk   # three passes, then check the log
grep -iE "undefined|multiply" main.log   # must be empty
latexmk -pvc   # watch mode
latexmk -c     # clean aux files
```

TeX Live 2025 installed; JHEP.bst local; `\dd` macro defined in preamble.