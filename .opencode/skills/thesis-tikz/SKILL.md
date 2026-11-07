---
name: thesis-tikz
description: Create or fix an inline TikZ figure for this thesis (RDMA/GPUDirect/Kubernetes). Use when adding a diagram to chapters/, fixing Fig 2.1/2.2/2.3 or the Ch. 1 figures (wedge, chains, ingestion), or checking label collisions and semantic correctness. Compiles via latexmk, enforces thesis figure conventions.
---

# Thesis TikZ diagrams (inline, RDMA-tuned)

You create or repair **inline** TikZ figures inside `chapters/*.tex`
(not standalone files). Every figure must compile with `latexmk`,
match the thesis visual language, and survive the semantic checklist
below. Adapted from `chennQwQ/opencode-academic-workflow`
(`new-diagram` skill + `tikz-measurement.md`, MIT) and Scott Cunningham's
`tikz_rules.md` (MixtapeTools, via that repo). The measurement math is
theirs; the styles, gallery, and semantic checks are thesis-specific.

## Hard rules (thesis invariants, never break)

1. **Inline only.** Figures live in `figure` floats with
   `\begin{figure}[htbp]` (author choice Oct 18, supersedes the earlier
   bare-float rule), `\centering`, `\caption` then `\label{fig:...}`.
   Never `[H]`.
   Code listings use the `float` option with default placement so they
   never break across pages. Every float is `\ref`'d at least once
   in the prose.
2. **Acronyms in figures use `\acs{}` short forms only.**
   `\ac{}` expands on first use — never inside a TikZ box.
3. **House styles.** Reuse the Chapter 2 palette so figures match:
   `font=\small`, `Stealth[length=2.6mm]`, `line width=1.1pt` boxes
   with `rounded corners`, fills `white` / `black!8` /
   `blue!10` / `orange!18`, labels `\scriptsize` with `fill=white`.
   Preamble loads only `arrows.meta,positioning,calc` — do not
   assume other TikZ libraries without adding them to
   `preamble/preamble.tex`.
4. **One paragraph per line** in `.tex` source; floats stay intact.
5. **Page budget.** Text width is 17cm (A4, 2cm margins). No figure
   wider than ~16cm; prefer 12-14cm. Never a bare `scale=X`
   (text does not scale with coordinates). Allowed: resize the
   layout, or `scale=X, transform shape` with a re-check of all labels.
6. **NEVER invent content.** Arrows, boxes, and captions must match
   the prose and the cited source (Mellanox manual v1.7, NVIDIA docs,
   `rdma-core` `verbs.h`). A figure showing what the text does not
   say is a bug.
7. **No title nodes inside figures** (author rule Oct 27): a figure
   carries no "title" text like `feeding a GPU-resident analysis` — the
   caption states what is drawn. Zone labels (`Online chain`,
   `Data plane`) are allowed: they group parts, they do not title the
   figure.
8. **Any node whose text contains `\\` carries `align=...`.** Without it
   the build dies with `LaTeX Error: Not allowed in LR mode`; hit twice
   on Oct 27 (fig 1.2's label column, fig 1.5's arrow labels). The house
   styles all include `align=` — keep it when you add styles.

## Gallery (pick one, do not start from scratch)

Scaffold from `templates/` in this skill dir (filenames without `.tex`).
Each template embeds the house styles + a coordinate map comment.

- `object-model` — two-sided resource map (sender/receiver columns:
  SQ/RQ, CQ each side, MR + lkey/rkey). For Fig 2.1 repairs.
- `bringup-flow` — vertical 3-phase flow (set up / connect / run +
  teardown) with parallel-branch joins. For Fig 2.2 repairs.
- `frame-layout` — horizontal header chain (Ethernet/IP/UDP/IB/payload/
  integrity) with transmission-order arrow. For Fig 2.3 repairs.
- `send-vs-write` — two-lane sequence: SEND consumes a posted
  receive WR vs WRITE lands by remote address+rkey + separate signal.
  Proposed new aid for §2.3.
- `completion-timeline` — post → in-flight → completion → consume →
  reuse lane with CQ poll points. Proposed new aid for §2.2 steady state.
- `dmabuf-path` — exporter/importer lanes (GPU driver ↔ DMA-BUF fd ↔
  HCA) with registration/keys still on the RDMA stack. Proposed new aid
  for §2.5.

If none fits, say so and pick the closest; keep its styles.

Chapter 1 added three in-chapter exemplars to scaffold from (copy the
`tikzpicture` out of `chapters/chapter-introduction.tex` and rename):

- `fig:intro-detector-layers` — the wedge: annular sectors drawn with the
  `arc[start angle=.., end angle=.., radius=..]` syntax in painter order
  (fill outer→inner), dashed arcs for the magnet and the muon chambers,
  annotation labels in an angle-ordered radial fan (tips inside each
  cell mid-radius), and the sector **unrolled along a radius rail**:
  one-line cells bottom-aligned on a rail with an `$r$` arrowhead, and a
  square bracket summing them into the crossing record.
- `fig:intro-daq-chain` — side-by-side vertical chains on a **shared row
  grid** (one pitch, one box width): two panels whose shared stages sit
  at the same row heights; a deliberately **empty slot** carries the
  annotation in place of the box the panel does not have (the absent
  hardware trigger level); each panel wrapped in its own dashed zone.
- `fig:ingestion-gap` — two-lane ingest from the same front end: today's
  staged lane vs the direct long hop (`DMA-BUF`) with the dashed live
  tail; no panel frame, no title node.

## Workflow

### Step 1 — Intent (one sentence)

Ask or state: what claim does this figure carry that the prose alone
cannot? Write it as a comment above the float, e.g.
`% Intent: show that WRITE names the remote buffer while SEND consumes a posted receive.`
If you cannot write this sentence, do not draw.

### Step 2 — Scaffold

Copy the template's `tikzpicture` into the target `figure` float.
Update the **coordinate map** comment (node centers, widths) when
coordinates change. Rename nodes/labels only; keep styles.

### Step 3 — Prevention pre-check (mandatory, cheap)

```bash
rg -n 'begin\{tikzpicture\}\[.*\bscale\s*=' chapters/<file>.tex
rg -n 'node\[' chapters/<file>.tex | rg -v 'above|below|left|right'
```

- Bare `scale=` → fix per Hard rule 5.
- Any edge label without a directional keyword
  (`above|below|left|right|above left|...`) → add one.
  `midway` alone is a position, not a direction.

### Step 4 — Measurement passes (compute, never eyeball)

Run in order; cite pass + numbers when reporting a collision.

- **P1 curves.** No `bend` in current Ch.2 figures; if you add one:
  `depth = chord/2 × tan(angle/2)`, safe distance = depth + 0.5cm.
  30°→×0.27, 35°→×0.32, 45°→×0.41 (half-chord multiplier).
- **P2 gaps.** `usable = center-to-center − w(A)/2 − w(B)/2 − 0.6cm`
  (0.3cm padding/side). Width per char: `\scriptsize` 0.10,
  `\footnotesize` 0.12, `\small` 0.15, `\normalsize` 0.18cm
  (bold +10%, mono +15%). Label wider than usable → move above/below.
  **CAVEAT (Oct 27, measured):** this table UNDERESTIMATES real TikZ
  text by ~25-30% — on the fig-1.2 row the boxes painted over each other
  exactly because of it. Never use the estimates to compute absolute
  positions or row widths; use them only as a cheap gap check with a
  ×1.3 safety factor, and lay rows out by anchor chaining
  (`([xshift=4mm]prev.south east)` with `anchor=south west`) with
  render verification at the end.
- **P3 arrow labels.** Horizontal: `above|below`. Vertical:
  `left|right`. Curved: outside of bend. Every label has `fill=white`
  so lines do not strike through text.
- **P4 boundaries.** 0.4cm clearance label ↔ shape edge. Verify
  against computed bounds (rectangle corners, circle top `cy+r`),
  not aesthetics. Do not share y-coordinates across different-size
  shapes.
- **P5 margins.** Label↔label / label↔arrow 0.3cm; arrow origin ↔
  box edge 0.15cm; any object ↔ text block edge 0.5cm.

### Step 5 — Semantic checklist (thesis-specific, runs after geometry)

- [ ] No wire arrow labelled "work requests" between endpoints.
  WRs are local descriptors; the wire carries RDMA messages.
- [ ] No single PD box spanning both hosts. One PD per device
  context per host — draw two dashed scopes or none.
- [ ] Caption says QP *pair* when two endpoints are drawn; a
  "single queue pair" caption needs a single-sided drawing.
- [ ] Bring-up order allows `ibv_post_recv` after INIT (legal
  interleave); do not force exchange → modify ×3 → post_recv
  as the only path unless prose says so.
- [ ] RoCE v2 is an encapsulation (headers on the wire), not a
  pipeline stage between post and poll.
- [ ] Every box/arrow in the figure is named in the prose or
  caption, and vice versa (e.g. ICRC box needs one prose sentence).
- [ ] Opcodes/fields match `verbs.h` (`wr_id/next/sg_list/num_sge/
  opcode/send_flags`, two unions, `ibv_sge{addr,length,lkey}`).

### Step 6 — Compile and confirm

```bash
latexmk -pdf -interaction=nonstopmode main.tex
grep -iE "undefined|multiply" main.log
```

`grep` must be empty. Open the PDF and visually confirm (red
`draw=red, very thin` debug outlines allowed temporarily; remove
before finishing). After any fix, re-audit all TikZ figures in the
chapter — the same pattern repeats.

### Step 7 — Report

Snippet → float (`fig:...`) + file:line; verdict per checklist item;
`main.pdf` page count; reminder that the prose must `\ref` the figure.

## Chapter-1 figure conventions (learned Oct 27, keep for new figures)

- **Frames, if any, are `fit` frames — never fixed-size panels.** A
  `minimum width=160mm` panel collapsed with the leftmost cells of
  fig 1.4 and was removed. A dashed zone = `zone/.style` + `fit=(...)`
  with `inner sep=0.35cm`, and it must clear every drawn cell (P4).
  Symmetric panels get symmetric zones; label a zone at its
  `north west`, `anchor=south west`, 2pt above the line.
- **Radial leaders for anything circular/sector-shaped:** spokes at
  distinct angles are parallel rays from one apex — they cannot cross.
  Order the labels in the same monotone direction as their angles, and
  land each leader tip *inside* its band (mid-radius), so the
  attachment is unambiguous. Never run leaders from a far label column
  across the whole drawing (the original messy onion).
- **Annotation text never sits ON a shaft.** Centering a label at the
  shaft's coordinate makes the vertical arrow strike through it (Oct
  27); place it `anchor=east|west` at least 0.3cm off the shaft, inside
  the zone when there is one, `fill=white` always (P3).
- **Rails and rows are anchor-chained, not measured:** cells
  bottom-aligned on a rail (`anchor=south west`, chained via
  `([xshift=4mm]prev.south east)`); the rail drawn *after* the cells
  and dimensioned by the last cell's anchors + a fixed clearance; a
  bracket spans `first.south west` to `last.south east` by anchors.
- **An empty slot is an annotation opportunity:** in fig 1.3(b) the
  Level-1 row is deliberately empty and the gray note sits exactly
  where the box would be — "the missing piece is the message". Use it
  for every absent-stage comparison.
- **Tall tikzpicture + long caption vs float placement:** a float+caption
  wider than ~0.7 of the text height fails `t` at default float params,
  then either occupies a sparse float page (empty bottom) or is swept
  past the section end by the chapter's `\clearpage`. The preamble now
  sets `\topfraction .85, \textfraction .1, \bottomfraction .5,
  \floatpagefraction .8` — when a figure drifts, check these before
  redrawing; move the float block in the source to make figures land
  on consecutive pages when the author asks for adjacency.

## Known issues in Chapter 2 (fix with this skill)

- Fig 2.1 (`fig:verbs-objects`): SQ→RQ "work requests" arrow (L49);
  one PD spanning two hosts + "single queue pair" caption (L54-57).
- Fig 2.2 (`fig:bringup-path`): forces d3→d4→d5a order, excluding
  the legal post-after-INIT interleave (prose L69 vs nodes L95-98);
  d6 "RoCE v2 transport" drawn as pipeline step (L99).
- Fig 2.3 (`fig:rocev2-frame`): ICRC box/caption (L182/185) with no
  prose sentence; UDP node shows only dport (acceptable, note it).
