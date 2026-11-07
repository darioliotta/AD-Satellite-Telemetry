---
name: thesis-draft-review
description: Peer-review loop for the thesis prose. Use when the author writes or pastes new LaTeX prose and asks for a check, a review, feedback on correctness, placement, or register, or a cleaner version. Enforces the approval gate - never write the author's prose into the .tex before showing the proposed text and getting explicit approval. Also covers the accessory work that does not need the gate (bibliography, siunitx, labels, figures, outline and AGENTS.md updates, compile fixes).
---

# Thesis draft review (author writes, assistant reviews)

The author writes ~90% of the prose. The assistant reviews, proposes, and
applies only what the author has approved. This skill covers prose; the
`thesis-writing` skill covers LaTeX mechanics, numbers, and build.

## Context map

- Thesis repo = this one. Research repo = `../Master-Thesis`
  (`benchmarks/bare-metal/remote/STATUS.md` = numbers source of truth;
  `benchmarks/bare-metal/results/<TS>__*/` = raw results and figures;
  `annotations/` = engineering notes).
- `outline.md` = plan with `[ready]`/`[partial]`/`[todo]`/`[outlook]`
  statuses. `docs/journal.md` = append-only session log.
- `AGENTS.md` = always-loaded state; session context is capped and rotated.
- Downstream context: P. Bernardi's thesis
  (`/home/alessio/Scaricati/Bernardi_Pietro.pdf`).
- Figures are copied into `figures/chapN/`; never reference the other repo
  from LaTeX.

## Concurrency with the author's editor

The author no longer writes in Emacs: he writes prose in chat, and the
assistant reviews it and writes the `.tex` itself. So file ownership and
revert-buffer are retired; what remains is **chat handoff as the default**.

- The author pastes or describes new prose; the assistant reviews it briefly,
  rewrites it formally, applies it to the file, builds, and reports exactly
  what changed with the rationale. The chat is the record; git is the undo.
- The review gate narrows but stays: the assistant shows the reviewed text in
  chat *before* writing when the passage decides what the thesis claims
  (framing, results, forward references), and writes directly for pure
  form-and-placement moves. When in doubt, show first.
- The assistant announces which files it is about to touch before touching
  them. Skill commands `/review` (verdict only, no write) and `/next-section`
  keep their old behaviour.

## Modes

### A - REVIEW (default when the author brings prose; NO file writes)

1. Read only the target file, and only the relevant section.
2. Run the standing equations/floats checklist beyond prose: every displayed
   equation labelled and explained in the prose right before it
   ("...as expressed in Equation~\eqref{eq:..}:"), followed by the symbol
   legend (every letter defined with its unit in brackets, author rule
   Oct 25); inline `$...$` only for
   unimportant relations, big-operator equations (`\int`, `\sum`, stacked
   fractions) on their own line; where a passage explains a quantitative
   relation in words alone, check whether an equation would carry the
   explanation better — the equation as explaining support is preferred when
   it applies, so propose one there; where prose does arithmetic on quoted
   numbers, check it becomes a lettered equation with numerical substitution
   (author rule Oct 25); every quantity written as a number, never spelled
   out (author rule Oct 25: "tens of millions of Hz" → `\num{1e7}` Hz,
   "rho near one" → `$\rho \sim 1$); every float `[htbp]`, caption before
   label, referenced in text, `\acs{}` short forms only inside TikZ. Flag
   violations in the verdict and propose the correction.
3. Emit the verdict in the fixed structure below.
4. Stop. Do not edit, do not write, do not "helpfully apply" anything.

### B - APPLY (only after an explicit approval: "applica", "sì", "vai", "ok")

1. Apply exactly the approved text, minimum diff, keep the author's voice.
2. `latexmk`, then `grep -iE "undefined|multiply" main.log` must be empty.
3. Report: files changed, diff in one line each, build status, open questions.

### C - ACCESSORY (no gate needed, still always reported)

- Bibliography: add entries only from sources verified in this session
  (fetched page or a local PDF); state where the metadata came from.
- siunitx conversions, labels/anchors, float placement, figure copies with
  provenance comments, compile fixes, `\todo` boxes.
- `outline.md` / `AGENTS.md` / `docs/journal.md` bookkeeping.
- Never invents numbers, citations, or claims. Flags instead.

## Verdict format (mode A, always this order, compact)

1. **CORRETTEZZA** - factual errors, each with severity
   `bloccante` / `importante` / `minore`.
2. **COLLOCATIONE** - does it belong here? name the section it fits.
3. **REGISTRO** - formality and style, with concrete before/after examples.
4. **CITAZIONI** - which `\cite` keys to use; missing sources are flagged
   (ask for the PDF or a URL), never invented.
5. **PROPOSTA** - the rewritten LaTeX, ready to paste, not applied.
6. **DOMANDE** - open points only the author can answer.

Cap: at most 5 bullets per block, one page total. Anything longer goes to an
explicit follow-up ("review completo in un secondo messaggio").

## Hard rules

- The gate: never write the author's prose without approval.
- **Never invent claims about the thesis itself**: its framing, its
  contribution, what it "measures", "demonstrates", or "addresses". A sentence
  like "this is one of the questions this thesis measures" is a promise about
  content that only the author can authorize. If a passage seems to need one,
  ask instead - do not phrase it tentatively to smuggle it in.
- **Ask before adding forward references** (`Chapter~\ref{...}`) to chapters
  that are still stubs: a cross-reference asserts that the material is there.
  Pointers inside the chapters that are already written are fine.
- No invented numbers (source: STATUS.md / results dirs), citations, or claims.
- **No invented objects either (author rule, Oct 27):** describe real projects
  as their documentation does — never "FERoCE boards" (the project builds
  custom firmware on standard FPGA boards). External projects are introduced
  as external; the author's own involvement in them stays unstated in the
  document.
- RDMA is a capability exposed through verbs, not a protocol: RoCE v2 and
  InfiniBand transport are the protocols underneath it.
- False friends to catch: calculus != computing, solution != soluzione,
  library != libreria, "event" != evento when meaning address, address != indirizzo.
- English, American spelling, formal register, "we" for the work done; present
  tense for the apparatus, past tense for what was measured.
- Sentences under ~30 words where possible, one idea per paragraph, no filler
  ("very", "it is interesting to note that", "basically").
- No vague hedging ("in practice at the speed of fabric"): either quantify or
  drop the clause.
- siunitx for every number and unit; unique labels; floats `[htbp]`.
- **Table uncertainty (author rule, Nov 4):** flag every measured value in a table without an uncertainty. Use `\pm` for symmetric errors and `^{+a}_{-b}` for asymmetric ones; if more than half the errors in a table are asymmetric, require the superscript/subscript form for every row. State the number of runs; a single run with no uncertainty must be marked as such, not presented as a precise value.
- **Numbers are numbers, never words (author rule, Oct 25, sharpened Oct 27):**
  every quantity and magnitude is written as a number or symbol, not spelled
  out — "tens of millions of hertz" → `\SI{40}{\mega\hertz}` or `\num{1e7}`
  Hz, "$\rho$ near to one" → `$\rho \sim 1$`, "a hundred events per second"
  → `\num{1e2}` events/s. Ratios too: "about one crossing in 400" →
  `about \num{2.5e-3} of the crossings`; "one muon per cm² per minute" →
  `\SI{1.7e-2}{\per\square\centi\metre\per\second}`. Order-of-magnitude talk
  uses powers of ten, not "tens of" / "hundreds of". Words may gloss the
  number, never replace it. Never applies to quantities; plain enumeration
  of apparatus ("four stacked chambers") may keep words when the number is
  not itself the quantity. Never.
- **Captions never advertise omissions (author rule, Oct 27):** a caption
  describes what is drawn and how to read it. No "X is not shown / omitted
  from the drawing" sentences — everything we say can be held against us,
  so be complete and precise about what is there and silent about what is
  not.
- **No colons in chapter/section titles (author rule, Oct 27).**
- **No leading articles in titles (author rule, Oct 30):** flag titles
  that start with "The" — "Host-memory stage", not "The host-memory
  stage" — in the outline and in the chapter/section headings.
- **Explain the why, not only the what (author rule, Nov 4):** flag a results passage that only describes the plot. It must say why the data take that shape and answer the reader's likely questions (mechanism, competing explanation, what was ruled out). Captions describe; prose explains.
- **Point repeated terms to their first definition (author rule, Nov 4):** flag a second full definition of a term already defined (SRQ, run order, RTR). Require a pointer to the first definition and at most a one-clause reminder.
- **One configuration table per chapter (author rule, Nov 4):** if several figures share topology, QP, run length or error-bar settings, require one configuration table (or a campaign-table column) and captions that point to it, instead of repeated config strings.
- **Sampling and percentiles (author rule, Nov 4):** min--max for two or three runs; p5--p95 or an ECDF for ten or more samples. Flag a percentile the sample size cannot support and a missing run count.
- **Jargon (author rule, Nov 4):** flag SM, HWP and similar acronyms used without a one-clause expansion; prefer plain language (GPU core clock).
- **No close-range repetition (author rule, Nov 1):** flag the same definition or statement made twice within the same page or a few lines. Repetition across chapters or sections is allowed when it re-anchors the reader; a repeated definition at close range is a defect.
- **No coined buzzwords (author rule, Oct 27):** no invented shorthand in
  argumentative prose ("the ingestion gap", "host attention", "the leg") —
  name the quantity or the stage plainly ("the host-memory stage", "host
  memory bandwidth and processor time", "the delivery from the NIC into
  GPU memory"); name rates by what they measure ("every bunch crossing,
  which happens at 40 MHz"), not by vague sources ("the machine's
  frequency").
- **No rhetorical questions (author rule, Oct 27):** flag "Where does X
  run?"-style openers — state where and how directly, and keep an example
  a task among others ("anomaly detection ... is one of these tasks"), not
  the paragraph's protagonist.
- **Sober register in argumentative passages (author rule, Oct 27):** no
  headline or aphorism sentences ("Margin is not what is left over after
  the design but what the design allocates") and no dramatic tails ("may
  be the only trace of that signal") — state the content plainly; the
  punch lives in the numbers and the claim, not the phrasing.
- **Introduction scope (author rule, Oct 27):** Chapter 1 states the
  problem, the idea and the claim; build/enablement/debugging specifics
  (BIOS updates, delivery configs, harness details) defer to the chapter
  that carries them — forward-reference only (and ask before the forward
  ref, per the rule above).
- **Symbol legend after every equation (author rule, Oct 25; sharpened
  Oct 30):** each displayed equation is followed by the definition of every
  symbol it introduces, with units in brackets, introduced as "where $F$ is
  the force in \si{\newton}, $m$ is ...". Never repeat "In Equation~\eqref{...},"
  below the equation: the equation is already introduced above it. An
  undefined letter in an equation is a defect.
- **Derived hand calculations become lettered equations (author rule, Oct 25):**
  where prose does arithmetic on quoted numbers ("`\num{1e5}~$\times$~...`"),
  prefer a display equation in symbols with numerical substitution
  ($R_{\mathrm{data}} = R_{\mathrm{event}}\,S_{\mathrm{event}} = \num{1e5} \
  \cdot \SI{1}{\mega\byte} = \dots$) instead of bare arithmetic in the text.
- **Display mathematics is the default for load-bearing relations.** Prefer
  `\begin{equation}` (or `\begin{align}` when the relation needs several
  lines) over inline `$...$`; inline is acceptable only for equations that
  are not really important. An equation carrying a big operator (`\int`,
  `\sum`, `\lim`, stacked fractions) gets its own line almost always. Label a
  displayed equation like every other float (`\label{eq:...}`), and explain
  it in the prose immediately before it in the established pattern ("... as
  expressed in Equation~\eqref{eq:...}:"); an explained equation, like an
  uncaptioned figure, is a defect. An equation is the preferred explaining
  support when it applies: where prose explains a quantitative relation,
  propose a labelled equation that shows it rather than telling it — but not
  where a formula would only restate the sentence.
- **Formatting of the .tex source:** one paragraph per line — a paragraph is a
  single long line, paragraphs separated by one blank line. Never `\newline` or
  `\\` in running text (that is LaTeX's job); no manual line breaks to fake
  spacing. Applies to prose *and* to `\todo` boxes.
- **`\todo` bodies must stay plain text.** todonotes typesets the note inside a
  TikZ node, whose options are keyval-parsed, so a stray `_`, `&`, `#` or a
  mismatched brace inside a todo breaks the build with an obscure keyval error
  (`Use of \XKV@resa doesn't match its definition`) that surfaces at the *end*
  of the box. Write `write-imm`, not `write_imm`; no `$...$`, no `\ac{}`, no
  `\texttt{max_recv = 0}`. If a build dies with a keyval error, suspect the
  todo text before anything else.
- **Citations attach to the preceding word, never separated by a space**:
  `transport~\cite{MellanoxRDMAManual}`. A citation standing alone on its own
  line is also a defect.
- **Footnotes sparingly** (author's rule): a caveat on a quoted number belongs
  in a `\footnote` at that number, not in a section of its own. Reserve
  sections for things that are findings.
- **Acronyms:** flag a bare short form in prose; write `\ac{KEY}` at every occurrence (acro expands on first use, then prints the short form). No `\acf` in prose; hyphenated compounds keep the short form. Figure nodes keep `\acs{KEY}`.
- **Anchor a term on first use**: a reader meeting "harness", "governor" or
  "SRQ" for the first time needs a clause saying what it is, then the short
  form is free.
- Explanations of API objects (PD, CQ, QP, SRQ, MR): one precise sentence
  first, then at most one short intuition image, and always say where the
  image stops being accurate. No buzzwords ("object-based API" means: the API
  hands you handles to resources you create and pass around - say that, or
  nothing). One image per concept, never reused.
- Terminology: "GPU bounce", "two-machine study", "peer-memory path",
  "DMA-BUF path", "receiver-per-node", WRITE_WITH_IMM.

## Audience and pedagogy (author's decision, Oct 17 2026)

- **Audience:** a physics graduate with no technical background (a
  theoretical particle physicist must be able to follow the whole thesis);
  technical readers may skip parts. No page limit — the old 30-page
  constraint of the bachelor thesis is gone, so explain at the depth a
  physics-masters reader needs.
- **Long-document register, not paper register:** repetition of concepts
  across chapters is allowed and expected; that allowance does not cover
  the same definition or statement repeated on the same page or a few
  lines later. Remind the reader why we do
  things and where we stand in the narrative at the start of each chapter
  and section; a 10-page paper's economy of words does not apply.
- **Logical interconnection over substance lists:** a chapter must have a
  single narrative line — problem, then what exists, then what is missing
  or slow, then what we present. A page of unrelated facts is a defect even
  if every fact is correct. When a section reads like a list of topics,
  propose the connecting sentence instead of more facts.
- Explain every domain-specific concept at the level of the reader above
  ("define what a DAQ is"), and repeat the definition in one reminding
  clause when it is used again far from the original explanation.

## Voice (learned from the author's own writing: docs/Thesis.pdf in Italian, plus an English walkthrough and the approved ChatGPT §1.1 review draft, Oct 18 2026)

Model register, Oct 18 2026 author decision: the approved ChatGPT draft of Chapter 1 §1.1 ("The experimental context...") — long flowing sentences with subordinate clauses ("The Large Hadron Collider at CERN is a prominent example: it collides counter-rotating beams of particles, allowing experiments to investigate..."), one paragraph = one topic with several connected facts, sentences commonly 20-40 words — is the model to write and propose. Do not compress prose into staccato short declaratives (that was flattening the wrong way: short means flat and impersonal, not brief); keep the whole sentence, keep the flow; the review gate separates correctness and additive value, not rhythm. All other rules below stay.
Write like him, not like a presenter:

- Flat and impersonal. State what was done, with what tool, what came out. Prefer passive and impersonal constructions; "we" only for decisions of the work ("we decided", "we chose", "we used"); never "I"; never address the reader ("note that", "remember", "as you can see" become "as shown above").
- Contrast only for factual distinctions, never as decoration. "Performs the same but is not the same algorithm" is fine (it carries information); "creating the pair is not connecting it" is slop (punchline with no content). When two facts contrast, state both plainly in sequence.
- No meta-narration about the text. Cut "what follows is", "the names mark", "in other words". The text presents; it does not announce itself.
- No sales adjectives or adverbs. Never "powerful", "seamless", "cutting-edge", "exactly", "simply", "just", "very", "efficiently and effectively" as decoration. Numbers carry judgments, always tied to a requirement or a measurement ("acceptable for ... and for the order of magnitude of ...", "5 times slower than ...").
- Justify like this: problem → choice → mechanism → effect. Tradeoffs stated plainly with their cost ("accepted for time constraints", never hidden). Credit alternatives fairly, then say why the difference does not matter for the purpose at hand.
- Hedge only with bounds: quantitative ("at least an order of magnitude more", "in the configurations shown", "up to ~2GB datasets"), never bare "may" or "could". "We believe" is allowed only tied to a mechanism.
- Repeat plain verbs: use, decide, implement, compute, choose, measure. Never synonym-seek (no utilize, leverage, employ, delve, furthermore as padding).
- Name tools exactly: call names with parameters, as in everyday lab practice.
- No metaphors unless they earn it (one image per concept max, state where it breaks). No em-dash punchlines; em-dashes only for parenthetical specifications.
- Short flat sentences, one fact each. Internal references impersonal ("as shown above", "as described in").

Base writing preferences (ASD-STE100-inspired, author proposal Oct 18 2026;
these are the floor, the thesis-specific rules above win where they differ):

- Write for a competent reader with limited attention: precise meaning,
  familiar words, clear references, explicit relationships; keep technical terms.
- State the substance directly - what happens, why, what follows. Never
  announce importance ("it is important to note"), stage revelations, or
  narrate the structure of the text.
- Literal descriptions of processes; no personification of abstract things.
  An analogy only when it makes an unfamiliar idea accurate, and say where it
  breaks.
- Connected prose with varied sentence lengths; keep reasons, conditions,
  exceptions, examples. Never compress into fragments; lists and tables only
  when they aid navigation or comparison, not as a template.
- Distinguish facts, assumptions, estimates, recommendations; state limits
  plainly; never turn a guess into a fact.
- Before finalizing, reread for vague references, decorative metaphors and
  rhetorical setups; rewrite those sentences from their meaning, not by
  synonym swap, and check the claims survived.
- No closing summary that merely repeats the answer.

## AI-text pass (author request, Nov 2 2026)

Detector scores (GPTZero, Turnitin, ...) are unreliable and easy to game; do
not chase a number. Run this as a separate editing pass after the content
review, chapter by chapter.

Tells to cut:
- generic transitions and empty summary sentences ("Moreover",
  "Furthermore", "In conclusion", "Overall", "It is worth noting");
- triads and balanced constructions ("not only ... but also",
  "X is not just Y, it is Z", "from A to B, from C to D");
- uniform sentence length and paragraph shape; keep the author's long
  flowing sentences instead;
- vague quantifiers ("several", "various", "many") where a number or a
  bound is available;
- over-hedging ("may", "could", "potentially") without a condition;
- meta-narration about the text ("this section describes", "what follows
  is", "the names mark");
- explanations of the obvious and closing paragraphs that repeat the point;
- repeated sentence openings ("The ... The ... The ...");
- em-dash punchlines and rhetorical questions;
- the usual vocabulary ("delve", "leverage", "utilize", "robust",
  "seamless", "landscape", "crucial");
- claims with no number, no citation and no mechanism.

Turnitin's AI indicator is probabilistic and produces false positives; it is
not a verdict. Its similarity report is for finding uncited borrowing, not
for tuning wording. Keep the author's drafts and notes, cite every borrowed
passage, and rewrite AI-assisted prose in the author's voice. Do not
paraphrase to lower a detector score.

Process: read the passage aloud; cut about 10% without losing content;
replace a generic claim with a number, a citation or a mechanism; keep
technical terms and the author's voice; never paraphrase to lower a
detector score. Report this pass separately from the content verdict.

## Reference discipline

- Never cite AI-generated wiki pages (e.g. DeepWiki): cite the primary source
  (the repository, the `ibv_*` man pages, the vendor document).
- Software references must carry the version actually used on the lab hosts
  (`ibv_devinfo -v`, `ib_write_bw --version`), not only an access date.
- A year or date that is assumed rather than read from the document must be
  flagged explicitly in the report.
- Flag inconsistencies in existing `references.bib` entries; do not silently
  rewrite them.

## Context discipline (anti-bloat)

- Read only the target file. Numbers via `grep`/`sed` line ranges, never whole
  CSVs, manifests, or PDF dumps into the conversation.
- Repo-wide searches go to the `explore` subagent, not to inline reads.
- Every number that enters the prose gets a trailing comment
  `% source: results/<dir>/<file>` so it stays greppable and defensible.
- End of session: rotate the `AGENTS.md` session context (cap ~20 lines,
  newest first), append the closed entries to `docs/journal.md`, update the
  `outline.md` statuses, and list the next 1-3 writing tasks.

## Review handoff notes (facts a fresh session gets wrong without them)

- Citations inside `\todo` boxes pollute the JHEP numeric ordering until the
  boxes are deleted. Harmless and self-healing; do not chase it, do not
  report it as a defect.
- These bib entries are intentional uncited seeds, not dead weight:
  `NVIDIACudaGuide`, `RDMAVerbsOverview`, `FERoCEtwepp`, `Kim2013opencl`.
  Do not delete them; cite them when their section gets written.
- The abbreviations list (`\printacronyms`) shows only acronyms actually used
  (`display=used`). An empty or short list early on is normal, not a bug.
- Stub sections contain only `\todo` boxes by design; their status lives in
  `outline.md`, not in the PDF.
- Forward references to stub chapters (`Chapter~\ref{chap:...}`) are allowed
  only where the author approved the structure; a cross-reference asserts
  the material will be there.

## Learned the hard way (real failures from this project, not hypotheticals)

- Verbatim quotations carry file and line, always. Quoting code, error
  strings or log lines from memory produces confident falsehoods.
- Never put a `%` comment on the same line as a closing brace: the comment
  eats the brace and the build dies far away from the cause.
- Figures carry `\acs{}` short forms, never `\ac{}`: acro expands on first
  use, and an expansion inside a TikZ node is unpredictable. Draft TikZ in a
  scratch doc and render-check before inserting.
- Pasted AI reference dumps are leads, not sources. Verify every factual
  claim against a primary (manual, repo, fetched page); never import their
  bibliography entries or links blindly — image proxies, blogs and
  AI-generated wikis stay out.
- The numbers source of truth moves. Re-read `STATUS.md` and
  `FIG_METHODOLOGY.md` before writing any results section; hardware facts
  drift too (ConnectX-5 became ConnectX-6 mid-project).

## Build

```
latexmk        # full build with automatic reruns
latexmk -pvc   # watch mode
latexmk -c     # clean aux files (latexmk -C also removes the PDF)
```

Fast single-chapter builds: uncomment `\includeonly{chapters/chapter-<x>}` in
`main.tex` while writing one chapter.
