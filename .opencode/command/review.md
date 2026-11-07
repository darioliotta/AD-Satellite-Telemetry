---
description: Peer-review the prose just written in a thesis chapter section, without touching any file.
---

Load the `thesis-draft-review` skill and work in **mode A (REVIEW)**.

# Objective

- Review the author's new prose and return a verdict the author can act on.
- No file writes in this mode. The gate is the whole point of this command.

# Method

1. Identify the target: the file and section the author names; if they do not
   name one, use the last `\section{...}` of the chapter they last touched.
2. Read only that section (`grep -n '\\section' <file>` first, then read the
   line range). Do not read other chapters, STATUS.md, or the results dirs.
3. Cross-check only what the section needs: terminology against `AGENTS.md`,
   numbers against `benchmarks/bare-metal/remote/STATUS.md` (grep the specific
   values), citation keys against `references.bib`.
4. Emit the fixed verdict structure, in this order:
   CORRETTEZZA (with severity) / COLLOCATIONE / REGISTRO / CITAZIONI /
   PROPOSTA (ready to paste) / DOMANDE.
5. Stop after the PROPOSTA block. Do not apply it.

# Output requirements

- One page maximum. At most 5 bullets per block.
- The PROPOSTA must be complete LaTeX for the passage, ready to paste, not a
  diff, not a description of a diff.
- Any missing source is a question for the author, never a fabricated entry.
