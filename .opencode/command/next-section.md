---
description: Decide and show the next writing step for the thesis, reading the minimum context.
---

Load the `thesis-draft-review` skill.

# Objective

- Answer one question: what does the author write next, and from which file.

# Context minimization

- Read `outline.md` and, for the target chapter only, its section list and
  `\todo` boxes (`grep -n '\\section\|\\todo' chapters/chapter-<x>.tex`).
- Do NOT read `STATUS.md`, the results directories, or other chapters unless a
  task below explicitly requires a number, and then only the matching lines.
- The `\listoftodos` page of the PDF is the authoritative task dashboard.

# Method

1. Cross `outline.md` statuses against the `\todo` boxes of the target chapter.
2. Respect the agreed writing order (Ch. 4, 5, 3, 2, 6, 7.1, 1, conclusions,
   abstract) unless the author redirects.
3. Output the next 1-3 tasks, in order, each with: the exact file and section to
   open, the material needed, and the path where that material lives.
4. Flag any blocking question only the author can answer (missing data,
   undecided framing, missing PDF for a citation).
5. No file writes in this command.

# Output requirements

- A short ordered list, not an essay. Paths only, no pasted file contents.
- End with a single line naming the one thing to do first.
