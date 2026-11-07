# Skill routing

Load the matching skill yourself; do not wait to be told.

| When the user says / asks for | Load |
|---|---|
| "review this prose", "check this passage", "review" | `thesis-draft-review` (mode A, no writes pre-approval) |
| "next section", "what do I write next", "next-section" | `thesis-draft-review`; then read outline.md and the target chapter's `\section` list only |
| "write/fix LaTeX", "figures", "tikz", "compile", "number check", "structure", "siunitx/cites" | `thesis-writing` |
| "tikz", "diagram", "figure", "label", "Fig 1.x" | `thesis-tikz` |
| "create/improve a skill", "eval" | `skill-creator` |
| "why is this broken", "compile error" (after reading the error) | `diagnosing-bugs` |
| "review this code/data", "is this over-engineered" | `ponytail` |

Rules:
- One task, one skill body. Project skills win: never fall back to a generic plotting/writing skill.
- Context discipline: grep for '\section{' first, then read only that range (sed -n 'A,Bp'); never read a whole chapter to locate one section.
- docs/journal.md is 1042 lines / ~58 KB: never cat it; grep '^\#\#' dates and read the matching block.
- AGENTS.md section context is read only when needed; prefer STATE.md.

Context hygiene:
- Never read a whole chapter; grep '\section' then slice.
- Never read docs/journal.md whole; grep '^\#\#' dates.
- Exploration-intensive work (figure redrawing, bib provenance deep dives) runs in a subagent; the main session reasons on its report.
- Do not re-read what STATE.md already records.
