---
name: thesis-tikz
description: Create and audit inline TikZ diagrams for satellite telemetry, anomaly detection model pipelines, dataset splits and resource-aware execution.
---

# TikZ figures for the thesis

## Conventions
1. Diagrams are inline TikZ in a `figure` float (default `[htbp]`), with `\centering`, `\caption{...}` then `\label{fig:...}`.
2. Every figure must be referenced in the prose; diagram semantics must match verified methodology or cited literature.
3. Keep figures within text width (17 cm for the inherited A4 layout); small labels must remain readable.
4. Consistent palette: white, `black!8`, `blue!10`, `orange!18`, rounded corners, `Stealth` arrows; explicitly load additional TikZ libraries if used.
5. Never invent blocks or edges in a TranAD+ architecture diagram. Check the actual implemented network, tensor dimensions and signal flow.
6. Distinguish dataset preprocessing, training, scoring and thresholding; keep data flow and model internals conceptually separate when appropriate.

## Workflow
1. State the one claim the figure is meant to clarify.
2. Decide a suitable geometry: pipeline for preprocessing; block flow for the model; split diagram for train/test; resource graph for deployment.
3. Name source evidence (implementation file, experiment report, cited paper) for every nontrivial shape/connection.
4. Draft TikZ with legible labels and meaningful caption. Keep labels off arrows and inside margins.
5. Run `latexmk -pdf main.tex`, check warnings and visually inspect the PDF. If graphics cannot be inspected, state the limitation.
6. Report modified file, figure label, and any aspect requiring experimental verification.

## Reusable minimal pattern
```latex
\begin{figure}[htbp]
  \centering
  \begin{tikzpicture}[node distance=1.8cm, >=Stealth]
    \node[draw, rounded corners, fill=blue!10] (a) {Input};
    \node[draw, rounded corners, fill=black!8, right=of a] (b) {Output};
    \draw[->] (a) -- (b);
  \end{tikzpicture}
  \caption{Schematic processing stages (replace with verified pipeline).}
  \label{fig:example-pipeline}
\end{figure}
```
