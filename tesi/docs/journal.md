# Writing / experiment journal

Append factual updates here: date, edited files, source used, decisions, unresolved questions.

- 2026-10-26 — Starter project created from the reusable LaTeX structure of a shared thesis repo. No third-party thesis prose or experiment-specific results retained.

## 2026-11-02 — Approved outline applied

- Authorization: the author approved applying the seven-chapter outline discussed
  in this session and explicitly required preserving the personalized title page.
- Scope: baseline reproduction on the six original TranAD+ benchmarks; the main
  simplification study uses OPS-SAT, SMAP and MSL, with OPS-SAT as the reference
  application. Architectural experiments are planned; their results and conclusions
  are not prewritten.
- Project source: `../../docs/papers/TOAST.pdf`, particularly WP1 and WP2.
  Hardware deployment, fault injection and irradiation are later TOAST activities;
  they are not requirements claimed as completed by this outline.
- Existing evidence to verify during writing: `../../results/legacy/`,
  `../../results/v10/study/`, `../../results/v12/study/` and
  `../../results/analysis_v10_v12/`. No numerical experiment results or unverified
  bibliographic entries were inserted in the thesis.
- Files updated: `main.tex`, `outline.md`, `README.md`,
  `chapters/chapter-introduction.tex`, `chapters/chapter-background.tex`,
  `chapters/chapter-datasets.tex`, `chapters/conclusions.tex`,
  `appendix/appendix.tex` and `docs/journal.md`.
- Files renamed and rewritten as chapter outlines:
  `chapters/chapter-methodology.tex` to `chapters/chapter-replication.tex`;
  `chapters/chapter-results.tex` to `chapters/chapter-ablation.tex`;
  `chapters/chapter-hardware.tex` to `chapters/chapter-architecture.tex`.
- Preserved: `frontmatter/titlepage.tex`, `metadata.tex`, the preamble, logos,
  abstract, acknowledgements and bibliography source. Only the main document's
  chapter includes were changed; existing visual settings were retained.
- Outline notes distinguish literature, measurements, computed quantities and
  proposed work; original and dense threshold searches remain separate protocols.
  Evaluation compatibility between campaigns remains an explicit verification task.
- Validation: `latexmk -pdf main.tex` completed successfully and generated `main.pdf`.
  The compiled contents contain the seven agreed chapters and Appendix A. Every
  section/subsection title matches `outline.md`, source links resolve and labels
  are unique. The contents page and the long ablation chapter title were visually
  inspected. The only LaTeX warning is the empty acronym list in the unwritten draft.
- SHA-256 checks confirmed that the title-page source, metadata, preamble and logo
  assets were unchanged. The existing abstract, acknowledgements and bibliography
  source were also left untouched. Local pre-edit source copies are available in
  `/tmp/thesis-outline-before-20261102/` for this session.

## 2026-11-05 — First introduction draft

- Authorization: the author explicitly requested writing Chapter 1 in English,
  following the approved outline, the project `.opencode` guidance and the style
  of the supplied master's theses. The chapter is a first draft for author review,
  not approved final prose. Other scientific chapters remain unchanged.
- Style references were read selectively: Bernardi, introduction, printed pp. 1--5;
  Tramarin, introduction, pp. 1--4; Ceoletta, introduction, pp. 1--5. The draft
  follows their progression from context to aims and scope, without copying prose.
- Scope: telemetry monitoring; computational motivation; the TOAST context;
  reproducibility, window/FF and architecture questions; contributions and roadmap.
  The original five section titles and existing chapter labels were preserved.
- Sources were verified against the supplied PDFs and primary publisher/author
  records: Hundman et al. (KDD 2018), Ruszczak et al. (Scientific Data 2025),
  Tuli et al. (PVLDB 2022), Yermakov et al. (SpaceOps 2025), Vaswani et al.
  (NeurIPS 2017), and Sze et al. (Proceedings of the IEEE 2017).
  Their identifiers, publication years, authors and pages in `references.bib`
  originate from those records, not from inferred metadata.
- Bibliographic primary sources:
  `https://arxiv.org/abs/1802.04431`;
  `https://www.nature.com/articles/s41597-025-05035-3`;
  `https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf`;
  `https://publications.spaceops.org/2025/0397`;
  `https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html`;
  `https://arxiv.org/abs/1706.03762`;
  `https://eems.mit.edu/wp-content/uploads/2017/11/2017_pieee_dnn.pdf`.
- TOAST source: `../../docs/papers/TOAST.pdf`, p. 1 (identity and objectives),
  pp. 6--7 (work packages). Cited as an undated internal proposal; Andrea Triossi
  is identified as principal investigator, not assigned as the document's author.
  Deployment, fault injection and irradiation remain broader proposed activities.
- Dataset roles: OPS-SAT is the main satellite application, with single-feature
  fragment inputs; SMAP is satellite telemetry; MSL is Curiosity rover telemetry.
  The latter datasets support complementary comparisons, not satellite-flight claims.
- Descriptions of thesis activities are supported by `../../results/legacy/`,
  `../../results/v10/study/`, `../../results/v12/study/`,
  `../../results/v13/architectures/`, `../../tranad_plus/opssat_pipeline.py`,
  `../../tranad_variants/data.py`, `../../tranad_variants/models/` and
  `../../tranad_variants/training/objectives.py`. Extended replications remain
  ongoing; the introduction includes no numerical performance or final ranking.
- No scientific figures were added: a model schematic belongs in the background
  chapter. Future source-derived figures must refer to the corrected OPS-SAT
  article where applicable (correction DOI 10.1038/s41597-025-05751-w).
- Edited files: `chapters/chapter-introduction.tex`, `references.bib`,
  `preamble/acronyms.tex`, `main.tex` (bibliography switch only), `outline.md`
  (Chapter 1 to `[draft]`), `README.md`, and this journal.
- The title page, personal metadata, global typography and other chapter sources
  were preserved. Bibliography output is enabled in `main.tex` after metadata
  loading, so `metadata.tex` remains byte-identical to the original.
- Validation: `latexmk -pdf main.tex` completed successfully. The final log and
  BibTeX log contain no undefined/multiply-defined references, bibliography
  warnings, or overfull/underfull boxes. All seven citation keys resolve and
  labels are unique. The introduction spans printed pp. 1--5 (PDF pp. 9--13)
  in the 23-page draft, with approximately 1,777 source-text words. Representative
  introduction pages were rendered and visually inspected. SHA-256 checks
  confirmed that the title-page source, metadata and global preamble are unchanged.
- Next writing tasks: author review of the introduction; then the agreed
  background chapter and the dataset/protocol chapter, using primary sources
  and source-linked experiment details rather than importing results into Chapter 1.

## 2026-11-05 — Background and reference models draft

- Authorization: the author requested the complete second chapter in English,
  following the agreed outline and the style of the first draft. Original-source
  figures or faithful reconstructions were explicitly requested. The new prose
  remains a draft; Chapter 1 and the other scientific chapters were preserved.
- The six sections and two TranAD+ subsections retain the approved titles.
  Chapter 2 is marked `[draft]`. The outline's obsolete statement that all
  architecture experiments were only planned was updated: the initial campaign
  is complete and additional seeds and controls extend it; Chapter 6 prose
  remains unwritten.
- Added verified bibliography: Chandola, Banerjee and Kumar, ACM Computing Surveys
  41(3), Article 15 (2009), DOI 10.1145/1541880.1541882; Goodfellow, Bengio and
  Courville, Deep Learning, MIT Press (2016); the exact TranADPlus source snapshot,
  revision 2f02268d1fb606d85e55e76422a5fc00a08dc6d5.
- Theory provenance: Chandola Sections 2.2.1--2.3 for anomaly categories and
  training-label conventions (`https://arindam.cs.illinois.edu/papers/09/anomaly.pdf`);
  Goodfellow Chapter 14 for autoencoders, reconstruction loss and capacity
  (`https://www.deeplearningbook.org/contents/autoencoders.html`);
  Vaswani Sections 3.2--3.5 for attention, feed-forward maps and positions.
- Published TranAD provenance: supplied PDF pp. 4--6 (printed pp. 1204--1206),
  Figure 1, Sections 3.3--3.5. The text distinguishes the published history/window
  encoders and adversarial/meta-learning formulation from the inspected TranAD+
  window-only implementation. No performance or deployment claims are inferred.
- Implementation provenance: `../../tranad_plus/TranADPlus/src/models.py`,
  lines 34--119, and `src/helpers.py`, lines 665--673, 726 and 831--849.
  The audited model and helper files match upstream revision 2f02268 byte-for-byte.
  Encoder and output-head sharing, decoder independence, broadcast squared focus,
  last-observed-point reconstruction, positive epoch-weighted MSE, pre-MSE output
  mixing, positional encoding and the absence of layer normalization/meta-learning
  are described as code behavior. Decoder mixing and threshold selection are
  deferred to Chapter 3.
- Mathematical provenance: feed-forward parameter counts and attention/FF
  operation scalings are analytical derivations, not measurements. Complexity
  assumes h*d_k=h*d_v=d. Physical input quantities are normalized in the model
  equations; scores are dimensionless. No numerical experiment results are given.
- Figure 2.1: `figures/chapter2/vaswani-attention.pdf`, original Figure 2 on p. 4,
  cropped from the official NeurIPS paper at
  `https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf`.
  Source SHA-256 d87d482d5ae7960e2e43d7dd6d21377e60e73e8fce1bf2a01aff7aca8a08c537;
  top-origin crop rectangle [143,68,472,244] points. Original 500-ppi panels and
  vector titles were preserved, not resampled.
- Figure 2.2: `figures/chapter2/tranad-original.pdf`, original Figure 1 from
  `../../docs/papers/TranAD.pdf`, PDF p. 4 / printed p. 1204.
  Source SHA-256 0eab4c8b52b2b4f92b1c9b778f851f12fc90198abbee8d2bd090b2e546efa2b8;
  top-origin crop [50,79,565,206] points. Vector content was preserved. Caption
  explicitly attributes the published architecture rather than the current code.
- Figure 2.3: inline TikZ derived from TranAD+ Figure 1 and the actual reference
  computation. Each repeated E/P denotes the same encoder/output-head parameters;
  D1/D2 are separate. The model source SHA-256 is
  3f6451beed30637248e005633d7c94d1adcede0aca0d53c53665d2a54c3deeef.
  Figure topology, query and focus were checked against the lines above.
- Edited sources: `chapters/chapter-background.tex`, `references.bib`,
  `outline.md`, `README.md`, this journal; added the two original-source PDF crops.
  A shorter Chapter 2 running mark avoids colliding headers while retaining the
  full heading and contents title. No global typography, title-page metadata,
  first-chapter prose, model code, datasets or scientific results were changed.
- Review: separate read-only checks of theoretical formulas and the reference
  model found no substantive errors. Their head-partition and matrix-dimension
  clarifications were incorporated. The implemented architecture variants in
  Section 2.6 are supported by `../../tranad_variants/models/` and
  `training/objectives.py`, with existing results in `../../results/v13/architectures/`.
- Next writing task: datasets and experimental protocol. This must document
  fragment labels, split/scaler provenance including the distributed SMAP/MSL
  preprocessing, test-label-based selection, point adjustment and AUC definitions.
- Validation: the full `latexmk -pdf main.tex` build succeeded. Final LaTeX
  and BibTeX logs contain no undefined/multiply-defined references, warnings,
  or overfull/underfull boxes. All chapter citations and labels resolve.
  The chapter contains 18 displayed equations and 3 figures, spanning printed
  pp. 6--14 (PDF pp. 14--22) of the 31-page draft. Figures, equations and
  running headers were visually checked after rendering. SHA-256 checks confirm
  that Chapter 1, title-page source, metadata and global preamble are unchanged.

## 2026-11-06 — Datasets and experimental protocol draft

- Authorization: the author requested the third chapter in English, following
  the existing outline and `.opencode` writing guidance, with reliable sources
  and simple figures. The author also explicitly requested removal of line
  numbering from the entire document. Chapter 3 is marked `[draft]`.
- Scope: dataset roles and input inventories; OPS-SAT fragment annotations and
  split; SMAP/MSL input semantics; normalization and temporal padding; training
  and validation; threshold selection, point adjustment and AUC definitions;
  resource measurement and experimental provenance. Experimental performance
  results are not introduced into this methods chapter.
- OPS-SAT literature: Ruszczak et al., Scientific Data (2025), Data Records and
  Table 1, `https://pmc.ncbi.nlm.nih.gov/articles/PMC12041257/`. Nine channels
  comprise three magnetometer channels and six photodiode channels. Eighteen
  supplied handcrafted fragment features are distinct from the single sampled
  value used as network input. The source supports varying sampling and weak
  fragment-level annotations, not precise within-fragment anomaly boundaries.
- OPS-SAT counts: `../../data/raw/OPSSAT/segments.csv`, SHA-256
  `dab2e01e873c93590692b50a7cc312d9cce4d5ba7e69ce8015603c47a8ffb97b`.
  Its MD5 `1f4b6ffd0359efac96a6698ff557e0f9` matches official Zenodo v1,
  DOI `10.5281/zenodo.12588359` (29 June 2024). The 2,123 fragments contain
  303,493 samples. Official training: 1,594 fragments, 1,273 nominal plus
  321 anomalous, 225,178 samples. Test: 529 fragments, 416 nominal plus
  113 anomalous, 78,315 samples. The article prints 1,494 training fragments,
  inconsistent with its own subtotals and released CSV; a cited footnote
  explains the use of the verified file count.
- OPS-SAT split: `opssat_pipeline.py prepare_data` and the architecture-run
  inventories. A per-channel approximately 20% nominal holdout gives
  1,019 optimization fragments and 254 nominal validation fragments. Adding
  all 321 official-training anomalous fragments gives 575 validation fragments.
  Sample counts are 123,313 optimization, 101,865 validation and 78,315 test.
  The split seed is 2026 (holdout RNG uses the implementation's +13579 offset),
  fixed across model seeds. No timestamp/channel-ID coordinate is added.
- SMAP/MSL sources: Hundman et al., KDD (2018), Sections 3.1, 4.2 and 4.3,
  `https://arxiv.org/html/1802.04431v3`; original repository README and
  `labeled_anomalies.csv`, `https://github.com/khundman/telemanom`.
  Input widths 25 and 55 include one telemetry value plus encoded commands;
  they are not independent physical-sensor counts. SMAP's 55 annotation rows
  contain 54 unique IDs because P-2 appears twice. Local labels use the first
  P-2 interval [5350,6575], via `.iat[0]`; the second source row gives
  [5300,6420]. There are 68 processed SMAP intervals and 36 MSL intervals.
  The chapter distinguishes entry counts from unique-track counts.
- Native inventory: sums of actual unpadded arrays in
  `../../data/processed/tranad_plus/{SMAP,MSL}/{Train,Test}/Tracks`.
  SMAP: 54 tracks, 138,004 training samples, 435,826 test samples, F=25.
  MSL: 27 tracks, 58,317 training samples, 73,729 test samples, F=55.
  Processed training labels are zero; test positives are respectively 55,922
  and 7,766 real samples. The chapter reports input inventories, not padded
  evaluation lengths or estimates inferred from published performance.
- Normalization provenance: Telemanom README states that supplied raw-release
  telemetry was scaled to (-1,1) using test extrema. Local preprocessing fits
  a further min-max transformation on training data. A positive affine
  transformation can cancel under that second scaling for nonconstant features
  without clipping/nonlinear steps; the text therefore documents inherited
  preprocessing rather than asserting local leakage from the README alone.
  OPS-SAT local scalers fit nominal optimization values separately by channel.
- Window provenance: `helpers.merge_tracks`, `TSDataset_tracks`,
  `opssat_pipeline.FragmentWindows` and `tranad_variants/data.py`.
  Native padding is performed per track before global concatenation/windowing;
  N+(K-1)(w-1) targets is the resulting analytical count. OPS-SAT windows stay
  within fragments and give one target per real sample. V14 controls use
  `real_point_mask` and `interior_mask` to compare common real/interior samples.
- Training/settings provenance: `tranad_variants/configs/baselines.json`,
  `training/objectives.py` and completed architecture manifests. Reference
  windows 10/10/50 and widths 8/50/110 (OPS-SAT/SMAP/MSL); reduced values
  5/5/25 and 4/25/55; batch sizes 256/1024/128; initial learning rates
  .001/.001/.0001; POT q=.01/.07/.01; one layer per Transformer stack;
  Adam, ordered batches, StepLR every 5 epochs with factor .9, 50 initial epochs.
- Evaluation provenance: `paper_metrics.py`, `pipeline_metrics.py`,
  `opssat_scoring.py`, `tranad_variants/evaluation/selection.py`,
  `TranADPlus/src/{omnianomaly,spot}.py`. The original-style pointwise paths
  inspect 21 alpha values; original Best F1 uses 200 uniform thresholds.
  OPS-SAT fragment validation inspects five alpha values, two calibrations
  and two aggregations (20 candidates; four for single-output models),
  selecting validation fragment ROC-AUC. The fixed-score control uses alpha
  .5 and the 99.5th percentile of training scores. The fragment aggregations
  are max and p95. Robust calibration uses median/IQR with a 1e-12 floor.
- POT details are reported as implementation behavior, separately from theory:
  initial empirical level .98, retry factor .999, locally bounded at 35
  attempts, static test execution, reset to the preliminary threshold at the
  first above-threshold test score and final averaging of recorded thresholds.
  These operations use scores; labels enter alpha selection and adjustment.
- Dense-search numerical audit: the historical below-minimum candidate can
  round back to the float32 minimum under NumPy 1.26.4; NumPy 2.2.6 promotion
  and the explicit dtype-aware V14 boundary avoid that issue. A read-only
  audit of 270 hyperparameter, 240 architectural and 30 continuation reports
  verified all saved dense F1 optima exceed the all-positive value 2P/(N+P).
  Minimum margin: .18577881966268162 for
  `toast13_opssat_encoder_only_reference_s2030` (.7233138336682838 versus
  .5375350140056022, N=78,315 and P=28,785). The endpoint issue therefore does
  not change the reported dense maxima. No experiment code or results changed.
- Metric equations are definitions/analytical identities, not measurements.
  The binary-after-adjustment AUC equals half the sum of sensitivity and
  specificity when both classes are present; it is distinguished from
  continuous-score ROC-AUC. Fragment-label propagation is explicitly weak.
- Resource provenance: saved A100-PCIE-40GB execution manifests and
  `resources.py measure`. PyTorch 2.8.0+cu128, CUDA 12.8, FP32, TF32 off;
  batch-one profiling uses 10 warm-ups and 100 timed calls for the architecture
  study, 50 timed calls for the earlier hyperparameter profiling. Analytical
  MAC estimates and four bytes per FP32 parameter are distinguished from
  measured runtime and from full checkpoint storage. No onboard claims made.
- Repetition provenance: combined v10/v12 input grid has six windows by five
  widths by three seeds 2026--2028 per dataset (270 runs). V13 has four models
  by four settings by five seeds 2026--2030 per dataset (240 runs). Completed
  continuation controls retain optimizer, scheduler and absolute loss epoch
  from 50 to 100 epochs. The 20 later OPS-SAT continuation runs launched by
  the author have not been received/verified and are not declared complete.
- Added verified references: official Telemanom data repository;
  Siffer et al., KDD 2017, pp. 1067--1075, DOI 10.1145/3097983.3098144,
  `https://www.amossys.fr/wp-content/uploads/anomaly-detection-with-evt-1.pdf`
  (Sections 3.3 and 4.1 for generalized Pareto tails and calibration).
- Figures: two inline vector TikZ schematics, drawn from the verified split
  and the native point-adjustment operation. The second is explicitly
  illustrative, not measured telemetry. Both are referenced and captioned;
  no anomaly boundaries are invented inside OPS-SAT fragments.
- Edited sources: `chapters/chapter-datasets.tex`, `references.bib`, `main.tex`,
  `preamble/preamble.tex` (line-number package/setup removal only), `outline.md`,
  `README.md`, and this journal. Title page, metadata, Chapters 1--2, model code,
  datasets and experimental outputs were preserved.
- Review: independent read-only checks covered primary dataset sources, actual
  preprocessing/evaluation and draft clarity. The normalization caveat was
  corrected to account for affine cancellation, and validation selection and
  random-seed wording were made precise. Detailed floating-point audit notes
  remain here rather than interrupting the chapter's main exposition.
- Validation: `latexmk -pdf main.tex` completed successfully after cleaning
  stale auxiliary line-number commands from the earlier draft. Final LaTeX
  and BibTeX logs contain no warnings, undefined/multiply-defined references
  or overfull/underfull boxes. All 11 chapter pages were rendered and visually
  inspected, with a further check of the final layout. The chapter spans
  printed pp. 15--25 (PDF pp. 23--33) in the 41-page draft, with four tables,
  two vector diagrams, seven displayed equations and approximately 3,657
  prose words. Line-number commands and package setup are absent from active
  LaTeX sources. SHA-256 checks confirm that title-page source, metadata and
  Chapter 1 retain their recorded hashes; no edits were made to Chapter 2.
- Next writing task: author review of Chapter 3; Chapter 4 should present the
  reproduction results with configuration provenance and clearly named
  evaluation protocols, separately from the main simplification study.
