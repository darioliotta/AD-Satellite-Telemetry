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

## 2026-11-08 — Baseline reproduction and adaptation draft

- Authorization: the author requested Chapter 4 in English, starting directly
  from TranAD+, with the initial-TranAD-attempts section removed. The author
  requested available results closest to the published paper, SMD POT retained
  at the current independently trained value near 67%, and an evidence-based
  account of the WADI NaN diagnostic. This is a draft for author review.
- Scope: replaced `chapters/chapter-replication.tex`; updated Chapter 4 in
  `outline.md` and the writing-status text in `README.md`. The chapter has five
  sections and four tables: effective configurations; published/local F1 and
  differences; local precision/recall/AUC-PA; and WADI numerical controls.
  No figure is needed for the presented comparisons.
- Numerical source of truth: `data/chapter4/benchmark-evidence.json`. It retains
  the twelve selected result sources and SHA256 hashes, effective profiles,
  recorded seeds (null for missing records), confusion counts, published rows,
  twenty recovered author-configuration records, and diagnostic source hashes.
  This small audit file contains scalar evidence, not raw datasets or models.
- Published reference: `../../docs/papers/TranAD+.pdf`, Tables 2--7 on PDF
  pages 6--7, visually verified from rendered originals. Dataset POT/BF F1:
  DSN_1k .9247/.8044; SMAP .8424/.8491; MSL .8901/.8196;
  SWaT .3290/.8239; SMD .7129/.8509; WADI .5084/.2788.
  The paragraph on p. 8 supports the finite-BF-grid caveat. PA and binary AUC
  are attributed to the implementation, not incorrectly to paper Section 4.2.
- Selection: native rows are retained independently per method for minimum
  absolute F1 difference from the rounded published entry. These are selected
  single-run results, not maxima over all metrics, seed means, confidence
  intervals, or unbiased estimates of expected training performance. WADI POT
  is separately marked as a diagnostic, rather than a finite-output result.
  Every BF entry is the original BF200; no dense result replaces a BF entry.
- Selected F1 POT/BF: DSN_1k .923432099416902/.8072652127454257;
  SMAP .8421503933191664/.8470132840679159;
  MSL .8928013985140788/.7903225806451613;
  SWaT .324939721423305/.8206691402924203;
  SMD .6703861721912263/.8406109554055622;
  WADI .5339018016778985 (masked diagnostic)/.27884117951370924.
  Exact P/R/AUC-PA and operating-point counts are in the evidence JSON.
- Configuration caveats are explicit. Tensor shapes take precedence over old
  integer/multiplier strings. The selected DSN POT width16 is not the author's
  width2064. The closest MSL POT row uses W50/FF16/L2 rather than the author
  W10/FF110/L3 configuration. Selected SWaT BF W50/FF102/L1 differs from the
  author W20/FF816/L4. Missing legacy seeds are not retrospectively assigned.
  Existing source identity is documented in the 2026-11-05 journal entry.
- SMD: the retained POT value .6703861721912263 comes from
  `../../results/legacy/SMD/_native_pickles/papercheck_SMD_pot_w20_ff76_l5_bs8192_s1_pot.pkl`;
  its configuration is E100/W20/FF76/L5/B8192/LR1e-4/q.001/seed1. Future SMD
  repetitions have not been received here and are not presented as complete.
  Author-weight evaluation .7170677332633927 is a separate fixed-weight check
  (`../../results/diagnostics/author_weights/smd20/fp32_20261103T163516090316Z/summary.json`),
  not substituted for the independently trained result. Low local precision
  rather than lower adjusted recall explains the local POT F1 deficit.
- WADI source checkpoint: `papercheck_WADI_pot_w50_ff2048_l2_bs1024_s1`,
  SHA256 07dfeccb49f278c0984790452aabc3723f1ca323a1a0e750088038002dd1a774.
  Finite FP64 inference is saved under
  `../../results/diagnostics/WADI/fp64_20261103T130744Z/`.
  The explicitly used mask is the authors' FP32 `nonfinite_masks.npz`, key
  `test_second`, under `author_weights/wadi/fp32_20261103T163516090316Z/`.
  Mask SHA256 0e1601b5f1c095b101de44c5495a50078540ef7368648767b3b851e79cdf53c7.
- Recomputed WADI diagnostic: all21 saved local alpha score files were loaded;
  test scores at that mask were set to NaN, training scores/labels/counts
  unchanged; original omnianomaly.pot_eval(q=.001) was run and the best POT
  alpha retained. Winning alpha .9 and threshold .22356700584968833 match the
  finite local winner. The resulting F1 .5339018016778985, precision
  .8503833515881709, recall .38909491831211784 and AUC-PA .6924501025010203
  reproduce the prior diagnostic. Counts TP3882/TN162141/FP683/FN6095.
  Unmasked local F1 .14351647543395116, TP7805/TN71838/FP90986/FN2172.
- The mask covers94226 of172801 test windows, zero-based indices78575--172800,
  including90303 nominal and3923 anomalous labels. The local FP32 log records
  the same nonfinite count but not its exact indices; exact local/author mask
  equality is not claimed. NaNs arise in D2 inference on finite inputs, not
  missing raw telemetry or a GPD fit. `0*NaN` still contaminates the original
  score at alpha1; `NaN>threshold` is false, suppressing direct alarms while
  retaining observations. PA is applied afterwards. This is not row deletion
  or a demonstrated improvement of a finite-output detector.
- Author-weight WADI controls: FP32 F1 .507694023193577, FP64 F1
  .15113853176655195, from the separately saved author-weight summaries.
  Their agreement/disagreement supports precision-sensitive score semantics.
  Train range approximately0--1 and test maximum268.29458618 are metadata
  observations. Squared focus/residual amplification without LayerNorm is a
  plausible mechanism; no retained operator trace proves the first overflow.
- Bibliography: all citations verified against existing entries
  Yermakov2025TranADPlus, TranADPlusReferenceCode and Ruszczak2025OPSSAT.
  No bibliography entry was added or altered. Chapter 3 supplies the existing
  POT/PA/AUC definitions; Chapter 4 references them without redefining them.
- Review: independent read-only audits checked every table/pickle/hash,
  effective configurations and seeds, literature/cross-references, WADI
  mechanics and causal qualifications. Minor wording and heading corrections
  were applied. No raw results, experiment code, datasets, title page,
  metadata, main.tex or Chapters 1--3 were modified.
- Build: overlapping verification builds initially conflicted in generated
  auxiliary files. Auxiliary files were cleaned and regenerated successfully;
  final latexmk checks were run sequentially. The final LaTeX and BibTeX logs
  have no warnings, undefined/multiply-defined references or overfull/underfull
  boxes. The draft PDF has47 pages; Chapter 4 spans printed pp26--32
  (PDF pp34--40). Its pages and all four tables were rendered and visually
  checked; the bibliography and personalized title page remain intact.

## 2026-11-08 — Window and feed-forward ablation draft

- Authorization: the author requested Chapter 5 using the same English/style,
  evidence and bibliography standards as Chapter 4, reporting what is already
  known and allowing later revisions after the forthcoming studies. Work resumed
  from the completed audits after a usage-limit interruption; no research runs
  were restarted or overwritten. The chapter is a draft for author review.
- Scope: `chapters/chapter-ablation.tex`, with eight concise sections, four
  tables, two vector figures and two displayed equations. Updated Chapter 5
  writing status/section titles in `outline.md` and status/folder documentation
  in `README.md`. Existing Chapters 1--4, title page, metadata, bibliography,
  main.tex, datasets, experiment code and raw results were preserved.
- Source: `../../results/analysis_v10_v12/runs_270.csv` plus all270 original
  integrated reports. The existing analysis figures were dense-search-primary;
  they were not reused. New figures and tables use original BF200/POT only,
  with PA and binary AUC-PA. No AUC-score or dense BF is substituted into the
  main results. Every selected operating point retains its own alpha/threshold.
- Persistent evidence: `data/chapter5/runs.csv` includes relative report paths,
  report SHA256, declared checkpoint SHA256, all8 BF/POT scalar metrics,
  confusion counts, effective dimensions, seeds, resource values and model hash.
  `data/chapter5/evidence.json` records the design, source revisions, four-corner
  means/SD/ranges, paired seed differences, interactions and resource reductions.
  `data/chapter5/README.md` documents definitions, scope and regeneration.
- Validation: all270 report hashes and canonical scalar entries checked;
  90cells, 3seeds per cell, 50epoch records per run; all540 AUCs verified against
  half the sum of sensitivity and specificity within1e-9. Learned parameter
  and weight-byte totals and recorded analytic MACs verified for every run.
  The existing original reports and metrics were not changed.
- Effective grids: OPS W[3,4,5,6,8,10], FF[2,3,4,6,8]; SMAP the sameW and
  FF[13,25,32,38,50]; MSL W[13,25,32,38,44,50], FF[28,55,69,83,110].
  Each dataset has48 v10 runs and42 v12 runs. Baselines are the largest
  W/FF coordinates (10,8), (10,50), (50,110). All principal four-corner
  endpoint comparisons belong to v10; this is asserted in the table builder.
- Fixed settings: L1/E50, batches256/1024/128, LR.001/.001/.0001,
  q.01/.07/.01, scaling/padding, ordered batches, no downsample, zero workers.
  OPS nominal optimization/validation/test fragments are fixed. NASA native
  per-track padding changes duplicated targets and minibatch counts withW;
  equal50epochs is not exactly equal optimizer-step counts.
- Campaign qualification: model hashes match, and OPS scoring hashes match;
  runner/evaluation revisions differ. The historical combined-analysis
  provenance explicitly did not certify full numerical scoring parity. The
  chapter states this limit and treats the full grid as descriptive; it does
  not infer equivalence merely from a shared model hash. The headline corner
  comparisons avoid crossing those source revisions.
- Endpoint means±sampleSD, BF200/POT F1 in percent, ordered reference,
  window-only, FF-only, joint: OPS88.34±.58/65.74±3.01;
  84.59±6.61/72.31±22.51;84.05±7.46/56.54±8.31;91.28±.18/88.49±.85.
  SMAP82.90±1.06/81.38±.98;85.26±.63/84.53±.09;
  83.88±1.03/81.92±.22;85.76±.15/84.55±.09.
  MSL51.26±17.61/85.69±.22;73.71±3.41/85.56±.65;
  47.62±21.56/85.61±.58;72.26±16.53/85.71±.78.
  All exact values and operating-point P/R/AUC statistics are retained in JSON.
- Interactions: the displayed four-corner contrast is computed perseed, not
  from separately chosen maxima. OPS BF200 interaction10.9877046337±1.4731123945
  percentage points. OPS joint BF/POT gains+2.94154/+22.74926pp and SMAP
  +2.86066/+3.16928pp are positive in each matched seed. MSL joint BF changes
  +48.45163,-15.45693,+30.01631pp; the mean gain is not a robust perseed gain.
  MSL POT joint-minus-reference mean+.023887pp is effectively stable at this
  descriptive resolution. No significance or confidence claim is made.
- Resource equation P=82F^2+47F+(12F+3)*FF is counted for this single-layer
  implementation and checked against all270 models. It is not a universal
  Transformer formula. Parameter counts are independent ofW; positional
  buffers are distinct. Largest-to-smallest learned parameter/MAC totals:
  OPS249/2124→159/364; SMAP67575/464700→56364/135500;
  MSL323565/11008140→269199/2297900. Learned storage is4bytes/parameter.
  Relative parameter reductions36.14/16.59/16.80%; MAC reductions
  82.86/70.84/79.13%. Counts exclude softmax/elementwise/backward/optimizer.
- Timing sources: original model_statistics/profiling records, pipeline_metrics.py
  and opssat_pipeline.py. A100-PCIE-40GB, PyTorch2.8.0+cu128, FP32, batch1,
  10warmups/50synchronized wall-clock forwards. Three seed-specific medians
  summarized by mean/sampleSD. NASA input is synthetic; OPS uses a real
  test window. No verified blanket claim about v10 TF32 flags is introduced.
  Latencies around1.8ms do not track the large MAC reductions; no integer
  kernel, onboard speedup or measured energy claim is made.
- Interpretation: reconstruction includes the current target; less past does
  not remove that target. Capacity, score distributions, BF200 grid and PA
  can affect detection differently. MSL BF variability is discussed alongside
  stable POT, not as proof of better raw score ranking. Feature count1/25/55
  is not an isolated physical factor; NASA coordinates include command inputs,
  and baseline FF/embedding ratios are4/1/1. No overfitting diagnosis,
  physical correlation time or uniquely identified spacecraft mechanism is
  claimed. Architectural conclusions remain in Chapter 6.
- Figures: `figures/chapter5/grid_bf200.pdf` shows mean BF200 and sampleSD in
  separate matched panels for all90cells. `slices_bf200_pot.pdf` shows fixed
  largest-FF/window sweeps with mean/min--max ranges over3seeds. Both are
  vector PDFs; PNGs are inspection previews. Red/gray identify BF200/POT;
  labels are short and detailed conditions are in the captions.
- Regeneration: `scripts/prepare_chapter5.py` verifies scalar evidence and
  writes three complete tabular files under `tables/chapter5/`;
  `scripts/plot_chapter5.py` validates the complete grid and generates figures.
  Neither script performs model training or inference. No new bibliography
  entries were added; code/efficiency citations use existing verified keys.
- Final validation: `prepare_chapter5.py` rebuilt and verified all270 source
  reports and all540 AUC identities. Figures were regenerated from the
  persisted CSV; all10 initially rendered chapter pages were inspected, then
  a one-line spill page was removed by shortening redundant closing prose.
  The final chapter spans printed pp33--41 (PDF pp41--49), with a54-page
  complete draft. Final sequential latexmk passes all succeeded; LaTeX/BibTeX
  logs contain no warnings, unresolved/multiply-defined references or overfull/
  underfull boxes. The final end page and all table/figure layouts were checked.
  Protected title-page/metadata/bibliography/main and Chapters1--3 have no diff.
  Chapter4 changes from the preceding session were retained without alteration.


### 2026-11-08 — SMD POT update

Verified every artifact hash in results/v16/smd/author_pot/seed_3/status.json.
Chapter 4 now selects formats.fp32.oracle.pot in that run's metrics.json:
F1 68.61043844384682%, precision 53.45988280583316%, recall
95.74446406738215%, AUC-PA 96.06631611746579%. Published POT F1
71.29%; difference -2.68 percentage points. The selected seed changes from
1 to 3; W20/FF76/L5/E100/B8192/LR0.0001/q0.001 are unchanged.
The former selection remains recorded in benchmark-evidence.json. This is
a retrospectively selected single run, not a seed average or an independent
validation estimate. SMD Best-F1 and other datasets are unchanged.


### 2026-11-08 — Source consolidation

At the author's request, the single current source root is tranad_plus/.
Historical references to tranad_variants now resolve to tranad_plus/core;
tranad_v14, tranad_v15 and tranad_v16 resolve to campaigns/exploration,
campaigns/temporal and campaigns/precision within that root. Old journal
entries and result manifests retain their historical paths. The explicit
relocation hash record is tranad_plus/relocation.json. Numerical core ASTs
were checked after canonicalizing imports; bootstrap/entry-point path changes
are separate. All 240 v13 parent source manifests remain compatible. Data,
results, checkpoints, model objectives and metric computations were not
changed by the relocation. A new integer-inference campaign is proposed as
v17; no measurements from it are included in the thesis yet.


## 2026-11-10 — Approved introduction and chapter structure

- Applied the author-approved sections 1.1–1.5, including quantization in the research questions and chapter roadmap.
- Added a separate Chapter 7, Quantization, as a section outline; conclusions become Chapter 8. No experimental results were written into the new chapter.
- Removed the reproducibility appendix source and include, and repaired its cross-references in Chapter 3. Experiment artifacts, provenance and datasets were preserved.
- Removed the visible draft to-do list, all to-do boxes and the todonotes dependency. Recolored the two orange diagram styles to neutral gray without changing diagram content.
- Updated main.tex, outline.md and README.md consistently. The personalized title page was not edited.


## 2026-11-10 — Approved concise TranAD section

- Applied the author-approved TranAD text, with attention/transformer concepts integrated into section 2.3; removed the standalone attention section and its figure inclusion. The original figure asset is preserved.
- Retained the original TranAD Figure 1 crop (paper p. 1204). Sources: Vaswani et al., sections 3.1–3.5; Tuli et al., sections 3.3–3.4.
- Repaired the two downstream references to removed positional/FFN equations without changing their formulas; updated section numbering in outline.md. TranAD+ prose remains pending review.


## 2026-11-11 — Approved TranAD+ architecture and objective

- Applied the author-approved section 2.4.1, retaining the existing TikZ model diagram. Simplified the opening sentence and separated the alpha-weighted reconstruction from the anomaly-score equation.
- Preserved the author's sentence structure and corrected the punctuation before “where”. Sources: Yermakov et al., section 3.3; reference models.py:34–60 and helpers.py:665–673,831–849, revision 2f02268. No experimental results or bibliography entries changed.
- Figure provenance: the TranAD+ diagram is a TikZ reconstruction based on paper Figure 1 and the inspected code, not an image crop; repeated encoder/output-head boxes denote shared parameters.


## 2026-11-11 — Verified TranAD+ motivation and actual changes

- Updated section 2.4.1 with the DSN numerical/categorical/missing-data motivation (paper abstract and introduction), pipeline corrections and reproducibility resources (README Changes from TranAD).
- Replaced section 2.4.2 with six concrete model/pipeline/resource points: FF width and bottleneck definition, output mixing, full training-track loading, initial window duplication, configurable training settings, and published reproduction resources. Hyperparameter optimization is attributed to the authors; no causal metric gain is asserted.
- Compared original TranAD revision 7ffb98d0c18189cc3d9ab732b4cb0278200a0af0, resolved using the official GitHub commits/main API. Added a pinned code bibliography entry. main.py:16–40 verifies the window indexing and fixed SMAP/MSL/SMD track selection; main.py:245–275 verifies positive weighted reconstruction loss and second-output-only inference; models.py:456–488 and dlutils.py:192–229 verify the window-only architecture and lack of causal masks/layer normalization. These are inherited, not TranAD+ innovations.
- Executed the extracted original convert_to_windows and actual TranAD+ dataset/merge functions on a distinct-valued synthetic track: original first two windows identical; revised padded dataset advances its final target from the first to the second observation. Both assertions passed. This verifies indexing behavior, not a measured benchmark improvement.
- Compared TranAD+ models.py:34–60, helpers.py:96–128,665–673,831–849, datasets.py and scripts/training_loop.py:123–164. README changes were cross-checked against the official GitHub README.
- Inspection also confirms FP64/AdamW in the original entrypoint versus FP32/Adam in the TranAD+ entrypoint; not added as a new explanatory thesis claim in this revision.
- SHA-256 TranAD main.py: `0529c581d3d871bf5cc32316b45428ce9642650b8a0fa2e40e754341f57d4c0c`.
- SHA-256 TranAD src/models.py: `43a30e7caa7c8b28e91b5e875c724219c9c15789da1f6009bb2015fb1f81d84b`.
- SHA-256 TranAD src/dlutils.py: `35265d47f3c1c0f999590ab25cf47774cfd2293c99ca3f842419f31dfef750fe`.
- SHA-256 TranAD+ preserved README: `2b527795e3abc7026601a232983b926a95610a7a60851c65993d9c71a98ea5c7`.
- SHA-256 TranAD+ src/models.py: `3f6451beed30637248e005633d7c94d1adcede0aca0d53c53665d2a54c3deeef`.
- SHA-256 TranAD+ src/helpers.py: `7af3ace8879174ca24068a278722efd362bb9bbc893c7a2e433ca0bdaa73879a`.
- SHA-256 TranAD+ src/datasets.py: `9389c4bf08144eba1739e979cde30bf54e122f7c2dafbf2fa861d96b599dbfcb`.
- SHA-256 TranAD+ scripts/training_loop.py: `183465d71e8a092739b10bbdf1242a7822712fa992dc9069ae8c0dde47adbcb5`.


## 2026-11-11 — Author-directed simplification of sections 2.4.1–2.4.2

- Removed the pipeline/reproducibility overview paragraph from 2.4.1; retained the concrete single-track versus full-training-split issue with numeric bibliography citations.
- Removed explicit paper/README comparison framing, bold item titles, the training-tracks item and the reproducibility item from 2.4.2. Reduced bottleneck notation to a single width mapping with prose explaining contraction/expansion.
- Restored the adversarial-loss comparison specifically against the published TranAD formulation described in section 2.3, without claiming that the released TranAD training code has an adversarial objective. Non-negative weights correctly include the zero second-branch weight at epoch 1.
- Kept output mixing, verified initial-window duplication correction and configurable training settings. No changes to experiments or bibliography.


## 2026-11-11 — Concise resource background

- Shortened section 2.5 to three paragraphs distinguishing parameter count, numerical storage, inference memory, operation estimates and hardware/software-dependent execution time. Removed detailed architectural previews and unused FF parameter formula.
- Moved the analytical attention/FF computation scaling to chapter 5, section Detection performance and computational cost, where it is used; renamed its label and repaired the existing reference. No measured results changed. Sources remain Vaswani (attention/feed-forward operations) and Sze (resource and implementation dependencies).


## 2026-11-11 — Resource discussion connected to TranAD+

- Rewrote section 2.5 with connected prose linking the resource discussion directly to TranAD/TranAD+ and future onboard use. The existing shared encoder illustrates why stored parameters and executed operations differ. Retained storage/precision, intermediate memory and hardware-dependent execution time; no previews of new variants or changes to experiments.


## 2026-11-12 — Chapters 1–2 aligned with the evaluation and lightweight-detector direction

- Approved by the author. Chapter 2: hyperparameter-optimization statement now attributed to the reference-code documentation (TranADPlus/README_original.md:130); added that min-max scaling is fitted on training data without clipping, so test inputs can leave [0,1] while the sigmoid output cannot; added section 2.6 "Lightweight detectors without transformers" (statistics descriptors, nearest neighbour, linear subspace, fully connected autoencoder; Ruszczak, Chandola, Goodfellow). No numbers.
- Chapter 1: reference model (not all models) is transformer-based; added research questions 5 (evaluation protocol) and 6 (lightweight alternatives) and one sentence each in scope and contributions. Chapter organization list for chapters 3+ not yet updated (structure pending). `latexmk -pdf main.tex` compiles without errors or undefined references.


## 2026-11-12 — Chapter 3 cleanup

- Approved by the author. Rewrote chapter 3 for clarity and concision without changing content, numbers or labels (all \ref targets preserved). Previous draft saved as docs/chapter-datasets_before_2026-11-12_cleanup.tex.
- Source comments updated to the current repository layout (tranad_variants/ -> tranad_plus/core/, tranad_v14/ -> tranad_plus/campaigns/exploration/); every cited function verified to exist. Added sources for bf_search, calc_point2point (binary AUC), adjust_predicts and the 0.995 training quantile (opssat_scoring.evaluate_selected, campaign_plan). Verified Eq. native-window-count on MSL (w=50, K=27, N=73729 -> 75003 = length of saved v13 test scores).
- Removed only redundant sentences (GPU-resident caching, batched-scoring timing, generic remarks on hashes). `latexmk -pdf main.tex`: no errors, no undefined references.
