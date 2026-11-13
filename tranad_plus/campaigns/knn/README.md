# V21 — Lightweight nearest-neighbour detector (KNN) versus TranAD+

Code: `tranad_plus/campaigns/knn/`. Results: `results/v21/knn/`.
No TranAD+ training. Existing V13 score arrays and V20 tables are read-only.

## Question

TranAD+ and its simplified/quantized variants show near-chance threshold-free
discrimination (V20 AUC-score) while point-adjusted F1 stays high. Does a
model-free descriptor of nominal behaviour do better under the same data
and an explicit, label-safe protocol?

## Detector (identical recipe on every dataset)

1. Describe a piece of signal with a few elementary statistics.
2. Standardise each statistic with nominal training pieces (median and
   standard deviation, floored at 1e-6 relative): per channel for OPS-SAT;
   pooled over all training tracks for SMAP/MSL, because some single training
   tracks are constant. The nearest-neighbour reference is still per
   track/channel: SMAP/MSL test windows are compared only with their own
   track's training windows.
3. Score = mean Euclidean distance to the k nearest nominal training pieces.
4. Alarm = score above a quantile of nominal scores (no anomaly label used).

Statistics (8 for OPS-SAT, 9 for SMAP/MSL): standard deviation, range, 1% and
99% quantiles, kurtosis (OPS-SAT) / mean, min, max, last value (SMAP/MSL),
standard deviation of first differences, maximum absolute first difference,
fraction of zero first differences, plus last difference for SMAP/MSL.

| | OPS-SAT | SMAP / MSL |
|---|---|---|
| Piece | whole fragment | causal window of the last `w` samples, telemetry coordinate 0 |
| Labels | one per fragment | one per sample (annotated intervals) |
| Nominal reference | 1019 nominal optimisation fragments (same split as TranAD+, seed 2026) | each track's own training file |
| Choices | k ∈ {1,3,5,10,20}, chosen by **validation** AUC-ROC (254 nominal + 321 anomalous official-train fragments) | fixed a priori: k = 5, w = TranAD+ reference window (SMAP 10, MSL 50); no labelled validation exists |
| Threshold | 95th percentile of validation-**nominal** scores | 99th percentile of leave-one-out training scores pooled over tracks (same 0.99 rule as the V20 TranAD+ fixed-quantile endpoint) |
| Test | official 529 fragments, measured once | official test tracks, real samples only (no padded copies) |

Sensitivity to k and w on SMAP/MSL is reported separately and is **not** used
for selection.

## Controls on the same test data

- all-positive classifier and uniform random scores (5 seeds);
- linear control: PCA reconstruction error on the same standardised
  statistics (components explaining 90% of nominal variance);
- TranAD+ (V13 reference and encoder-only, reference W/FF, seeds 2026–2030,
  alpha = 0.5): OPS-SAT direct fragment score (95th percentile of point
  errors) and the same KNN applied to per-fragment error statistics (mean,
  95th percentile, max, std of log errors); SMAP/MSL values from V20.

## Metrics

Threshold-free: AUC-ROC, AUC-PR (average precision). At the label-free
threshold: precision, recall, F1, MCC; SMAP/MSL also nominal false-alarm rate
and event coverage. Paper-comparison only: BF200 point-adjusted F1 (test-label
oracle) next to the random-score control.

Run: `python -B -m tranad_plus.campaigns.knn.run` from the repository root.

---

# V23 — MLP autoencoder on the same statistics, and quantization (protocol fixed before running)

Code: `tranad_plus/campaigns/knn/mlp.py`. Results: `results/v23/lightweight/`.
Written 2026-11-12, before any V23 result was computed.

## Detector
Fully connected autoencoder on the standardised statistics of V21 (8 for
OPS-SAT, 9 for SMAP/MSL): `d → 16 → 3 → 16 → d`, ReLU after the two hidden
layers of width 16, linear bottleneck and output. Score = sum of squared
reconstruction errors of the statistics. No attention, no transformer.

Training (fixed a priori, no early stopping, no choice made on test or on
labelled validation): Adam, learning rate 1e-3, batch 64, MSE loss on nominal
data only; 300 epochs on OPS-SAT (1019 fragments), 30 epochs on the pooled
SMAP/MSL memory windows; seeds 2026–2030.

| | OPS-SAT | SMAP / MSL |
|---|---|---|
| Training data | 1019 nominal optimisation fragments (V21 split) | first 80% of every training track (V22 memory part), all tracks pooled: **one network per dataset** |
| Threshold | 95th percentile of validation-nominal scores | 99th percentile and POT (level 0.98, alpha 0.01) of the held-out last 20% (V22 rule) |
| Sensitivity (not used for selection) | — | one network per track, 100 epochs, FP32 only |

## Quantization (simulated: fake quantization in floating point, no integer kernel)
- Weights: symmetric, per output neuron, round-to-nearest, INT8 or INT4.
  Biases kept in floating point (in an integer pipeline they would be 32-bit
  accumulators).
- Activations (W8A8, W4A4): the input of every linear layer is quantized per
  tensor; signed for the input and the bottleneck, unsigned after ReLU. Scales
  = maximum absolute value on the training data; larger test values saturate.
  The output and the score are computed in floating point.
- Formats: FP32, W8 and W4 (weights only), W8A8 and W4A4.
- PTQ: quantize the trained FP32 network. QAT: start from the same FP32
  network, train 20% more epochs with the quantizer in the forward pass
  (straight-through gradient), learning rate 1e-4. QAT is run for W8A8 and
  W4A4 regardless of the PTQ result.
- The threshold is recomputed for every format on the same label-free
  calibration data.

KNN equivalent (no training, so no QAT): reference statistics and queries
quantized per feature (symmetric, scale from the reference), INT8 and INT4;
distances computed on the integer grid. Same k, window and thresholds as
V21/V22.

## Reported
AUC-ROC, AUC-PR, precision, recall, F1, MCC, observed nominal false-alarm
rate; SMAP/MSL also event coverage and the BF200+PA oracle (paper comparison
only). Resources: parameters, storage per format, multiply-accumulate
operations per decision (MLP) or stored values and distance operations (KNN).
TranAD+ rows (AUC by precision, parameters, MACs) are read from V13/V20/V22.

---

# V24 — Extensions (protocol fixed before running, 2026-11-12)

Code: `tranad_plus/campaigns/knn/extensions.py`. Results: `results/v24/`.
Same detectors, statistics, training settings and thresholds as V21–V23.

## A. Quantization of the per-track MLP (SMAP/MSL)
V23 measured the per-track MLP (one network per track) in FP32 only. Here the
same networks (100 epochs, seeds 2026–2030) are quantized with the V23 formats:
PTQ W8, W4, W8A8, W4A4 and QAT for W8A8 and W4A4 (20% more epochs, lr 1e-4).
Activation scales are calibrated per track on that track's training data.
Thresholds: V22 rule (99% quantile and POT on the held-out 20%).

## B. OPS-SAT per channel
Literature reference: Ruszczak et al. 2025, Table 4 ("specialized models",
one model per channel, metrics averaged over the 6 channels with anomalous
fragments in both training and test: CADC0872, 0873, 0874, 0888, 0892, 0894).
Same official test set; differences to declare: they use 16 handcrafted
features and train unsupervised models on the full training split with
contamination 0.2, while we train on nominal fragments only (V21 split).

Reported for every channel (with test counts) and as the mean over those 6 channels:
- specialized KNN: reference = that channel's nominal optimisation fragments,
  k = 5 (fixed: the value selected globally in V21; no per-channel choice);
- specialized MLP: one network per channel, V23 settings, FP32, 5 seeds;
- universal KNN and MLP (V21/V23 models, one for all channels) sliced by channel;
- TranAD+ reference and encoder-only (V13, alpha 0.5), direct fragment score of V21, sliced by channel.
Threshold: 95th percentile of the channel's validation-nominal scores (specialized)
or of all validation-nominal scores (universal, as in V21).

## C. Operation counts (analytical, per decision)
Elementary arithmetic operations (add, subtract, multiply, compare) for:
statistics (naive recomputation over the window or fragment), standardization,
detector (KNN distances; MLP multiply-accumulates and ReLU). Quantiles of the
OPS-SAT fragment features are counted as a sort, L·log2(L) comparisons.
These are estimates, not measurements on hardware.

Amendment before the full run (2026-11-12): a dry run showed that channel
CADC0886 has only 4 nominal optimisation fragments, fewer than k = 5. For the
specialized KNN, k = min(5, number of reference fragments of the channel).
This affects only CADC0886, which is not among the 6 literature channels.

---

# V25 — Direct comparison on identical scores (protocol fixed before running, 2026-11-13)

Code: `tranad_plus/campaigns/knn/comparison.py`. Results: `results/v25/comparison/`.
Purpose: one table where every detector is measured by the same code on the same
test samples, with three rulers: the paper protocol (BF200 + PA), threshold-free
ranking, and detection at an equal false-alarm rate.

Detectors (test scores):
- uniform random scores (5 seeds);
- TranAD+ reference and encoder-only (V13 reference W/FF, seeds 2026–2030),
  alpha fixed at 0.5 (not chosen on test), real samples only;
- KNN on statistics (V22 memory: first 80% of each training track; OPS-SAT: V21);
- MLP autoencoder on statistics, FP32, 5 seeds: one network per track for
  SMAP/MSL (V24 setting, retrained identically because V24 did not store scores),
  one universal network for OPS-SAT (V23 setting).

Metrics:
- paper ruler (SMAP/MSL): BF200 + point adjustment, best test F1 (oracle);
- AUC-ROC and AUC-PR on continuous scores;
- equal false-alarm comparison: threshold set so that exactly 1%, 5% and 10% of
  the test nominal samples (OPS-SAT: fragments) raise an alarm; report recall,
  precision, F1 and, for SMAP/MSL, event coverage (fraction of annotated
  intervals with at least one alarm). This threshold uses test nominal labels
  and is an analysis tool, applied identically to every detector; it is not a
  deployable threshold.

---

# V26 — Multivariate statistics on SMAP/MSL (extension, protocol fixed before running, 2026-11-13)

Motivation (declared: it follows the V25 result on MSL): V21–V25 use only the
telemetry coordinate, whereas TranAD+ uses all input coordinates (25 SMAP, 55 MSL;
the extra coordinates are binary command encodings). This extension gives KNN
and MLP the same input coordinates as TranAD+.

Code: `tranad_plus/campaigns/knn/multivariate.py`. Results: `results/v26/multivariate/`.
- The same 9 window statistics (V21) on every input coordinate, concatenated.
- Coordinates with zero variance over all training tracks are dropped (SMAP 1,
  MSL 22, counted before running), because their standardization is undefined.
  Remaining dimension: SMAP 24 × 9 = 216, MSL 33 × 9 = 297.
- Standardization pooled over training memory windows (V22 split), as before.
- KNN per track, k = 5; MLP one network per track, same architecture
  (d → 16 → 3 → 16 → d), 100 epochs, FP32, seeds 2026–2030. No tuning.
- Same metrics as V25 (BF200+PA oracle, AUC-ROC/PR, equal false-alarm 1/5/10%),
  same test samples; the V25 univariate rows are the reference.
