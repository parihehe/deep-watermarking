# Proposed Contributions and Research Plan — Phase 0

## Inherited versus proposed work

| Category | Item | Claim allowed now |
|---|---|---|
| Base-paper reference | YIQ + Catalan transform + NMF + ANN watermarking | Existing published method |
| Baseline foundation | DWT + SVD singular-value embedding + 1D-CNN blind decoding | Existing/supplied baseline to reproduce; provenance pending |
| Proposed contribution C1 | Reproducible baseline implementation with configuration, tests and held-out protocol | Planned artifact, not completed |
| Proposed contribution C2 | Content-aware adaptive embedding strength map | Hypothesis under evaluation |
| Proposed contribution C3 | Attack-aware training and standardized attack simulator | Hypothesis under evaluation |
| Proposed contribution C4 | Systematic capacity/quality/robustness study | Planned evaluation contribution |
| Proposed contribution C5 | Optional ECC with overhead-aware evaluation | Planned evaluation contribution |
| Proposed contribution C6 | Ablation, generalization and false-positive analysis | Planned validation contribution |

## Mathematical baseline to validate later

For a grayscale/luminance working image \(I\), one-level DWT will produce \((LL,LH,HL,HH)\). The selected subband is decomposed as:

\[
LL = U\Sigma V^T.
\]

The exact baseline bit-to-singular-value modulation rule is intentionally **not frozen in Phase 0**. The brief gives a generic additive candidate, \(\sigma'_i = \sigma_i + \alpha f(w_i)\), but requires experimental justification rather than arbitrary selection. Phase 6 will select and document one rule for the baseline, including allocation, index mapping, clipping/range behavior and blind-decoding implications.

This restraint matters: a simple additive singular-value rule can be difficult to decode blindly after image processing unless the decoder learns a stable, carefully defined signal.

## Sequential execution plan

| Milestone | Phases | Outcome |
|---|---|---|
| Foundation | 0–2 | research boundary, environment and legally documented, split dataset |
| Signal-processing baseline | 3–6 | deterministic payload generator, DWT/SVD tests and fixed embedding |
| Learned baseline | 7–9 | blind CNN decoder, reproduction record and frozen robustness baseline |
| Proposed variants | 10–15 | adaptive embedding, attack-aware training, capacity, wavelet/subband, decoder and ECC studies |
| Validation | 16–20 | frozen final model, ablation, generalization, security and performance results |
| Product/reporting | 21–26 | tracking/dashboard, demo, containers, final experiments, report and viva material |

Every phase ends with tests, saved configuration, artifacts, documented limitations and explicit approval before the next phase.

## Decisions made in Phase 0

1. Treat the existing DWT–SVD architecture as a required baseline specification until its source is supplied and verified.
2. Do not make comparative performance or novelty claims before experiments.
3. Use DIV2K’s published 800/100/100 partition where access and licensing permit; do not mix external data until Phase 18.
4. Keep research code independent from the existing Dayflow app in this workspace. No Dayflow files are modified in Phase 0.

## Phase 0 exit criteria

- [x] Base paper identity and high-level methodology verified.
- [x] Base-paper, baseline and proposed work separated.
- [x] Research gap, questions, hypotheses and contribution boundaries documented.
- [x] Sequential experiment strategy specified.
- [ ] DWT–SVD foundation full text/citation verified — unresolved external input.
