# Paper and System Comparison — Phase 0

| Dimension | Base paper: CT–NMF–ANN | Supplied DWT–SVD foundation | Planned research system |
|---|---|---|---|
| Status | Verified 2025 IEEE Access article | Source identity/full text not verified; architecture comes from brief | Proposed and unimplemented |
| Transform/domain | YIQ I channel; 2×2 blocks; Catalan transform | DWT, initially single-level Haar; LL subband | Same baseline plus experimentally selected adaptive policy/subband variants |
| Matrix factorization | NMF | SVD, `A = UΣVᵀ` | SVD baseline retained |
| Embedding | Keyed shuffling and coefficient-difference rule | Fixed-strength modification of selected singular values | Fixed baseline first; then content-dependent strength |
| Watermark representation | Watermark image/signature; scrambling stated | Binary bits from UUID + SHA-256 (per brief) | Binary/text payload; optional ECC |
| Decoder | ANN trained on block statistics | 1D CNN over singular-value windows (per brief) | Baseline 1D CNN; later controlled decoder variants |
| Original image at extraction | Needs verification from full paper before a blind/non-blind claim | Intended to be blind | Must remain blind |
| Dataset | Verify from article experimental section before replication | DIV2K (per brief) | DIV2K train/validation/test, then documented external generalization sets |
| Quality metrics | Article reports PSNR, SSIM and NC | Exact reported protocol unknown | PSNR, SSIM, BER, bit accuracy, NC, runtime and capacity |
| Attack protocol | Article discusses robustness; exact matched protocol requires the article tables | Unknown until source supplied | Standardized JPEG/noise/blur/resize/rotation/crop/combined attacks |
| Original contribution | Catalan transform + NMF + ANN framework | Existing DWT–SVD–CNN baseline to reproduce | Adaptive embedding + attack-aware training + capacity/ECC study, contingent on experimental evidence |

## Boundary of claims

- The CT–NMF–ANN pipeline is the **base-paper reference**, not an implementation template for the DWT–SVD system.
- The DWT–SVD–CNN pipeline is the **baseline reproduction target**, not an original contribution.
- The proposed system must not claim that any individual established component—DWT, SVD, CNN, ECC or common attacks—is novel by itself.
- No paper-reported number will be called a direct comparison unless experimental settings are matched and documented.

## Required source recovery

Before Phase 8, add a citation record for *Transform and Extract – Invisible Deep Watermarking System* with authors, venue, year, DOI/URL and accessible full text. If it is an unpublished internal project rather than a paper, label it as such and define a reproducible baseline from its available code/specification instead of calling it a literature reproduction.
