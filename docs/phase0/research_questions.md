# Research Questions and Hypotheses — Phase 0

## Research questions

| ID | Question | Primary evidence |
|---|---|---|
| RQ1 | Can DWT–SVD preserve high visual fidelity while allowing blind recovery? | PSNR, SSIM, clean BER/accuracy |
| RQ2 | Does content-aware embedding improve the imperceptibility–robustness trade-off over fixed α? | Paired fixed/adaptive experiments, PSNR/SSIM/BER/NC |
| RQ3 | Does attack-aware training improve recovery under real-world distortions? | Clean, seen-attack and unseen-attack test sets |
| RQ4 | How does payload size affect quality and recovery? | Capacity sweep: 8–1024 attempted bits; effective payload reported |
| RQ5 | Does optional ECC improve final recovery enough to justify its overhead? | Pre/post-ECC BER, effective payload, decode failures |
| RQ6 | Which wavelet, subband and singular-value rule offer the best trade-off? | Controlled ablation with common splits/attacks |
| RQ7 | How does the final system compare to its fixed DWT–SVD baseline? | Same dataset, seed, payload and attack protocol |
| RQ8 | How does it compare to the CT–NMF base paper? | Conceptual comparison; direct metrics only if protocols match |
| RQ9 | Which additions cause measured changes? | Full-model ablation |
| RQ10 | Does performance generalize to held-out images and attack configurations? | Locked test split and unseen attacks |

## Falsifiable hypotheses

**H1 (baseline feasibility).** At one or more documented α/payload settings, the baseline will achieve acceptable visual fidelity and above-chance blind bit recovery on held-out clean images. “Acceptable” will be pre-registered in Phase 6/7 from the dataset and payload setting; it will not be retrospectively chosen from test results.

**H2 (adaptivity).** With the same average embedding budget, content-aware α will improve at least one quality/robustness Pareto frontier point relative to fixed α. It may be rejected if benefits are absent or the map causes instability.

**H3 (attack awareness).** A decoder trained on a declared distribution of attacks will reduce BER on those attacks relative to clean-only training. Evaluation on unseen attacks is separate; no gain is assumed.

**H4 (capacity).** Increasing payload will eventually worsen at least one of PSNR, SSIM, BER or runtime. The objective is to locate a reliable operating range, not claim that all sizes are feasible.

**H5 (ECC).** ECC will improve message-level recovery only when the raw error pattern lies within its correction capability; overhead can make it inferior at a fixed channel-use budget.

## Experimental discipline

- Split host images before model development; do not tune on the test set.
- Report mean, dispersion and per-attack severity; retain raw predictions for BER calculations.
- Use paired host images and fixed seeds for baseline-versus-variant comparisons.
- Predefine attack parameters and thresholding in configuration files.
- State whether any conclusion is direct, reported-only or conceptual.
