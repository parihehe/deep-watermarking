# Research Gap — Phase 0

## What we are building

This project will investigate a **blind, invisible image watermarking system**. It will embed a binary payload into a host image so that the image remains visually close to the original, then recover the payload from only the watermarked (possibly attacked) image. The research question is not whether a watermark can be embedded, but how to balance four competing objectives:

1. imperceptibility (PSNR and SSIM),
2. recovery reliability (BER, bit accuracy and NC),
3. robustness to realistic image transformations, and
4. usable payload capacity.

The planned baseline is a single-level Haar DWT on the LL subband, SVD-based coefficient modification and a 1D-CNN blind decoder. Proposed work will add content-aware embedding, attack-aware decoder training, payload studies and an optional error-correction layer. These are hypotheses to test, not claimed improvements.

## Evidence reviewed

### Base paper (verified)

Saikat et al., *Deep Learning-Based Image Watermarking Using Catalan Transform and Non-Negative Matrix Factorization*, IEEE Access 13 (2025), pp. 68995–69020, DOI: [10.1109/ACCESS.2025.3558121](https://doi.org/10.1109/ACCESS.2025.3558121).

From the article’s open-access record and text, the method operates in YIQ; selects the I component; splits it into 2×2 blocks; applies Catalan transform, keyed shuffling, NMF and a coefficient-difference embedding rule; then inverts the transforms. Its extraction creates features from watermarked blocks and uses an ANN. The watermark is also scrambled with a stated security mechanism. This is a distinct transform/NMF/ANN approach, not a DWT–SVD method.

### Existing DWT–SVD foundation (unverified source document)

The supplied brief names *Transform and Extract – Invisible Deep Watermarking System* and specifies its intended baseline: host image → DWT → LL → SVD → singular-value modification → IDWT; blind decoding through singular-value windows and a 1D CNN. It also states DIV2K and UUID + SHA-256 binary watermarks.

An exact public record or full text for that title was not located during Phase 0. Therefore, these details are recorded as **requirements supplied by the project brief**, not independently verified claims about a publication. Its DOI/authors/year/full paper or repository must be obtained before Phase 8 reproduction and numerical comparison.

## Gap

The base paper is strong evidence for a learned transform-domain watermarking line, but it does not establish the requested DWT–SVD blind-decoding baseline. The supplied DWT–SVD specification, meanwhile, uses a fixed embedding strength and has no verified experimental protocol available yet. The research gap to test is:

> Can a reproducible, fixed-strength DWT–SVD–CNN baseline be extended with content-aware strength allocation and attack-aware training to improve the quality–robustness–capacity trade-off, while preserving blind extraction?

This is deliberately narrower than “DWT + SVD + CNN is novel.” DWT/SVD watermarking and learned extraction are established technique families. Novelty, if supported, must arise from the controlled integration and ablation of adaptive embedding, attack-aware training, capacity management and optional ECC.

## How the gap will be tested

1. Freeze a documented fixed-strength DWT–SVD–CNN baseline.
2. Evaluate it on held-out DIV2K images, clean and under standardized attacks.
3. Change one contribution at a time: adaptive embedding, attack-aware training, then ECC.
4. Measure PSNR, SSIM, BER, bit accuracy, NC, effective payload and runtime with fixed data splits and seeds.
5. Run ablations and report negative results as well as gains.

## Constraints and risks

- Blind extraction means the original host image cannot be used at inference; any design needing it is not an eligible final decoder.
- Direct numerical comparison with either paper is valid only after matching data, preprocessing, payload, attack protocol and metric definitions.
- ECC can lower post-correction BER but reduces effective payload. It must be evaluated separately.
- The supplied foundation paper’s unavailable provenance is unresolved and blocks a faithful Phase 8 reproduction claim.

## Sources

- Saikat et al. publication record: [DBLP](https://dblp.org/rec/journals/access/SaikatPDDS25.html).
- Open-access article text/abstract: [ResearchGate record](https://www.researchgate.net/publication/390510688_Deep_Learning-Based_Image_Watermarking_using_Catalan_Transform_and_Non-Negative_Matrix_Factorization).
- DIV2K provenance and split: [official DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [NTIRE 2017 dataset paper](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Agustsson_NTIRE_2017_Challenge_CVPR_2017_paper.pdf).
