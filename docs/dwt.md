# Discrete Wavelet Transform — Phase 4

## Intuition

The 2D DWT separates an image into a coarse approximation and directional details. One level creates four coefficient arrays:

- `LL`: low-pass filtering horizontally and vertically; the coarse image content.
- `LH`: horizontal-detail output in the PyWavelets convention.
- `HL`: vertical-detail output in the PyWavelets convention.
- `HH`: diagonal/high-frequency detail.

This is useful for watermarking because it lets later phases compare where a small signal change is least visible and most robust. Phase 4 only implements the transform; it does not select a subband or embed anything.

## Formulation

For image \(I\), analysis filters \(h\) (low-pass) and \(g\) (high-pass), and downsampling \(\downarrow 2\):

\[
LL = \downarrow 2_h\downarrow 2_v(I * h_h * h_v),\quad
LH = \downarrow 2_h\downarrow 2_v(I * h_h * g_v),
\]
\[
HL = \downarrow 2_h\downarrow 2_v(I * g_h * h_v),\quad
HH = \downarrow 2_h\downarrow 2_v(I * g_h * g_v).
\]

The inverse transform applies the paired synthesis filters and sums the upsampled subbands. Boundary extension can add a pixel or more at the reconstruction edge, especially for odd dimensions and longer filters. `reconstruct_2d()` therefore crops to the source shape stored with the coefficients.

## Implemented API

```python
coefficients = decompose_2d(image, wavelet="haar")
recovered = reconstruct_2d(coefficients)
error = reconstruction_error(image, wavelet="db4")
```

Inputs must be finite, numeric, two-dimensional arrays at least 2×2. The Phase 4 controlled comparison covers `haar`, `db2`, and `db4`, all at one level with symmetric boundary extension. Color-channel handling and luminance selection are postponed to baseline embedding design.

## Complexity and limitation

For fixed-length wavelet filters, one-level DWT and IDWT are \(O(HW)\) in time and memory proportional to the image size. DWT does not itself make a method blind, robust, or imperceptible; those properties must be demonstrated after embedding and extraction are defined.
