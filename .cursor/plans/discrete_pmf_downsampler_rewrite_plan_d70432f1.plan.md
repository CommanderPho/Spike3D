---
name: Discrete PMF Downsampler Rewrite Plan
overview: ""
todos:
  - id: api-clarify
    content: Clarify and update `RigorousPDFDownsampler` docstring and API semantics for discrete PMFs with spatial and non-spatial axes.
    status: completed
  - id: impl-block-sum
    content: Implement fast, integer-factor block-sum downsampling kernel for specified spatial axes and integrate into `downsample`.
    status: completed
    dependencies:
      - api-clarify
  - id: impl-renorm-slices
    content: Implement per-slice normalization over spatial axes and remove density-style mass checks based on bin sizes.
    status: completed
    dependencies:
      - impl-block-sum
  - id: impl-bins-metadata
    content: Implement optional `bins` handling as metadata (block-averaged centers) without affecting probability mass calculations.
    status: completed
    dependencies:
      - impl-block-sum
  - id: tests-discrete-pmf
    content: Add tests for discrete per-time PMFs in `tests/test_prob_downsampler.py`, checking per-time normalization and correctness of block aggregation.
    status: completed
    dependencies:
      - impl-block-sum
      - impl-renorm-slices
      - impl-bins-metadata
---

# Discrete PMF Downsampler Rewrite Plan

## Overview

Rebuild `RigorousPDFDownsampler` to operate on large **discrete PMFs per time (or per non-spatial index)**, where bin geometry is incidental, focusing on **fast index-based grouping and summation** and enforcing **per-time spatial normalization**. Keep the external API reasonably compatible but simplify semantics around densities and bin sizes.

## Plan

### 1. Define the new semantics and API

- **Clarify core contract** in `neuropy/utils/probability_downsampling.py`:
- `fine_pdf` represents **discrete probability masses**, typically shaped like `(n_x, n_y, ..., n_t)` where some axes are spatial and others are non-spatial (e.g. time, conditions).
- `spatial_axes` explicitly marks the axes over which each distribution must sum to 1.
- For each fixed index of non-spatial axes, the class downscales along `spatial_axes` and ensures the resulting distribution over coarse spatial bins is normalized.
- **Simplify bin handling**:
- Treat `bins` as **optional metadata** used only to compute coarse bin centers (no effect on probability calculations).
- Ignore `bin_sizes` for normalization; geometry is not used to scale probabilities.
- Keep `bins` support lightweight and index-based when used: coarse centers computed via block-averaging or mid-point of covered fine centers.

### 2. Implement fast index-based downsampling for discrete PMFs

- In `RigorousPDFDownsampler` (same file):
- Replace the current `_downsample_along_axis` integration logic with a **discrete block-sum kernel**:
    - Require or at least optimize for **integer factors** along each spatial axis.
    - Implement downsampling via reshaping and summation when the dimension is divisible by the factor, e.g. reshape `(N,) -> (N_c, factor)` then sum over the factor axis.
    - For non-divisible cases, either: (a) drop or partially include the tail with correct renorm, or (b) pad and mask; keep behavior explicitly documented.
- Generalize this kernel to arbitrary axes by moving the target axis, applying the block-sum, and moving back, but avoid unnecessary copies for large arrays.
- Support multi-axis downsampling by applying the kernel sequentially to each spatial axis in `downsample` while keeping non-spatial axes untouched.

### 3. Enforce per-slice normalization over spatial axes

- In `__attrs_post_init__`:
- Remove / bypass any **density-style mass checks that multiply by bin sizes**.
- Add an optional **diagnostic check**:
    - If `spatial_axes` is provided, compute `mass = np.sum(fine_pdf, axis=spatial_axes)` and warn with max deviation from 1.
- In `downsample`:
- After all spatial downsampling steps are applied, compute `mass_coarse = np.sum(coarse_pdf, axis=spatial_axes)`.
- Vectorized renormalization:
    - Broadcast `mass_coarse` back to the full `coarse_pdf` shape and divide where `mass_coarse > 0`.
    - This ensures each non-spatial index (e.g., each time bin) has a spatial sum of 1, without touching their relative scaling across non-spatial axes.

### 4. Optional bin-center handling (geometry as metadata)

- If `bins` are provided for spatial axes:
- Precompute **fine bin centers** for each axis.
- When downsampling with integer factors, compute coarse centers via **block means of fine centers** for the indices that map to each coarse bin.
- If `bins` is `None` or not provided for an axis:
- Operate purely in index space; optionally expose trivial centers like `np.arange(n)` (or skip centers entirely if not needed).
- Ensure that any bin-related work is **O(N)** and avoids per-slice Python loops, staying vectorized across large arrays.

### 5. Optimize for large arrays and memory efficiency

- Avoid repeated `np.moveaxis`/`reshape` chains when possible by:
- Implementing a small helper that computes the reshaped view for a target axis once and reuses it.
- Using `reshape` + `sum` on contiguous blocks instead of `np.interp` or cumulative sums.
- Minimize temporary allocations:
- Work in-place where safe for renormalization (e.g., `coarse_pdf /= mass_broadcast` with `where` masks).
- Reuse buffers or intermediate shapes in multi-axis downsampling loops.
- Ensure the implementation handles shapes like `(62, 62, 2, 150_000+)` smoothly without Python-level loops over time.

### 6. Tests and validation

- In [`NeuroPy/tests/test_prob_downsampler.py`](NeuroPy/tests/test_prob_downsampler.py):
- Add tests for a synthetic **per-time PMF** with shape like `(n_x, n_y, n_cond, n_t)`:
    - Verify `np.allclose(np.sum(fine, axis=(0,1,2)), 1)` and that the same holds for `coarse` after downsampling.
    - Check that mass in each coarse spatial bin equals the sum of constituent fine bins (up to float tolerance).
- Add performance smoke tests on larger random arrays to ensure runtime is acceptable and no excessive memory blow-ups.

## Todos

- **api-clarify**: Clarify and update docstring and comments in `RigorousPDFDownsampler` to describe the discrete PMF semantics, `spatial_axes`, and non-spatial axes.
- **impl-block-sum**: Replace `_downsample_along_axis` with an efficient integer-factor block-sum implementation and wire it into `downsample`.