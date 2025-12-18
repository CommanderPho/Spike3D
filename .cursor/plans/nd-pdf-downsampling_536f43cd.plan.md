---
name: nd-pdf-downsampling
overview: Generalize the existing approximate PDF downsampler to support N-dimensional arrays with user-specified downsample axes while keeping non-selected dimensions independent and preserving total probability mass.
todos:
  - id: analyze-current-implementation
    content: Review current 2D/3D `approx_downsample_pdf` behavior, including padding strategy, normalization, and bin-center downsampling, to ensure N-D generalization preserves existing semantics.
    status: pending
  - id: design-nd-api
    content: Specify extended `approx_downsample_pdf` signature and parameter semantics for `axis` and vector-valued `downsample_factor`, ensuring backward compatibility.
    status: pending
    dependencies:
      - analyze-current-implementation
  - id: implement-nd-downsampling
    content: Implement the N-D block-sum downsampling and per-slice normalization over user-specified axes using vectorized NumPy operations.
    status: pending
    dependencies:
      - design-nd-api
  - id: update-bin-center-logic
    content: Generalize or constrain bin-center downsampling logic to work consistently with the new axis-based API while preserving existing use cases.
    status: pending
    dependencies:
      - implement-nd-downsampling
  - id: add-tests
    content: Add unit tests covering N-D use cases, axis variations, probability conservation, and regression cases for existing behavior.
    status: pending
    dependencies:
      - implement-nd-downsampling
      - update-bin-center-logic
  - id: update-docs
    content: Update the `approx_downsample_pdf` docstring and any related documentation to describe the N-D, axis-based behavior and limitations clearly.
    status: pending
    dependencies:
      - add-tests
---

# Generalize approximate PDF downsampling to N dimensions

### Scope

- **Target code**: Generalize `approx_downsample_pdf` in [`NeuroPy/neuropy/utils/probability_downsampling.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\probability_downsampling.py) so it can downsample along an arbitrary subset of axes (e.g. `axis=(0, 1)`, `axis=(0, -1)`), leaving all other axes unchanged.
- **Behavioral goal**: For input of shape `(n_x, n_y, n_t, ...)` and axes `(0, 1)` with downsample factor `(a, b)`, return an array of shape `(n_x/a, n_y/b, n_t, ...)` (with appropriate padding for non-divisible sizes), preserving probability mass per independent slice along non-downsampled dimensions.

### API adjustments

- **Extend function signature** (backwards compatible) to accept:
- `axis`: tuple of ints selecting the dimensions to be spatially downsampled; default `(0, 1)` to preserve existing behavior.
- `downsample_factor`: either a scalar `int` (applied to all selected axes) or a sequence of `int` of the same length as `axis`.
- **Backwards compatibility**:
- Preserve current semantics when `axis` is left as default `(0, 1)` and `downsample_factor` is a scalar, including handling of 2D vs 3D inputs and the `xbin_centers`/`ybin_centers` behavior.
- Keep `xbin_centers` and `ybin_centers` support limited to the first two logical axes (matching current use) and document that they are only applied when `axis` selects those dimensions.

### Core N-D downsampling algorithm

- **Preprocessing**:
- Convert `a_p_x_given_n` to `np.asarray(a)` and, for 2D inputs, temporarily add a singleton last dimension (as currently done) so that the implementation works uniformly for 2D/3D/ND.
- Normalize `axis` to a tuple of non-negative axis indices relative to the current array `a` using `a.ndim`.
- Normalize `downsample_factor` to a tuple `k_per_axis` of ints matching `axis` length.
- **Per-axis block-sum downsampling (vectorized)**:
- For each `(ax, k)` in `zip(axis, k_per_axis)`:
    - Compute the size `n = a.shape[ax] `and required padding `pad = (k - (n % k)) % k`.
    - Pad only along that axis with `mode='edge'` (matching the current 2D/3D behavior).
    - Use `np.moveaxis` to bring the target axis to the front, reshape to `((n+pad)//k, k, ...)`, and `sum` over the block axis to effect block-summing along that dimension.
    - Move the axis back to its original position; repeat for the next axis.
- **Normalization per independent slice**:
- After all selected axes are block-summed, compute `mass = a_blocksum.sum(axis=axis, keepdims=True)` using the original (normalized) `axis` tuple (converted to positive indices).
- Form `mass_safe = np.where(mass == 0, 1.0, mass)` and set `a_small = a_blocksum / mass_safe` so each slice over non-downsampled dimensions remains a normalized PDF.
- For originally 2D inputs, squeeze out the temporary singleton dimension as currently done.

### Bin-center handling

- **Maintain existing behavior** for `xbin_centers` and `ybin_centers`:
- When `axis` selects the first and second logical dimensions (e.g. `(0, 1)` or equivalent with negatives), compute `pad_h` and `pad_w` from the corresponding `k_per_axis` entries and reuse the current logic: pad with `mode='edge'`, then reshape into blocks of size `k` and take the mean along that block axis.
- If `axis` does not include the dimensions associated with `xbin_centers` / `ybin_centers`, either leave those unchanged or return them as-is, and clearly document the limitation in the docstring.

### Testing and validation

- **Unit tests** (add to or extend [`NeuroPy/tests/test_prob_downsampler.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\tests\test_prob_downsampler.py) or a new test module for `approx_downsample_pdf`):
- Tests for N-D shapes (e.g. `(n_x, n_y, n_t)`, `(n_x, n_y, n_t, n_cells)`):
    - Verify output shapes match expectations for various `axis` and `downsample_factor` combinations, including negative axis indices.
    - Verify probability mass conservation per independent slice (e.g. for each `t` and/or additional dims, sums over downsampled axes are ~1).
- Regression tests for existing 2D and 3D behavior with default `axis=(0, 1)` and scalar `downsample_factor` to ensure results match (within numerical tolerance) the current implementation.
- Tests for bin center downsampling when `axis=(0, 1)` to ensure existing behavior is preserved.
- **Performance checks**:
- Optionally add a simple timing test or benchmark snippet (not as a strict unit test) to confirm the new implementation remains vectorized and efficient for typical posterior sizes.

### Documentation