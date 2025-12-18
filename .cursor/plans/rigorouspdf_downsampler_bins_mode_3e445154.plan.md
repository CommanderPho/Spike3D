---
name: rigorouspdf_downsampler_bins_mode
overview: Extend `RigorousPDFDownsampler` so it accepts bin centers/edges via a `bins` argument, infers per-axis `bin_sizes`, and supports `None` entries that default to index-based bins.
todos:
  - id: update-fields
    content: Update `RigorousPDFDownsampler` fields to make `bin_sizes` optional and add an optional `bins` field, keeping the init signature effectively supporting both modes.
    status: pending
  - id: derive-bin-sizes-from-bins
    content: Extend `__attrs_post_init__` to derive `bin_sizes` from a provided `bins` sequence, handling `None` entries as index-based bins and validating shapes and monotonicity.
    status: pending
    dependencies:
      - update-fields
  - id: validate-and-normalize
    content: Retain and adapt validation and normalization logic so total mass checks use the inferred `bin_sizes` and guard against inconsistent `bins`/`bin_sizes` inputs.
    status: pending
    dependencies:
      - derive-bin-sizes-from-bins
  - id: tests-and-docs
    content: Add or update tests in `test_prob_downsampler.py` and tweak `RigorousPDFDownsampler` docstrings to cover the new `bins`-based mode and example usage.
    status: pending
    dependencies:
      - validate-and-normalize
---

## Extend `RigorousPDFDownsampler` to support `bins`

### Goals

- **Allow construction with `bins`** so you can write `RigorousPDFDownsampler(fine_pdf=fine_pdf, bins=(x_centers, y_centers, None, t_centers))`.
- **Infer `bin_sizes` automatically** from the provided `bins`, falling back to index-based bins (step = 1) where an entry is `None`.
- **Preserve current behavior** for existing callers that pass `bin_sizes` directly.

### Key design decisions

- **Init signature** (conceptual, respecting your single-line preference):
- `def **init**(self, fine_pdf: np.ndarray, bin_sizes: Optional[ArrayLike] = None, bins: Optional[Sequence[Optional[ArrayLike]]] = None, ...) `(actual fields via `attrs.field`).
- **Mutual exclusivity**:
- Error if both `bin_sizes` and `bins` are provided and imply conflicting information.
- Error if neither `bin_sizes` nor `bins` is provided.
- **Bins semantics**:
- Each `bins[d] `is either a 1D array-like of centers (or edges) or `None`.
- For non-`None` bins, compute spacing as `np.median(np.diff(bins[d]))` and require it to be positive.
- For `None` bins, construct `np.arange(shape_f[d])` and use spacing 1.0.

### Implementation steps (code-focused)

- **Update `RigorousPDFDownsampler` fields** in [`NeuroPy/neuropy/utils/probability_downsampling.py`](NeuroPy/neuropy/utils/probability_downsampling.py):
- Add an optional `bins` field (stored for potential debugging/inspection, but not required for downsampling).
- Make `bin_sizes` optional at the field level (but enforce one of `bin_sizes` or `bins` is provided in `__attrs_post_init__`).
- **Extend `__attrs_post_init__` logic**:
- If `bins` is provided and `bin_sizes` is `None`:
    - Validate `len(bins) == ndim`.
    - For each axis `d`:
    - If `bins[d] is None`, build `axis_bins = np.arange(shape_f[d])`.
    - Else coerce `bins[d] `to `np.asarray` and validate it's 1D with length matching `shape_f[d]` (or compatible, depending on whether we treat them as centers).
    - Compute `spacing_d = np.median(np.diff(axis_bins))`; ensure `spacing_d > 0`.
    - Assemble `bin_sizes_arr` from the per-axis spacings.
- If `bin_sizes` is provided (with or without `bins`): keep the current `bin_sizes` validation and `_bin_sizes_arr` assignment, but optionally check consistency if `bins` is also passed.
- Keep the existing normalization check on the total mass.
- **Keep core downsampling unchanged**
- `downsample` and `_downsample_along_axis` already work with `self._bin_sizes_arr`; no changes needed beyond using the newly inferred values.

### Usage and tests

- **Intended usage example** (conceptual, no code changes yet):
- `downsampler = RigorousPDFDownsampler(fine_pdf=p_x_given_n, bins=(x_centers, y_centers, None, t_centers))`.
- Internally, this will infer `dx_f, dy_f, dv_f=1.0, dt_f` and populate `bin_sizes`.
- **Add or adapt tests** in [`NeuroPy/tests/test_prob_downsampler.py`](NeuroPy/tests/test_prob_downsampler.py):
- Test constructing with `bin_sizes` only (current behavior).
- Test constructing with `bins` only, with one or more `None` entries, and verify `bin_sizes` match the expected `median(diff(bins[d])) `(or 1.0 for `None`).
- Test error cases: mismatched `bins` length vs `ndim`, non-monotonic bins, both `bins` and `bin_sizes` with conflicting spacings.