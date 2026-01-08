---
name: Add area to posterior peaks DataFrame
overview: Add contour area calculation and propagation through the posterior peaks computation pipeline, updating both the main computation function and the compatibility reconstruction function, and improving the scoring function to use actual area instead of bounding box area.
todos:
  - id: "1"
    content: Add summit_slice_area_arr array initialization in _compute_single_posterior_slab (after line 934)
    status: completed
  - id: "2"
    content: Extract area from level_slices and populate summit_slice_area_arr in _compute_single_posterior_slab loop (around lines 980-987 and 978-979)
    status: completed
    dependencies:
      - "1"
  - id: "3"
    content: Add 'summit_slice_area' column to posterior_peaks_df in _compute_single_posterior_slab (around line 1009)
    status: completed
    dependencies:
      - "2"
  - id: "4"
    content: Add area calculation to level_slices in _reconstruct_posterior_peaks_from_efficient_computation (around line 1833-1839)
    status: completed
  - id: "5"
    content: Add summit_slice_area_arr array initialization in _reconstruct_posterior_peaks_from_efficient_computation (after line 1796)
    status: completed
    dependencies:
      - "4"
  - id: "6"
    content: Extract area from level_slices and populate summit_slice_area_arr in _reconstruct_posterior_peaks_from_efficient_computation loop (around lines 1863-1870 and 1854-1862)
    status: completed
    dependencies:
      - "5"
  - id: "7"
    content: Add 'summit_slice_area' column to posterior_peaks_df in _reconstruct_posterior_peaks_from_efficient_computation (around line 1893)
    status: completed
    dependencies:
      - "6"
  - id: "8"
    content: Update score_slab_quality to use actual area instead of bounding box area (around line 2133)
    status: completed
    dependencies:
      - "3"
      - "7"
---

