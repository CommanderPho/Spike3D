---
name: parallelize_posterior_peak_prominence
overview: Introduce optional process-based parallelization to PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation with a clean fallback to the existing serial behavior.
todos:
  - id: review-implementation
    content: Review the current `_perform_find_posterior_peaks_peak_prominence2d_computation` implementation and identify the per-slab inner loop to refactor.
    status: pending
  - id: add-slab-worker
    content: Implement `_compute_single_posterior_slab` helper that encapsulates the existing per-slab logic and returns (epoch_idx, t_idx, posterior_peaks_df, slab_result_dict).
    status: pending
    dependencies:
      - review-implementation
  - id: extend-api-flags
    content: Add `parallel` and `max_workers` parameters to `_perform_find_posterior_peaks_peak_prominence2d_computation` and update its docstring accordingly.
    status: pending
    dependencies:
      - add-slab-worker
  - id: implement-serial-path
    content: Wire the serial (fallback) path to call the new helper inside the nested loops, preserving existing behavior and ordering when `parallel=False`.
    status: pending
    dependencies:
      - extend-api-flags
  - id: implement-parallel-path
    content: Implement the parallel path using `ProcessPoolExecutor`, gather and sort results by (epoch_idx, t_idx), and populate outputs in deterministic order when `parallel=True`.
    status: pending
    dependencies:
      - implement-serial-path
  - id: reuse-postprocessing
    content: Reuse the existing empty/non-empty post-processing logic (concatenation, binning, filtering, blurring, result wrapping) unchanged after collecting per-slab results.
    status: pending
    dependencies:
      - implement-parallel-path
---

# Parallelize posterior peak prominence computation with optional fallback

### Goal

Add **simple, optional process-based parallelization** to `PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation` in `pyphoplacecellanalysis/External/peak_prominence2d.py`, while preserving identical numerical results and allowing a full fallback to the current serial implementation when parallelization is disabled.

### Steps

- **Review current implementation**
- Re-skim `PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation` to confirm the exact inner-loop logic over `(epoch_idx, t_idx)` slabs and the shape/contents of `out_results` and the per-slab `posterior_peaks_df`.
- Identify precisely which code is per-slab (embarrassingly parallel) and which code must remain global (concatenation, binning, filtering, and blurring).
- **Extract a per-slab worker helper**
- Implement a new internal helper (e.g. `_compute_single_posterior_slab(...)`) that:
    - Accepts `epoch_idx`, `t_idx`, the 2D `slab`, `xbin_centers`, `ybin_centers`, `step`, `min_considered_promenence`, `peak_height_multiplier_probe_levels`, and `debug_print`.
    - Calls `compute_prominence_contours` exactly as in the current inner loop and builds `peaks_dict`, `id_map`, `prominence_map`, and `parent_map`.
    - Reuses the existing per-peak array logic to construct the per-slab `posterior_peaks_df` in exactly the same way as now.
    - Returns a tuple `(epoch_idx, t_idx, posterior_peaks_df, slab_result_dict)` where `slab_result_dict` matches the current `out_results[(epoch_idx, t_idx)]` value.
- **Extend the main API with an optional parallel flag**
- Update the signature of `_perform_find_posterior_peaks_peak_prominence2d_computation` to include optional arguments (kept on a single line):
    - `parallel: bool = True`
    - `max_workers: Optional[int] = None`
- Document in the docstring that:
    - When `parallel=False`, behavior is identical to the original serial implementation.
    - When `parallel=True`, slabs are processed in parallel via a process pool, but results are returned in the same order.
- **Implement serial path (full fallback)**
- Retain the existing outer loops over epochs and time bins.
- When `parallel=False`, instead of using a pool, call `_compute_single_posterior_slab(...)` directly inside the `for epoch_idx`/`for t_idx` loops.
- Populate `out_results` and `out_posteriors_peak_dfs_list` exactly as before, ensuring row ordering is unchanged.
- **Implement parallel path using `ProcessPoolExecutor`**
- When `parallel=True` and there is more than one slab and `os.cpu_count() > 1`, build a list of tasks `(epoch_idx, t_idx, slab)`.
- Use `concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)` to submit `_compute_single_posterior_slab(...)` for each task.
- Collect all results, then **sort the list of results by `(epoch_idx, t_idx)`** to enforce deterministic ordering.
- Populate `out_results[(epoch_idx, t_idx)] `and `out_posteriors_peak_dfs_list` from the sorted results.
- If there is only a single slab or only one CPU, skip the pool and fall back to the serial path for simplicity.
- **Reuse existing global post-processing logic**
- Keep the "no peaks anywhere" empty case identical: create an empty DataFrame and zero-filled counts arrays, blur them, and return a `PosteriorPeaksPeakProminence2dResult` as now.
- For the non-empty case, reuse the existing logic to:
    - Concatenate `out_posteriors_peak_dfs_list` with `ignore_index=True`.
    - Call `build_df_discretized_binned_position_columns` to add binned peak position columns.
    - Call `_build_filtered_summits_analysis_results` to compute filtered peaks and `pf_peak_counts_map`.
    - Blur counts with `uniform_filter` and `gaussian_filter`, wrap in `PeakCounts`, and return `PosteriorPeaksPeakProminence2dResult`.
- **Keep edits minimal and style-compliant**
- Preserve the existing calculation code as much as possible; mainly refactor the inner loop into the helper and introduce the parallel/serial branching.