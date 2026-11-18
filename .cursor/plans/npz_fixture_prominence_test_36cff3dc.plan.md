---
name: NPZ fixture prominence test
overview: Add a test in test_peak_prominence2d.py that loads the real NPZ fixture from Spike3D, runs the prominence computation, and asserts the high-alpha mask shape matches the posterior array, skipping when the file is absent.
todos: []
isProject: false
---

# Add NPZ-based peak prominence test

## Context

- **Existing tests** in [tests/test_peak_prominence2d.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_peak_prominence2d.py) use in-memory 3D arrays and already cover return shape, caller dict-zip pattern, and memory options.
- **API**: `PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n, alpha=alpha_list)` expects **3D** `a_p_x_given_n` (n_x, n_y, n_t) and returns `(epoch_promenence_tuples, epoch_masks_list)`. Implementation in [peak_prominence2d.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\peak_prominence2d.py) (lines 2109–2110) raises `ValueError` if `a_p_x_given_n.ndim != 3`.
- **Fixture**: NPZ at `Spike3D/data/2026-03-18_peak_promenences_debugging.npz` with keys `a_p_x_given_n` and `alpha_list`. User snippet indicates shape can be 4D `(103948, 1, 41, 63)`; the API only accepts 3D, so the test must use a 3D slice when the array is 4D (e.g. first epoch).

## Implementation plan

### 1. Fixture path and skip logic

- Resolve NPZ path in order of preference:
  - Environment variable `PEAK_PROM_NPZ` (if set).
  - Else: sibling path from pyPhoPlaceCellAnalysis root: `root_project_folder.parent / "Spike3D" / "data" / "2026-03-18_peak_promenences_debugging.npz"` (works when repo is beside Spike3D).
- Decorate the test with `@unittest.skipUnless(npz_path.exists(), "NPZ fixture not found")` so the test is skipped when the file is absent (e.g. CI), and no hardcoded absolute path is required for portability.

### 2. New test method

Add a new test in `TestCompute2dDtPosteriorPeakPromenences` that:

1. **Resolve path**: Use the same `tests_folder` / `root_project_folder` as the rest of the file; compute `npz_path` as above (env or sibling).
2. **Load NPZ**: `container = np.load(npz_path.resolve())` (or `np.load(npz_path)`); get `a_p_x_given_n = container["a_p_x_given_n"]`, `alpha_list = container["alpha_list"].tolist()`.
3. **Ensure 3D**: If `a_p_x_given_n.ndim == 4`, use `a_p_x_given_n = a_p_x_given_n[0]` (first epoch) so shape becomes (1, 41, 63) or equivalent 3D. If already 3D, use as-is. This keeps the test compatible with the current 3D-only API and with a 4D saved array.
4. **Run prominence**:
  `epoch_promenence_tuples, epoch_masks_list = PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=alpha_list)`  
   (no `a_computation_measure_name` or prints in the test.)
5. **Caller pattern**:
  `epoch_masks_dict = dict(zip(alpha_list, epoch_masks_list))`  
   `a_high_alpha = alpha_list[-1]`  
   `an_alpha_epoch_masks = epoch_masks_dict[a_high_alpha]`
6. **Assert**:
  `self.assertEqual(np.shape(an_alpha_epoch_masks), np.shape(a_p_x_given_n))`  
   where `a_p_x_given_n` is the 3D array actually passed to the API.

Optional: add a brief assertion that `epoch_masks_list` length equals `len(alpha_list)` and that `an_alpha_epoch_masks.dtype == bool` for consistency with existing tests.

### 3. Naming and placement

- Name: e.g. `test_npz_fixture_prominence_and_shape` or `test_debug_npz_prominence_computation`.
- Place it after the existing “caller compat” tests and before the memory tests, or at the end of the class, following the file’s existing style (two blank lines between methods).

### 4. No changes elsewhere

- No changes to `peak_prominence2d.py` or to other tests; single new test method only.

## Summary


| Item         | Detail                                                                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| File to edit | [tests/test_peak_prominence2d.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_peak_prominence2d.py) |
| Fixture path | Env `PEAK_PROM_NPZ` or `root_project_folder.parent / "Spike3D" / "data" / "2026-03-18_peak_promenences_debugging.npz"`                     |
| Skip         | `@unittest.skipUnless(npz_path.exists(), "NPZ fixture not found")`                                                                         |
| 4D handling  | Use `a_p_x_given_n[0]` when `ndim == 4` so input to API is 3D                                                                              |
| Assertion    | `np.shape(an_alpha_epoch_masks) == np.shape(a_p_x_given_n)` (3D used in call)                                                              |


