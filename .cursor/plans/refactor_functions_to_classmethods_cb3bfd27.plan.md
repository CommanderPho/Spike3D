---
name: Refactor functions to classmethods
overview: Move `_compute_single_posterior_slab` and `_compute_single_posterior_slab_efficient` from standalone functions into the `PeakPromenence` class as classmethods, and update all call sites accordingly.
todos:
  - id: move-functions
    content: Move _compute_single_posterior_slab and _compute_single_posterior_slab_efficient from module level into PeakPromenence class as classmethods
    status: completed
  - id: update-references
    content: Update internal references to use cls._find_contours_at_levels instead of PeakPromenence._find_contours_at_levels
    status: completed
    dependencies:
      - move-functions
  - id: update-call-sites
    content: Update call sites in _perform_find_posterior_peaks_peak_prominence2d_computation to use cls._compute_single_posterior_slab and cls._compute_single_posterior_slab_efficient
    status: completed
    dependencies:
      - move-functions
  - id: remove-old-functions
    content: Remove the original standalone function definitions from module level
    status: completed
    dependencies:
      - move-functions
      - update-call-sites
---

# Refactor `_compute_single_posterior_slab` and `_compute_single_posterior_slab_efficient` to PeakPromenence classmethods

## Overview

Convert two standalone functions (`_compute_single_posterior_slab` and `_compute_single_posterior_slab_efficient`) located at lines 906-1308 in `peak_prominence2d.py` into classmethods of the `PeakPromenence` class.

## Current State

- Both functions are currently module-level functions (lines 906-1308)
- They are called from `PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation` (lines 1760-1836)
- They reference `PeakPromenence._find_contours_at_levels` (already a classmethod)
- They use `compute_prominence_contours` (standalone function) and module-level imports (`ndimage`, `reconstruction`)

## Changes Required

### 1. Move functions into PeakPromenence class

- Remove functions from module level (lines 906-1308)
- Add them as classmethods inside `PeakPromenence` class (after line 1359, before `_build_filtered_summits_analysis_results`)
- Add `@classmethod` decorator
- Add `cls` as first parameter
- Update `PeakPromenence._find_contours_at_levels` calls to use `cls._find_contours_at_levels` for consistency

### 2. Update call sites

- In `_perform_find_posterior_peaks_peak_prominence2d_computation` (lines 1760-1762, 1816, 1830):
- Change `_compute_single_posterior_slab` → `cls._compute_single_posterior_slab`
- Change `_compute_single_posterior_slab_efficient` → `cls._compute_single_posterior_slab_efficient`

### 3. Preserve function signatures

- Keep all parameters unchanged (including defaults)
- Maintain `@function_attributes` decorators
- Keep function docstrings

## Files to Modify

- `pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/External/peak_prominence2d.py`

## Implementation Details

- Place the new classmethods after `_find_contours_at_levels` (line 1359) and before `_build_filtered_summits_analysis_results` (line 1360)
- Maintain two blank lines between class methods per user preference
- Keep function signatures on single line per user preference
- No changes needed to imports (all dependencies are already at module level)