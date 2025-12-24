---
name: Add 3D marginal z support
overview: Extend `perform_build_marginals` to optionally return `curr_unit_marginal_z` for 3D posteriors, and update all call sites to handle the new return value.
todos:
  - id: modify_perform_build_marginals
    content: Modify perform_build_marginals to detect 3D decoders, compute z marginal, and return (x, y, z) tuple
    status: completed
  - id: update_call_site_1710
    content: Update call site at line 1710 to unpack 3 values instead of 2
    status: completed
    dependencies:
      - modify_perform_build_marginals
  - id: update_call_site_2262
    content: Update call site at line 2262 to unpack 3 values and include z in return
    status: completed
    dependencies:
      - modify_perform_build_marginals
  - id: update_call_site_2474
    content: Update call site at line 2474 to unpack 3 values and append z to marginal_z_list
    status: completed
    dependencies:
      - modify_perform_build_marginals
  - id: update_call_site_2999
    content: Update call site at line 2999 to unpack 3 values
    status: completed
    dependencies:
      - modify_perform_build_marginals
  - id: update_call_site_3444
    content: Update call site at line 3444 to handle z marginal via indexing
    status: completed
    dependencies:
      - modify_perform_build_marginals
---

# Extend `perform_build_marginals` to return optional `curr_unit_marginal_z` for 3D posteriors

## Overview

The function `perform_build_marginals` currently returns `(curr_unit_marginal_x, curr_unit_marginal_y)` for 1D/2D decoders. This plan extends it to also return `curr_unit_marginal_z` for 3D decoders, where `z` will be `None` for 1D/2D cases.

## Current Implementation Analysis

The function at [reconstruction.py:2579-2663](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) currently:

- Detects 1D decoders: `most_likely_positions.ndim < 2`
- For 1D: returns `(marginal_x, None)`
- For 2D: computes x and y marginals by summing over appropriate axes, returns `(marginal_x, marginal_y)`

For 3D decoders:

- `p_x_given_n` shape: `(x_bins, y_bins, z_bins, time_bins)`
- `most_likely_positions` shape: `(n_time_bins, 3)` with columns [x, y, z]
- Detection: `most_likely_positions.ndim == 2 and most_likely_positions.shape[1] == 3`

## Implementation Plan

### 1. Modify `perform_build_marginals` function

**Location**: [reconstruction.py:2579-2663](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py)**Changes**:

- Add detection for 3D decoders: check if `most_likely_positions.shape[1] == 3 `(when `ndim == 2`)
- For 3D decoders:
- Update x marginal: sum over y and z axes `np.sum(p_x_given_n, axis=(1, 2))`
- Update y marginal: sum over x and z axes `np.sum(p_x_given_n, axis=(0, 2))`
- Compute z marginal: sum over x and y axes `np.sum(p_x_given_n, axis=(0, 1))`
- Normalize z marginal and ensure 2D shape (same pattern as x and y)
- Extract z positions: `most_likely_positions[:, 2].T`
- For 1D/2D decoders: set `curr_unit_marginal_z = None`
- Change return statement to: `return curr_unit_marginal_x, curr_unit_marginal_y, curr_unit_marginal_z`
- Update docstring to document 3D case

### 2. Update all call sites

**Location 1**: [reconstruction.py:1710](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py)

- Already expects `curr_unit_marginal_z` (lines 1715-1716 check for it)
- Change: `curr_unit_marginal_x, curr_unit_marginal_y, curr_unit_marginal_z = ...`

**Location 2**: [reconstruction.py:2262](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py)

- Change: `curr_unit_marginal_x, curr_unit_marginal_y, curr_unit_marginal_z = ...`
- Update return statement at line 2263 to include `curr_unit_marginal_z`

**Location 3**: [reconstruction.py:2474](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py)

- Change: `curr_unit_marginal_x, curr_unit_marginal_y, curr_unit_marginal_z = ...`
- Add: `filter_epochs_decoder_result.marginal_z_list.append(curr_unit_marginal_z)` after line 2476

**Location 4**: [reconstruction.py:2999](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py)

- Change: `curr_unit_marginal_x, curr_unit_marginal_y, curr_unit_marginal_z = ...`
- Update line 3000 to include z in the container if needed (or leave as-is if z not used here)

**Location 5**: [reconstruction.py:3444](pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py)

- Change: `_revised_marginals = ...` (already uses tuple unpacking via indexing)
- Add handling for `_revised_marginals[2]` (z marginal) similar to lines 3448-3450

## Implementation Details

### 3D Marginal Computation

For a 3D posterior with shape `(x_bins, y_bins, z_bins, time_bins)`:

- **X marginal**: Sum over y and z → `np.sum(p_x_given_n, axis=(1, 2))` → shape `(x_bins, time_bins)`
- **Y marginal**: Sum over x and z → `np.sum(p_x_given_n, axis=(0, 2))` → shape `(y_bins, time_bins)`
- **Z marginal**: Sum over x and y → `np.sum(p_x_given_n, axis=(0, 1))` → shape `(z_bins, time_bins)`

Each marginal should be normalized and have its `most_likely_positions_1D` extracted from the corresponding column of `most_likely_positions`.

### Backward Compatibility