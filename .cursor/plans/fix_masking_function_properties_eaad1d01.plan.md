---
name: Fix Masking Function Properties
overview: Review of `mask_computed_DecodedFilterEpochsResult_by_time_bin_inclusion_masks` found that all required `DecodedFilterEpochsResult` properties ARE being updated correctly, but there are two bugs that need fixing.
todos:
  - id: fix-break-to-continue
    content: Change break to continue on line 1707 to skip problematic epochs instead of stopping
    status: completed
  - id: fix-inactive-mask-append
    content: Change line 1826 to append inactive_mask instead of last_valid_indices
    status: completed
---

# Fix Masking Function Property Updates

## Analysis Summary

The `DecodedFilterEpochsResult` class has these per-epoch list properties that need to stay consistent:

| Property | Updated in 'dropped' mode | Updated in other modes |

|----------|--------------------------|------------------------|

| `p_x_given_n_list[i]` | Yes | Yes |

| `most_likely_position_indicies_list[i]` | Yes | Yes |

| `most_likely_positions_list[i]` | Yes | Yes |

| `marginal_x_list[i]` | Yes | Yes |

| `marginal_y_list[i]` | Yes | Yes |

| `marginal_z_list[i]` | Yes | Yes |

| `time_bin_containers[i]` | Yes | No (correct - bins unchanged) |

| `time_bin_edges[i]` | Yes | No (correct - bins unchanged) |

| `nbins[i]` | Yes | No (correct - count unchanged) |

| `spkcount[i]` | Yes (if not None) | No (correct - raw data) |

**Properties that correctly remain unchanged**: `decoding_time_bin_size`, `filter_epochs`, `num_filter_epochs`, `epoch_description_list`, `pos_bin_edges`

## Issues Found

### Issue 1: `break` should be `continue` (Line 1707)

```1702:1707:pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py
            if (len(a_time_bin_edges) != (num_time_bins+1)):
                #@IgnoreException
                print(f'WARN: Epoch[{i}]: len(a_time_bin_edges): {len(a_time_bin_edges)} != (num_time_bins+1): {(num_time_bins+1)}.') # continuing.
                # raise IndexError(f'len(a_time_bin_edges): {len(a_time_bin_edges)} != (num_time_bins+1): {(num_time_bins+1)}') #@IgnoreException
                # continue
                break
```

Using `break` stops processing ALL remaining epochs when one has a mismatch. Should use `continue` to skip just that problematic epoch.

### Issue 2: Wrong variable appended to `inactive_mask_list` (Line 1826)

```1825:1828:pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py
            _out_is_time_bin_active_list.append(is_time_bin_active) ## why are we changing this?
            inactive_mask_list.append(last_valid_indices)
            all_time_bin_indicies_list.append(all_time_bin_indicies)
            last_valid_indices_list.append(last_valid_indices)
```

Line 1826 appends `last_valid_indices` but should append `inactive_mask` based on:

- The variable naming convention
- The return tuple structure documented in the docstring (line 1656)

## Proposed Changes

1. **Line 1707**: Change `break` to `continue`
2. **Line 1826**: Change `inactive_mask_list.append(last_valid_indices)` to `inactive_mask_list.append(inactive_mask)`