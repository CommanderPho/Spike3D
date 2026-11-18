---
name: CSV export try/catch
overview: Add try/except CSV exports for each epoch-key’s `context_probability_df` and `context_probability_performance_df` inside `_run_all_compute_and_figures_for_all_epochs_all_maze_by_maze_context`, matching the existing WARN-on-failure save style already used in that function.
todos:
  - id: add-csv-exports
    content: Add try/except to_csv exports for both DFs per epoch key in the compute loop (lines 472-480)
    status: pending
isProject: false
---

# Add CSV export for context probability DataFrames

## Target

In [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) inside `_run_all_compute_and_figures_for_all_epochs_all_maze_by_maze_context`, immediately after each key’s dataframes are computed (lines 472–480).

## Approach

Reuse the same save pattern already in this function (pkl saves at ~413–417 and ~454–465):

- Write under `curr_active_pipeline.get_output_path()`
- Ensure parent exists (`mkdir(parents=True, exist_ok=True)` once before the loop, same as the pkl path)
- Wrap each `to_csv` in `try`/`except Exception`
- On success: print the path
- On failure: `print(f"[WARN] Failed to save ...: {e}")` so computation/plotting continues

Filenames (aligned with `{today_str}_decoded_results_{decoding_time_bin_size_ms}ms.pkl`):

- `{today_str}_context_probability_df_{k}_{decoding_time_bin_size_ms}ms.csv`
- `{today_str}_context_probability_performance_df_{k}_{decoding_time_bin_size_ms}ms.csv`

where `k` is `lap` / `replay` / `pbe`.

## Concrete edit

After assigning into the two dicts inside the existing `for k, a_decoded_result in decoded_results_dict.items():` loop, add:

```python
try:
    context_probability_csv_path: Path = curr_active_pipeline.get_output_path().joinpath(f'{today_str}_context_probability_df_{k}_{decoding_time_bin_size_ms}ms.csv')
    context_probability_df.to_csv(context_probability_csv_path)
    print(f'context_probability_csv_path: "{context_probability_csv_path.as_posix()}"')
except Exception as e:
    print(f"[WARN] Failed to save context_probability_df[{k}] to csv: {e}")

try:
    context_probability_performance_csv_path: Path = curr_active_pipeline.get_output_path().joinpath(f'{today_str}_context_probability_performance_df_{k}_{decoding_time_bin_size_ms}ms.csv')
    context_probability_performance_df.to_csv(context_probability_performance_csv_path)
    print(f'context_probability_performance_csv_path: "{context_probability_performance_csv_path.as_posix()}"')
except Exception as e:
    print(f"[WARN] Failed to save context_probability_performance_df[{k}] to csv: {e}")
```

Also close the edited for-loop with:

```python
## END for k, a_decoded_result in decoded_results_dict.items():...
```

`Path` is already imported at module top; `today_str` and `decoding_time_bin_size_ms` are already in scope. No new dependencies. `output_dict` population for the in-memory dicts stays as-is.