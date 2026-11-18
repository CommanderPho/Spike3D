---
name: Size table n_total
overview: Extend DataFrameFilter.filtered_size_info_df with an unfiltered row-count column looked up from original_df_dict via the existing filtered_ name mapping.
todos:
  - id: extend-size-df
    content: Add n_total column to filtered_size_info_df from original_df_dict via removeprefix('filtered_')
    status: completed
isProject: false
---

# Add unfiltered row count to size table

## Change

Update [`filtered_size_info_df`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) (~2857–2860) so each filtered DF row also reports the matching original (unfiltered) length.

Current:

```python
n_records_tuples = [(name, len(df)) for name, df in self.filtered_df_dict.items() if (df is not None)]
return pd.DataFrame(n_records_tuples, columns=['df_name', 'n_elements'])
```

New columns: `df_name`, `n_elements` (filtered), `n_total` (unfiltered).

Name mapping already used elsewhere in this class (`removeprefix('filtered_')`, see ~3734–3735):

```python
original_name = name.removeprefix('filtered_')
n_total = len(self.original_df_dict.get(original_name, df))
```

`table_widget` is already bound to this property (`~3159` init, `~3959` on filter update), so no widget wiring changes are needed.

## Out of scope

- No notebook edits
- No changes to filtering logic