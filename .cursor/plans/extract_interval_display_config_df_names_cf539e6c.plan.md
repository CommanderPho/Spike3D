---
name: extract_interval_display_config_df names
overview: Align `extract_interval_display_config_df` with `EpochDisplayConfig.init_configs_list_from_interval_datasource_df`, so exported rows use each config’s `name` (from datasource `label` on the primary path) instead of synthetic `{datasource}[{i}]` keys.
todos:
  - id: update-df-extract
    content: In EpochRenderingMixin.extract_interval_display_config_df, use a_config.name for config_name (remove a_name/[i] branching).
    status: completed
isProject: false
---

# Update `extract_interval_display_config_df` for label-driven names

## Context

- [`init_configs_list_from_interval_datasource_df`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\mixins\epochs_plotting_mixins.py) now builds each `EpochDisplayConfig` with `name` taken from the dataframe **`label`** (primary path) or the datasource key (fallback in the `except` branch).
- [`extract_interval_display_config_df`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\Mixins\RenderTimeEpochs\EpochRenderingMixin.py) still overwrites the display identity with `f'{a_name}[{i}]'` when `num_configs > 1`, and uses `a_name` when `num_configs == 1` — bypassing `a_config.name`.

## Intended behavior

- Set each row’s **`name`** field to **`a_config.name`**, matching what the config objects already carry after `extract_interval_display_config_lists()`.
- Remove the `num_configs > 1` / `f'{a_name}[{i}]'` branch so multi-interval rows export real period labels, not positional indices.
- For **`num_configs == 1`**, use **`a_config.name`** as well (so a single-row datasource whose `label` differs from the datasource key is reflected correctly).

## Implementation (minimal)

In [`EpochRenderingMixin.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\Mixins\RenderTimeEpochs\EpochRenderingMixin.py) around lines 1213–1221:

- Replace the `if num_configs > 1: ... else: ...` block with a single assignment, e.g. `config_name = a_config.name` (or inline into `out_config_dict`).
- Keep the rest unchanged: `out_config_dict = {'name': config_name} | deepcopy(a_config.to_dict())`, plus `y0_location` / `y1_location` derivation.

## Optional hardening (only if you want extra safety)

- If you ever have duplicate `a_config.name` across rows in one datasource (the primary path currently asserts unique labels in the mixin), you could disambiguate e.g. `f"{a_config.name}[{i}]"` only when duplicates appear — **not required** if you keep the datasource invariant.

## Verification

- Grep shows **no other call sites** for `extract_interval_display_config_df` in the repo; risk is limited to notebooks/scripts calling it manually.
- Manually: multi-interval datasource with distinct `label` values → DataFrame `name` column matches those labels; single-interval → `name` matches row label / config name.
