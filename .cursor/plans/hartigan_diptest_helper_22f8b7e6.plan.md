---
name: Hartigan diptest helper
overview: Add a `PrePostDeltaDistributionComparison.compute_hartigan_dip_test` classmethod that wraps the `diptest` package for Hartigan’s unimodality test on raw P_Short observations (1D array or DataFrame grouped by condition).
todos:
  - id: add-classmethod
    content: Add compute_hartigan_dip_test after compute_mode_ratio with array/DataFrame inputs, diptest call, and iid caveat docstring
    status: completed
  - id: verify-import
    content: Confirm diptest import works in the Spike3D env (already declared in Spike3D pyproject.toml)
    status: completed
isProject: false
---

# Add Hartigan dip-test helper

## Placement

Add a new `@classmethod` on [`PrePostDeltaDistributionComparison`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) immediately after `compute_mode_ratio` (~line 3602), before `compute_ordinal_mode_shift`. Tag it like the other fig4 stats helpers (`stats`, `fig4`, `P_Short`, `dip`, `unimodality`).

## API (chosen default)

```python
@classmethod
def compute_hartigan_dip_test(cls, values, value_col='P_Short', condition_col='pre_post_delta_category',
                              alpha: float = 0.05, boot_pval: bool = False, n_boot: int = 10000) -> Union[Dict[str, Any], pd.DataFrame]:
```

- **1D array / Series**: finite observations only → return `dict` with `n`, `dip`, `p_value`, `reject_unimodality` (`p_value < alpha`), plus echo of `boot_pval` / `n_boot` / `alpha`.
- **DataFrame**: group by `condition_col` (same pattern as `compute_mode_ratio`) → one result row per condition in a `DataFrame`.
- Lazy-import `from diptest import diptest` inside the method (same style as `OrderedModel` in `compute_ordinal_mode_shift`).
- Pass observations to `diptest(x)` or `diptest(x, boot_pval=..., n_boot=...)`; never histogram counts/KDE densities.
- Docstring: H0 = unimodality; small p rejects unimodality but does not prove exactly two modes; ordinary p-values assume i.i.d. samples — do not treat temporally correlated time bins / repeated measures within animals as independent.

## Dependency

[`Spike3D/pyproject.toml`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\pyproject.toml) already has `diptest>=0.12.0`. No dependency change unless import fails in the working env (then `uv add diptest` in Spike3D).

## Out of scope

No four-panel laps/PBE wrapper, plot wiring, notebook edits, or tests unless requested later.
