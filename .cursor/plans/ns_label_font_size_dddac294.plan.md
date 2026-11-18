---
name: NS label font size
overview: Reduce the hard-coded matplotlib font size used for non-significant "N.S." labels in `add_significance_bars`, while leaving asterisk labels at the current size.
todos:
  - id: ns-fontsize
    content: In add_significance_bars, use smaller fontsize for 'N.S.' (default 10) via ns_fontsize param; keep asterisks at 16
    status: completed
isProject: false
---

# Smaller font for significance-bar "N.S."

## Source

[`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) — `add_significance_bars` (around line 676):

```python
_out_ann['label'] = ax.text(..., sig_astrisks_str, fontsize=16, ha='center')
```

Both `"N.S."` and `"*"/"**"/"***"` currently share `fontsize=16`. Multi-character `"N.S."` looks oversized next to the brackets (as in the PBE participation plot).

## Change

In `add_significance_bars` only:

- Keep asterisk labels at `fontsize=16`.
- Use `fontsize=10` when `sig_astrisks_str == 'N.S.'`.
- Add an optional `ns_fontsize: float = 10` parameter so callers can override without changing the asterisk size.

No call-site updates required; existing `add_significance_bars(...)` uses in Fig2 / AcrossSessionResults paths pick up the new default automatically after replot.