---
name: matplotlib fig serialization
overview: Add a standalone MATLAB-.fig-like save/load API in pyPhoCoreHelpers that pickles a live matplotlib Figure (plus optional caller plot_data) so a fresh notebook can restore an interactive figure.
todos:
  - id: add-module
    content: Create figure_serialization.py with MplFigArchive, save_mplfig, load_mplfig (dill payload, .mplfig)
    status: completed
  - id: add-tests
    content: Add test_figure_serialization.py round-trip + plot_data + overwrite guards (Agg)
    status: completed
isProject: false
---

# MATLAB-style matplotlib figure serialization (standalone)

## Goal
Provide `save_mplfig` / `load_mplfig` that pickle a live `matplotlib.figure.Figure` (and optional caller-supplied `plot_data`) so a fresh notebook can reopen an interactive figure — analogous to MATLAB `.fig`. No pipeline/`ExportHelpers` wiring in v1.

## Placement
New module: [`pyPhoCoreHelpers/src/pyphocorehelpers/plotting/figure_serialization.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoCoreHelpers/src/pyphocorehelpers/plotting/figure_serialization.py)

Leverage existing dill stack in [`pickling_helpers.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoCoreHelpers/src/pyphocorehelpers/Filesystem/pickling_helpers.py) (`import dill as pickle`). Match project conventions: `@function_attributes`, single-line signatures when ≤400 chars, docstring Usage blocks.

## File format (`.mplfig`)
Single dill pickle of a versioned payload dict (not a bare Figure), so optional data can ride along without a second file:

```python
{
  'format': 'pho_mplfig',
  'format_version': 1,
  'matplotlib_version': mpl.__version__,
  'created_at': iso_timestamp,
  'figure': fig,              # live matplotlib.figure.Figure
  'plot_data': plot_data,     # optional dict / RenderPlotsData-like mapping; default {}
  'metadata': metadata,       # optional plain dict (title, context str, notes, …)
}
```

Extension default: `.mplfig` (avoids clashing with MATLAB `.fig`). Paths without a suffix get `.mplfig` appended.

## Public API

```python
def save_mplfig(fig: Figure, path: Union[str, Path], plot_data: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None, overwrite: bool = True) -> Path:
    """Pickle fig (+ optional plot_data/metadata) to path.mplfig."""

def load_mplfig(path: Union[str, Path], *, attach_to_current_pyplot: bool = True) -> MplFigArchive:
    """Load archive; return container with .figure, .plot_data, .metadata."""
```

`MplFigArchive` — small `@define` / simple namespace container with fields `figure`, `plot_data`, `metadata`, `path`, plus convenience `.fig` alias.

**Load behavior:** after unpickle, if `attach_to_current_pyplot=True`, register the figure with the active pyplot figure manager (`plt.figure(fig.number)` / canvas manager pattern) so notebook display and `plt.show()` work. Document that custom artists / callbacks / GUI widgets may fail to unpickle.

Also accept `MatplotlibRenderPlots` / objects with `.fig` or `.figure` by extracting the first figure in `save_mplfig` (minimal convenience; no deep container round-trip).

## Out of scope (v1)
- No `ExportHelpers.write_to_file` / `output_figure` flag
- No auto-extraction of artist arrays / style rebuild path
- No PNG preview sidecar / ZIP hybrid
- No notebook edits

## Tests
Add [`pyPhoCoreHelpers/tests/test_figure_serialization.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoCoreHelpers/tests/test_figure_serialization.py):

- Round-trip a simple `plt.subplots` line plot with Agg backend: save → load → assert axes count, line y-data, title/suptitle preserved
- Round-trip with non-empty `plot_data` dict (e.g. small `np.ndarray`)
- Reject overwrite when `overwrite=False` and file exists

Use `matplotlib.use('Agg')` in the test module before importing pyplot.

## Notebook usage (docstring only)

```python
from pyphocorehelpers.plotting.figure_serialization import save_mplfig, load_mplfig

save_mplfig(fig, 'output/my_plot.mplfig', plot_data={'x': x, 'y': y}, metadata={'note': 'session foo'})
archive = load_mplfig('output/my_plot.mplfig')
fig2 = archive.figure  # interactive / displayable in a fresh notebook
```
