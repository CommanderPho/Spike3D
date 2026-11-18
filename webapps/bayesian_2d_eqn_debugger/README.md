# Bayesian 2D Equation Debugger (web)

Static browser port of `InteractiveBayesian2DEquationDebugger`.

Uses **classic scripts** (not ES modules) because `python -m http.server` on Windows often serves `.js` as `text/plain`, which browsers reject for `type="module"`.

## Export from Python

```python
from pathlib import Path
from pyphoplacecellanalysis.Analysis.Decoder.eqn_debugger_export import export_bayesian_2d_eqn_debugger

export_bayesian_2d_eqn_debugger(
    a_dst_decoder2D,
    out_path=Path(r".../webapps/bayesian_2d_eqn_debugger/data/bayesian_2d_eqn_debugger.zarr"),
    group_key="JS15_cells_27_29_31",
    neuron_ids=(27, 29, 31),
)
```

This writes:

- Zarr store under `data/*.zarr` (archive)
- `data/groups.json` — group catalog for the UI
- `data/<group_key>.json` — payload the browser loads

## Serve and open

```bash
cd webapps/bayesian_2d_eqn_debugger
python -m http.server 8000
```

Open http://localhost:8000/ and hard-refresh (Ctrl+F5).
