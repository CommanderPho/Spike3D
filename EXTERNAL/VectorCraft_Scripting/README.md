# VectorCraft Scripting Bridge

Python interface for scripting [VectorCraft](https://github.com/storytold/vectorcraft) via its headless Rust CLI (`vectorcraft-cli`). Useful for placing standardized figures (Plotly SVG, etc.) into `.vectorcraft` templates and exporting PDF/SVG/PNG.

## Prerequisites (Windows)

1. Install VectorCraft (includes `vectorcraft-cli.exe`). Default location:

   `C:\Program Files\VectorCraft\vectorcraft-cli.exe`

2. Optionally add that folder to `PATH`, or set:

   ```powershell
   $env:VECTORCRAFT_CLI = "C:\Program Files\VectorCraft\vectorcraft-cli.exe"
   ```

The Python bridge resolves the binary in this order: constructor argument → `VECTORCRAFT_CLI` → `PATH` → Program Files defaults.

## Install

From this folder (uses [uv](https://docs.astral.sh/uv/)):

```powershell
cd EXTERNAL/VectorCraft_Scripting
uv sync
```

Optional Plotly + Kaleido extras (for `export_plotly_figure`):

```powershell
uv sync --extra plotly
```

Or editable into another env:

```powershell
uv pip install -e EXTERNAL/VectorCraft_Scripting
# with plotly helpers:
uv pip install -e "EXTERNAL/VectorCraft_Scripting[plotly]"
```

## Quick start (Plotly → place → export)

```python
from pathlib import Path

import plotly.graph_objects as go
from vectorcraft_scripting import VectorCraftClient, export_plotly_figure

fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[2, 1, 3]))
fig.update_layout(width=560, height=360, margin=dict(l=40, r=16, t=32, b=32))

# 36 pt from the left, 480 pt down from the artboard top, 280×180 pt box
results = export_plotly_figure(
    fig,
    chart_svg=Path("chart.svg"),
    template=Path("template.vectorcraft"),
    rect=[36, 480, 280, 180],
    exports=[Path("out.vectorcraft"), Path("out.pdf")],
    client=VectorCraftClient(),  # optional; discovers CLI on Windows
)
```

Equivalent low-level CLI-style call:

```python
from pathlib import Path
import plotly.graph_objects as go
from vectorcraft_scripting import PlaceSpec, VectorCraftClient, place_into_template, write_plotly_svg

fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[2, 1, 3]))
chart = write_plotly_svg(fig, Path("chart.svg"))

place_into_template(
    Path("template.vectorcraft"),
    [PlaceSpec(path=chart, rect=[36, 480, 280, 180], crop="crop", link=False)],
    exports=[Path("out.vectorcraft"), Path("out.pdf")],
)
```

Which runs the same subprocess shape as:

```text
vectorcraft-cli run --in template.vectorcraft --cmd file.place --params "{...}" --export out.vectorcraft --export out.pdf
```

## Core API

```python
from vectorcraft_scripting import VectorCraftClient

vc = VectorCraftClient()  # or VectorCraftClient(r"C:\Program Files\VectorCraft\vectorcraft-cli.exe")

# Command catalogue (list of dicts with id, label, params docs, …)
catalogue = vc.commands()

# Headless batch
steps = vc.run(
    input="template.vectorcraft",
    commands=[
        ("file.place", {"path": str(Path("chart.svg").resolve()), "rect": [36, 480, 280, 180], "crop": "crop", "link": False}),
        "select.all",
    ],
    exports=["out.pdf"],
    scale=1.0,
)

# Convert / inspect
vc.convert("in.svg", "out.pdf")
summary = vc.info("art.vectorcraft")
```

`VectorCraftClient.run` returns one parsed JSON object per CLI step (`open` / `cmd` / `export`). Non-zero CLI exits raise `VectorCraftError`. Missing binary raises `CliNotFoundError`.

## Notes

- Paths passed to `--in`, `--export`, and `file.place` are made absolute so Windows working-directory quirks do not matter.
- Live GUI control (`vectorcraft --control PORT`) and MCP stdio are not wrapped in v1; use the CLI batch path above for automation.
