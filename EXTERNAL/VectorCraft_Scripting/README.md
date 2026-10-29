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

## Live GUI control (`--control`)

Start VectorCraft with a control port (loopback only; default used by discovery is **7979**):

```powershell
& "C:\Program Files\VectorCraft\vectorcraft.exe" --control 7979
```

Optional: `$env:VECTORCRAFT_CONTROL_PORT = "7979"` (or `VECTORCRAFT_CONTROL_PORTS` comma-separated).

```python
from vectorcraft_scripting import ObjectQuery, connect_first, discover_instances

# Running instances that answer the control protocol
instances = discover_instances()
print([(i.port, i.pid) for i in instances])

live = connect_first()  # raises NoInstanceError if none
try:
    docs = live.list_documents()
    print([d.get("title") for d in docs])

    tree = live.inspect_document(depth=0, child_limit=20)
    paths = live.find_objects(ObjectQuery(kind="Path", min_width=100, max_width=400))
    print(paths)
finally:
    live.close()
```

Discovery order: explicit ports → `VECTORCRAFT_CONTROL_PORT(S)` → `7979` → Windows listeners owned by `vectorcraft.exe`, each verified with `ui.inspect`.

Live helpers:

| Method | Role |
|--------|------|
| `list_documents()` | Open tabs via `document.activate` + `file.info` |
| `assert_active_document(expected)` | Fail unless the active tab matches `expected` (title, filename, or stem) |
| `select_document(name)` | Activate the open tab matching `name` (title, filename, or stem) |
| `inspect_document(...)` / `list_nodes(...)` | Layer tree / scoped nodes (`document.inspect`) |
| `get_node(id)` | One node (`document.node`) |
| `find_objects(ObjectQuery(...))` | `document.find` + client-side size/visibility/regex filters |
| `find_by_name(name, exact=True)` | `document.find` by Layers-panel name |
| `node_bounds(id)` | `{x, y, width, height}` from `document.node` |
| `document_path()` | On-disk path of the active document |
| `wait_for_background(timeout_sec=120)` | Poll `ui.inspect` until Background Save/Export jobs finish |
| `save_copy(out_path)` | `file.saveCopy` then wait for background jobs; assert the file exists |
| `backup_document()` | Copy into `<doc folder>/_BAK/<stem>_<date><ext>` (plus a `.vectorcraft` copy) |
| `duplicate(ids)` / `unique_name(base)` / `place_file(path, rect, name=…)` | Duplicate, uniquify a Layers-panel name, `file.place` into a rect |
| `undo_group(label)` | Context manager: `edit.undoGroup.begin` / `.end` |

Replace named figure layers (backup first, then one undo step):

```python
from vectorcraft_scripting import connect_first, layer_geometry, replace_layers

live = connect_first()
layer_names = ['ScatterOverTime_Laps_perTimeBin.svg', 'ScatterOverTime_PBEs_perTimeBin.svg']
layer_geom_df = layer_geometry(live, layer_names)
rects = layer_geom_df[['x', 'y', 'width', 'height']].to_dict(orient='index')

live.select_document(expected_title)  # activate the matching open tab
backup_paths = live.backup_document(expect_title=expected_title)
with live.undo_group('Replace figure layers'):
    replacement_df = replace_layers(live, rects, source_folder, expect_title=expected_title)
```

## Notes

- Paths passed to `--in`, `--export`, and `file.place` are made absolute so Windows working-directory quirks do not matter.
- Live control does not auto-launch the GUI; start VectorCraft with `--control` first.
- MCP stdio is not wrapped; use the CLI batch path or the live TCP client above.
