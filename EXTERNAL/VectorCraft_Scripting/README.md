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

With `auto_reconnect=True` (default), idle peer closes (~5 minutes on the app side) are recovered automatically: the next RPC reconnects and retries once. Pass `auto_reconnect=False` (or call `live.reconnect()` yourself) for strict fail-fast behavior.

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
| `TextHelpers.set_text_font_size_centered(live, ids, size_pt)` | Set Type objects to `size_pt` via `text.setStyle`, then `object.move` so each bounds center stays fixed (module alias: `set_text_font_size_centered`) |
| `TextHelpers.group_text_by_font_size(live, limit=10000, effective=True)` | Find Type objects; group by run font size; each row has `id`, `name`, `text`, `sizes`, `runs` (module alias: `group_text_by_font_size`) |

```python
from vectorcraft_scripting import group_text_by_font_size, set_text_font_size_centered

grouped = group_text_by_font_size(live)
for size_pt, rows in sorted(grouped['by_size'].items()):
    print(f'{size_pt:g} pt → {len(rows)} objects')
    for r in rows[:50]:
        print(f"  id={r['id']}  {r['text']!r}")

with live.undo_group("Bump axis labels"):
    set_text_font_size_centered(live, label_ids, size_pt=14)
```

Replace named figure layers (filesystem backup first, then one undo step). With `backup_layers=True`, each source is duplicated into a root-level `{YYYY-MM-DD}_Backup` folder as `{name}_{date}` (same-day re-runs reuse that folder); `*_PROGRAMMATIC` replacements stay beside the original:

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

## Record and replay changes

Record a series of changes made to an object, then apply them to a similar object via script — in the same document or a different one. Two complementary routes:

### Route A: record GUI edits (native Actions, `.vcactions`)

Wraps VectorCraft's Actions panel recorder. Edits made by hand inside the block are recorded; playback is selection/id-relative, so **select the similar target object first** (`select_ids`) before playing.

```python
from vectorcraft_scripting import Actions, connect_first

live = connect_first()
with Actions.record_action(live, name='Bump axis labels'):
    ...  # make the edits by hand in the VectorCraft GUI now
Actions.save(live, 'bump_labels.vcactions', set_index=0)

# Later, on a similar figure: activate its tab, select the similar object, play.
target = live.find_by_name('ScatterOverTime_PBEs_perTimeBin.svg', exact=True)
Actions.play(live, action='Bump axis labels', target_document='Other Figure.pdf', select_ids=[m['id'] for m in target])

# Reuse a saved set in a fresh session:
Actions.load(live, 'bump_labels.vcactions')
```

`Actions.batch(live, commands, folder, ...)` runs explicit steps on every file in a folder (`action.batch`).

### Route B: record scripted edits (JSON recipe, id-independent)

`CommandRecorder` wraps `live.execute` for the duration of a `with` block and captures every mutating command (including `TextHelpers` / `replace_layers` helpers). Object references are stored as name/kind/text/bounds descriptors instead of raw ids, so the recipe replays onto similar objects in another document.

```python
from vectorcraft_scripting import CommandRecorder, TextHelpers, connect_first, load_recipe, replay_recipe, save_recipe

live = connect_first()
with CommandRecorder(live) as rec:
    with live.undo_group('Bump axis tick labels'):
        TextHelpers.set_text_font_size_centered(live, text_axis_tick_label_ids, size_pt=8)
save_recipe(rec.recipe, 'bump_labels.json')

# Replay onto similar objects in another open document (backup first, ONE undo step):
replay_recipe(live, load_recipe('bump_labels.json'), target_document='Other Figure.pdf')

# Same-document replay onto different objects: resolve them yourself and pass id_map:
replay_recipe(live, rec.recipe, id_map={old_id: new_id for old_id, new_id in pairs}, backup=True)
```

Resolution order per object: explicit `id_map` entry → exact name (+kind) match → for Type objects, text match disambiguated by nearest recorded bounds. Ambiguous or unresolved descriptors fail loudly; pass `id_map` for those cases. Replay runs through `command.batch`: one undo step, full rollback on the first error.

Caveats:
- Geometry-derived deltas (e.g. the corrective `object.move` inside `set_text_font_size_centered`) replay exactly as recorded — correct for same-size similar objects, not re-derived for other sizes.
- Steps that create objects (`file.place`, `layer.new`, …) reference ids that only exist in the source document; replaying those onto another document needs an explicit `id_map`.
- `.vcactions` playback (Route A) bakes in the recorded absolute ids/coordinates: a recorded `select.set` re-selects the original ids, so `select_ids` cannot redirect an action that captured its own selection. Route A is reliable for re-running on the same document/ids; prefer Route B recipes for cross-document portability.
- `ui.inspect` exposes the Actions panel's recorded steps as `[command, params]` pairs (`ui.action_sets[].actions[].steps`), so a recorded action can be read back and converted into Route B steps if needed.

## Notes

- Paths passed to `--in`, `--export`, and `file.place` are made absolute so Windows working-directory quirks do not matter.
- Live control does not auto-launch the GUI; start VectorCraft with `--control` first.
- MCP stdio is not wrapped; use the CLI batch path or the live TCP client above.
