---
name: Raster window attribute error
overview: The spike-raster window crashes while reading `type_of_3d_plotter` back off `self.params`. That read is not explained by the recent computation-parameter edits, and the current `VisualizationParameters` class does not raise this AttributeError when the field is missing.
todos:
  - id: pass-local-plotter
    content: Pass the local type_of_3d_plotter into initUI instead of reading it back from self.params
    status: pending
isProject: false
---

# Why the spike raster window crashes

The notebook call reaches [`Spike3DRasterWindowWidget.find_or_create_if_needed`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\SpikeRasterWindows\Spike3DRasterWindowWidget.py), which builds the window through `_display_spike_rasters_pyqtplot_2D`. That display function forces `type_of_3d_plotter=None` (2D raster only) and then constructs `Spike3DRasterWindowWidget`.

Inside `__init__`, the value is stored and then read back:

```232:270:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\SpikeRasterWindows\Spike3DRasterWindowWidget.py
self.params = VisualizationParameters(..., type_of_3d_plotter=type_of_3d_plotter, ...)
self.params.type_of_3d_plotter = type_of_3d_plotter
# setupUi / mixin setup
self.initUI(..., type_of_3d_plotter=self.params.type_of_3d_plotter)
```

Line 270 is the crash. `initUI` only needs that value to choose no 3D plot (`None`), pyqtgraph, or vedo. The local argument `type_of_3d_plotter` is already in scope; the failure is the round-trip through `self.params`.

`self.params` is a [`VisualizationParameters`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\DataStructure\general_parameter_containers.py), a [`DynamicParameters`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\DataStructure\dynamic_parameters.py) bag. Fields live in an internal `_mapping`. A missing key raises **KeyError** from `__getattr__`, not AttributeError. On this environment (Python 3.9.13) that was checked directly: storing `type_of_3d_plotter=None` reads back as `None`, and a missing key raises `KeyError: 'type_of_3d_plotter'`.

The pasted exception is `AttributeError: 'VisualizationParameters' object has no attribute 'type_of_3d_plotter'`, and the stack never enters `dynamic_parameters.py`. That is normal Python attribute lookup on an object whose class is not using this dynamic `__getattr__`. So the instance at line 270 is not behaving like the current `DynamicParameters` implementation, even though the source still assigns the field a few lines earlier.

## Was this a recent change?

No source change in the last few weeks removes this field.

- **2026-07-14** (`fix Spike3DRasterWindowWidget params init order`): moved `self.params = ...` to before `setupUi`, because the overridden `setWindowTitle` reads `self.params` while the `.ui` file is applied. The `self.params.type_of_3d_plotter` read was already there.
- **2026-09-02** (`LauncherWidget title`): added `self.params.launcher_display_suffix = None` on the same object and included it in the window-title composer. It does not delete `type_of_3d_plotter`.
- `DynamicParameters.__getattr__` last changed in 2023 (deepcopy / AttributeError only for `__deepcopy__`). The April 2026 edit only touched `__repr__`.
- The in-progress computation parameter sync ([`SpecificComputationParameterTypes.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\SpecificComputationParameterTypes.py)) is a different set of classes and is not on this call path.

If this same cell worked after 2026-09-02, nothing committed since then explains the crash. The nearest code edits are the July 14 init-order move and the September 2 title field.

## If you want it to open again

In [`Spike3DRasterWindowWidget.__init__`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\SpikeRasterWindows\Spike3DRasterWindowWidget.py), pass the local `type_of_3d_plotter` into `initUI` instead of `self.params.type_of_3d_plotter`. Keep the assignment onto `self.params` so later readers still see it.

Before retrying, restart the notebook kernel so it is not holding a redefined `VisualizationParameters`. If it still fails after that one-line change, the next check is what `self.params` actually is just before `initUI` (`type(self.params)`, its MRO, and `dict(self.params._mapping)`).
