---
title: watchable parameters
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/837954ad-7264-4e33-b125-8dfb6576fcb1
url: https://phohale.sourcegraph.app/deepsearch/295
created: 2025-12-09T21:33:21Z
updated: 2026-07-22T17:45:35Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis find all information about my preferred implementation of watchable parameters

Based on my analysis of your codebase, here's comprehensive information about your preferred implementation of watchable parameters:

## Your Preferred Watchable Parameters Implementation

### Core Approach: `param.Parameterized`

You explicitly state in multiple places that **"Upon reviewing many different versions of my plotting implementations, this Param-based one is the most succinct and smooth."** This appears in:
- [EpochDisplayConfig](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/epochs_plotting_mixins.py?L33-L36)
- [OccupancyPlottingConfig](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/occupancy_plotting_mixins.py?L8-L11)
- [TrackShapePlottingConfig](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/MazeRenderingMixin.py?L13-L16)

### Key Components

#### 1. **Base Classes**
- [`BasePlotDataParams`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Model/Configs/ParamConfigs.py?L13-L17) - Base class inheriting from `param.Parameterized`
- [`BaseDisplayStateParams`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Model/Configs/ParamConfigs.py?L7-L10) - Alternative base class

#### 2. **Main Implementation Examples**

**[EpochDisplayConfig](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/epochs_plotting_mixins.py?L33-L90)** - The canonical example:

```python
class EpochDisplayConfig(BasePlotDataParams):
    """ This class uses the 'param' library to observe changes to its members 
    and perform corresponding updates to the class that holds it when they happen.
    """
    # Overriding defaults from parent
    name = param.String(default='SessionEpochs', doc='Name of the epochs')
    isVisible = param.Boolean(default=False, doc="Whether the epochs are visible")
    
    # Bar properties:
    pen_color = param.Color(default='#00ffff', doc="The edge Color")
    pen_opacity = param.Number(default=0.8, bounds=(0.0, 1.0), step=0.1)
    brush_color = param.Color(default='#00ffff', doc="The fill Color")
    brush_opacity = param.Number(default=0.5, bounds=(0.0, 1.0), step=0.1)
    
    # Location Properties:
    height = param.Number(default=7.5, bounds=(0.1, 50.0), step=0.1)
    y_location = param.Number(default=-12.0, bounds=(-200.0, 1000.0), step=2.0)
    
    @staticmethod
    def _config_update_watch_labels():
        """Returns list of parameter names that trigger full updates"""
        return ['pen_color', 'pen_opacity', 'brush_color', 'brush_opacity', 
                'height', 'y_location', 'isVisible']
    
    @staticmethod
    def _config_visibility_watch_labels():
        """Returns list of parameter names that trigger visibility updates"""
        return ['isVisible']
```

**[SingleNeuronPlottingExtended](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Model/Configs/NeuronPlottingParamConfig.py?L177-L197)** - For neuron visualization:

```python
class SingleNeuronPlottingExtended(ExtendedPlotDataParams):
    """Represents the visual config for a single neuron."""
    spikesVisible = param.Boolean(default=False, doc="Whether the spikes are visible")
    
    @property
    def neuron_id(self):
        return int(self.name)
    
    @property
    def qcolor(self):
        """Converts param.Color to QtGui.QColor"""
        return QtGui.QColor(self.color)
```

### 3. **Setting Up Watchers**

From [TrackShapePlottingConfig](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/MazeRenderingMixin.py?L18-L25):

```python
# Setup watchers in your implementor class:
self.occupancy_plotting_config.param.watch(
    self.plot_occupancy_bars, 
    OccupancyPlottingConfig._config_update_watch_labels(), 
    queued=True
)

self.occupancy_plotting_config.param.watch(
    self.on_occupancy_plot_update_visibility, 
    OccupancyPlottingConfig._config_visibility_watch_labels(), 
    queued=True
)
```

### 4. **Using `@param.depends` Decorator**

From [panel_placefield.py](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/Panel/panel_placefield.py?L80-L92):

```python
@param.depends('name', 'color', 'isVisible', 'spikesVisible', watch=True)
def _sync_widgets(self):
    """Automatically called when any watched parameter changes"""
    self._wgt_label_button.name = self.name
    self._wgt_color_picker.value = self.color
    self._wgt_toggle_visible.value = self.isVisible
    self._wgt_toggle_spikes.value = self.spikesVisible

@param.depends('_wgt_toggle_visible.value', watch=True)
def _on_toggle_plot_visible_changed(self):
    """Responds to specific widget changes"""
    if self._callbacks is not None:
        self._callbacks['pf'](self.config_from_state())
```

### 5. **Available Parameter Types**

From your codebase, you commonly use:
- `param.String` - String values
- `param.Boolean` - Boolean flags
- `param.Number` - Numeric values with optional bounds and step
- `param.Color` - Color values (hex format)
- `param.Dict` - Dictionary values
- `param.Integer` - Integer values
- `param.List` - List values
- `param.ObjectSelector` - Selection from predefined options
- `param.ClassSelector` - Type-constrained class instances

### 6. **Key Benefits You Noted**

1. **Auto-observable**: Changes to parameters automatically trigger callbacks
2. **Succinct**: Minimal boilerplate compared to manual property setters
3. **Smooth**: Integrates well with Panel widgets and PyQt
4. **Type-safe**: Built-in validation and bounds checking
5. **GUI-ready**: Works seamlessly with `panel` and PyQt parameter trees

### 7. **Integration with GUI Frameworks**

**Panel Integration**:
```python
import panel as pn
# Automatically creates widgets from param config
pn.Param(config.param, name="Config Panel")
```

**PyQt Integration** via [ParameterTree](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ParameterTreeWidget.py):
```python
from pyphoplacecellanalysis.External.pyqtgraph.parametertree import Parameter, ParameterTree
paramTree = ParameterTree()
paramTree.setParameters(p, showTop=False)
```

I've created a class diagram showing the architecture of your preferred watchable parameters implementation. This `param.Parameterized`-based approach is your go-to pattern for creating observable configuration objects throughout your visualization codebase.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis watchable parameters implementation](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20watchable%20parameters%20implementation)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis preferred watchable](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20preferred%20watchable)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ param.Parameterized](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20param.Parameterized)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis param parameterized class](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20param%20parameterized%20class)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ import param](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20import%20param)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Model/Configs/ParamConfigs.py@26f673d](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Model/Configs/ParamConfigs.py)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/epochs_plotting_mixins.py@26f673d](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/epochs_plotting_mixins.py)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Model/Configs/NeuronPlottingParamConfig.py@26f673d](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Model/Configs/NeuronPlottingParamConfig.py)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis attrs_to_parameters decorator](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20attrs_to_parameters%20decorator)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Model/SpecificComputationParameterTypes.py@26f673d?L1-150](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Model/SpecificComputationParameterTypes.py?L1-150)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ @param.depends watch=True](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20@param.depends%20watch=True)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Panel/panel_placefield.py@26f673d](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/GUI/Panel/panel_placefield.py)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "most succinct" OR "preferred" param](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22most%20succinct%22%20OR%20%22preferred%22%20param)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis observe changes param library](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20observe%20changes%20param%20library)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/MazeRenderingMixin.py@26f673d](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/MazeRenderingMixin.py)

## Suggested Follow-ups

- Can you show me an example of how to use `param.depends` to trigger a function when multiple parameters change?
- What are the performance implications of using `param.watch` with `queued=True`, and when should I use it?

