# Serves to document the various components that are used to render flexible/dynamic time-dependent curves in 3D.




## Properties:


```python
class RenderDataseries(SimplePrintable, PrettyPrintable, QtCore.QObject):
    """ Serves as a very flexible mapping between any temporal data values and the final spatial location to render them by storing a list of configs for each series (self.data_series_config_list)
        It uses its internal pre_spatial_to_spatial_mappings (set on initialization) when self.get_data_series_spatial_values(curr_windowed_df) is called to get the spatial_values for each series from its internal non-spatial ones
        ...
```

data_series_specs


```python
class Specific3DTimeCurvesHelper:
    """ Static helper methods that build commonly known 3D time curve datasources and add them to the provided plot.
    ...
```


## Datasources:

```python
class SpikesDataframeDatasource(DataframeDatasource):
    """ Provides neural spiking data for one or more neuron (unit) and the timestamps at which they occur 't'.
    
```

## Windowed Datasources:
These objects wrap both a TimeWindow and a Datasource of some kinda, meaning they're able to provide some sort of active_windowed_data for the times of the it's time window.

```python
class LiveWindowedData(SimplePrintable, PrettyPrintable, QtCore.QObject):
    """ an optional adapter between a DataSource and the GUI/graphic that uses it.
    Serves as an intermediate to TimeWindow and Datasource.
    It subscribes to TimeWindow updates, and for each update it fetches the appropriate data from its internally owned DataSource and emits a singal containing this data that can be used to update the GUI/graphic classes that subscribe to it.
    ...
```

```python
class SpikesDataframeWindow(LiveWindowedData):
    """ a zoomable (variable sized) window into a SpikesDataframeDatasource with a time axis windowed by the active TimeWindow
    ...
```





### Datasource Properties:

```python
active_random_test_plot_curve_datasource.data_series_specs
```

## Time Window - owned by parent:
```python
class TimeWindow(SimplePrintable, PrettyPrintable, QtCore.QObject):
    """ a zoomable (variable sized) window into a dataset with a time axis
    ...
```



data_series_specs_changed_signal
