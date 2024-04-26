import functools
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
# pd.options.mode.dtype_backend = 'pyarrow' # use new pyarrow backend instead of numpy
from attrs import define, field, fields, Factory
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from copy import deepcopy

import numpy as np
import pandas as pd

from silx.gui import qt
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.plot import PlotWindow, ImageView
from silx.gui.plot.Profile import ProfileToolBar

from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.tools.roi import RegionOfInterestTableWidget
from silx.gui.plot.tools.roi import RoiModeSelectorAction
from silx.gui.plot.items.roi import RectangleROI, BandROI, LineROI
from silx.gui.plot.items import LineMixIn, SymbolMixIn, FillMixIn
from silx.gui.plot.actions import control as control_actions

from silx.gui.plot.ROIStatsWidget import ROIStatsWidget
from silx.gui.plot.StatsWidget import UpdateModeWidget
from silx.gui.plot import Plot2D


""" 

Uses Silx


"""


from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from attrs import define, field, Factory
from typing import Tuple, List

@define(slots=False)
class RadonDebugValue:
    """ Values for a single epoch. Class to hold debugging information for a transformation process """
    a_posterior: float = field()
    active_epoch_info_tuple: Tuple = field()	
    active_num_neighbors: int = field()
    active_neighbors_arr: List = field()


    start_point: Tuple[float, float] = field()
    end_point: Tuple[float, float] = field()
    band_width: float = field()
    
    

# decoder_laps_radon_transform_df_dict
# │   ├── decoder_laps_radon_transform_df_dict: dict
# 	│   ├── long_LR: pandas.core.frame.DataFrame (children omitted) - (84, 4)
# 	│   ├── long_RL: pandas.core.frame.DataFrame (children omitted) - (84, 4)
# 	│   ├── short_LR: pandas.core.frame.DataFrame (children omitted) - (84, 4)
# 	│   ├── short_RL: pandas.core.frame.DataFrame (children omitted) - (84, 4)
# │   ├── decoder_laps_radon_transform_extras_dict: dict
# 	│   ├── long_LR: list - (1, 1, 2, 84)
# 	│   ├── long_RL: list - (1, 1, 2, 84)
# 	│   ├── short_LR: list - (1, 1, 2, 84)
# 	│   ├── short_RL: list - (1, 1, 2, 84)

# decoder_ripple_radon_transform_df_dict 
# a_radon_transform_output = np.squeeze(deepcopy(decoder_laps_radon_transform_extras_dict['long_LR'])) # collapse singleton dimensions with np.squeeze: (1, 1, 2, 84) -> (2, 84) # (2, n_epochs)


# np.shape(a_radon_transform_output)

# np.squeeze(a_radon_transform_output).shape
# len(a_radon_transform_output)

@define(slots=False)
class RadonTransformDebugger:
    """ interactive debugger """
    pos_bin_size: float = field()
    decoder_laps_filter_epochs_decoder_result_dict: Dict = field()
    decoder_laps_radon_transform_extras_dict: Dict = field()
    
    active_decoder_name: str = field(default='long_LR')
    active_epoch_idx: int = field(default=3)

    @property
    def result(self) -> DecodedFilterEpochsResult:
        return self.decoder_laps_filter_epochs_decoder_result_dict[self.active_decoder_name]

    @property
    def active_filter_epochs(self) -> pd.DataFrame:
        return self.result.active_filter_epochs.to_dataframe()

    @property
    def time_bin_size(self) -> float:
        return float(self.result.decoding_time_bin_size)

    @property
    def num_neighbours(self) -> NDArray:
        return  np.squeeze(deepcopy(self.decoder_laps_radon_transform_extras_dict[self.active_decoder_name]))[0]
    
    @property
    def neighbors_arr(self) -> NDArray:
        return  np.squeeze(deepcopy(self.decoder_laps_radon_transform_extras_dict[self.active_decoder_name]))[1]
    

    @property
    def active_radon_values(self) -> RadonDebugValue:
        """ value for current index """
        # a_posterior, (start_point, end_point, band_width), (active_num_neighbors, active_neighbors_arr) = self.on_update_epoch_idx(active_epoch_idx=self.active_epoch_idx)
        # return RadonDebugValue(a_posterior=a_posterior, active_epoch_info_tuple=active_epoch_info_tuple, start_point=start_point, end_point=end_point, band_width=band_width, active_num_neighbors=active_num_neighbors, active_neighbors_arr=active_neighbors_arr)
        return self.on_update_epoch_idx(active_epoch_idx=self.active_epoch_idx)
            

    def on_update_epoch_idx(self, active_epoch_idx: int):
        """ 
        Usage:
            a_posterior, (start_point, end_point, band_width), (active_num_neighbors, active_neighbors_arr) = on_update_epoch_idx(active_epoch_idx=5)
        
        captures: pos_bin_size, time_bin_size """
        ## ON UPDATE: active_epoch_idx
        self.active_epoch_idx = active_epoch_idx ## update the index
        
        ## INPUTS: pos_bin_size
        a_posterior = self.result.p_x_given_n_list[active_epoch_idx].copy()

        # num_neighbours # (84,)
        # np.shape(neighbors_arr) # (84,)

        # neighbors_arr[0].shape # (57, 66)
        # neighbors_arr[1].shape # (57, 66)

        # for a_neighbors_arr in neighbors_arr:
        # 	print(f'np.shape(a_neighbors_arr): {np.shape(a_neighbors_arr)}') # np.shape(a_neighbors_arr): (57, N[epoch_idx]) - where N[epoch_idx] = result.nbins[epoch_idx]

        active_num_neighbors: int = self.num_neighbours[self.active_epoch_idx]
        active_neighbors_arr = self.neighbors_arr[self.active_epoch_idx].copy()

        # n_arr_v = (2 * num_neighbours[0] + 1)
        # print(f"n_arr_v: {n_arr_v}")

        # flat_neighbors_arr = np.array(neighbors_arr)
        # np.shape(flat_neighbors_arr)


        ## OUTPUTS: active_num_neighbors, active_neighbors_arr, a_posterior
        # decoder_laps_radon_transform_df: pd.DataFrame = decoder_laps_radon_transform_df_dict[active_decoder_name].copy()
        # decoder_laps_radon_transform_df

        # active_filter_epochs[active_filter_epochs[''
        active_epoch_info_tuple = tuple(self.active_filter_epochs.itertuples(name='EpochTuple'))[self.active_epoch_idx]
        # active_epoch_info_tuple
        # (active_epoch_info_tuple.velocity, active_epoch_info_tuple.intercept)

        ## build the ROI properties:
        # start_point = (0.0, active_epoch_info_tuple.intercept)
        # end_point = (active_epoch_info_tuple.duration, (active_epoch_info_tuple.duration * active_epoch_info_tuple.velocity))
        # band_width = pos_bin_size * float(active_num_neighbors)

        start_point = [0.0, active_epoch_info_tuple.intercept]
        end_point = [active_epoch_info_tuple.duration, (active_epoch_info_tuple.duration * active_epoch_info_tuple.velocity)]
        band_width = self.pos_bin_size * float(active_num_neighbors)

        ## convert time (x) coordinates:
        time_bin_size: float = float(self.result.decoding_time_bin_size)
        start_point[0] = (start_point[0]/time_bin_size)
        end_point[0] = (end_point[0]/time_bin_size)
        # end_point[1] = (end_point[1]/time_bin_size) # not sure about this one

        ## convert from position (cm) units to y-bins:
        pos_bin_size: float = float(self.pos_bin_size) # passed directly
        start_point[1] = (start_point[1]/pos_bin_size)
        # end_point[1] = (end_point[1]/pos_bin_size) # not sure about this one
        
        ## OUTPUTS: a_posterior, (start_point, end_point, band_width), (active_num_neighbors, active_neighbors_arr)
        # Initialize an instance of TransformDebugger using the variables as keyword arguments
        # transform_debug_instance = RadonDebugValue(a_posterior=a_posterior, start_point=start_point, end_point=end_point, band_width=band_width, active_num_neighbors=active_num_neighbors, active_neighbors_arr=active_neighbors_arr)

        # return a_posterior, active_epoch_info_tuple, (active_num_neighbors, active_neighbors_arr), (start_point, end_point, band_width)
        return RadonDebugValue(a_posterior=a_posterior, active_epoch_info_tuple=active_epoch_info_tuple,
                                active_num_neighbors=active_num_neighbors, active_neighbors_arr=active_neighbors_arr,
                                start_point=start_point, end_point=end_point, band_width=band_width)
    




# ---------------------------------------------------------------------------- #
#                            Widgets/Visual Classes                            #
# ---------------------------------------------------------------------------- #

class AutoHideToolBar(qt.QToolBar):
    """A toolbar which hide itself if no actions are visible"""

    def actionEvent(self, event):
        if event.type() == qt.QEvent.ActionChanged:
            self._updateVisibility()
        return qt.QToolBar.actionEvent(self, event)

    def _updateVisibility(self):
        visible = False
        for action in self.actions():
            if action.isVisible():
                visible = True
                break
        self.setVisible(visible)

class _RoiStatsWidget(qt.QMainWindow):
    """
    A widget used to display a table of stats for the ROIs
    Associates ROIStatsWidget and UpdateModeWidget
    """
    def __init__(self, parent=None, plot=None, mode=None):
        assert plot is not None
        qt.QMainWindow.__init__(self, parent)
        self._roiStatsWindow = ROIStatsWidget(plot=plot)
        self.setCentralWidget(self._roiStatsWindow)

        # update mode docker
        self._updateModeControl = UpdateModeWidget(parent=self)
        self._docker = qt.QDockWidget(parent=self)
        self._docker.setWidget(self._updateModeControl)
        self.addDockWidget(qt.Qt.TopDockWidgetArea,
                           self._docker)
        self.setWindowFlags(qt.Qt.Widget)

        # connect signal / slot
        self._updateModeControl.sigUpdateModeChanged.connect(
            self._roiStatsWindow._setUpdateMode)
        callback = functools.partial(self._roiStatsWindow._updateAllStats,
                                     is_request=True)
        self._updateModeControl.sigUpdateRequested.connect(callback)

        # expose API
        self.registerROI = self._roiStatsWindow.registerROI
        self.setStats = self._roiStatsWindow.setStats
        self.addItem = self._roiStatsWindow.addItem
        self.removeItem = self._roiStatsWindow.removeItem
        self.setUpdateMode = self._updateModeControl.setUpdateMode

        # setup
        self._updateModeControl.setUpdateMode('auto')

class _RoiStatsDisplayExWindow(qt.QMainWindow):
    """
    Simple window to group the different statistics actors
    """
    def __init__(self, parent=None, mode=None):
        qt.QMainWindow.__init__(self, parent)
        self.plot = Plot2D()
        self.setCentralWidget(self.plot)

        # 1D roi management
        self._curveRoiWidget = self.plot.getCurvesRoiDockWidget().widget()
        # hide last columns which are of no use now
        # for index in (5, 6, 7, 8):
        #     self._curveRoiWidget.roiTable.setColumnHidden(index, True)

        # 2D - 3D roi manager
        self._regionManager = RegionOfInterestManager(parent=self.plot)

        # Create the table widget displaying
        self._2DRoiWidget = RegionOfInterestTableWidget()
        self._2DRoiWidget.setRegionOfInterestManager(self._regionManager)

        # tabWidget for displaying the rois
        self._roisTabWidget = qt.QTabWidget(parent=self)
        if hasattr(self._roisTabWidget, 'setTabBarAutoHide'):
            self._roisTabWidget.setTabBarAutoHide(True)

        # widget for displaying stats results and update mode
        self._statsWidget = _RoiStatsWidget(parent=self, plot=self.plot)

        # create Dock widgets
        self._roisTabWidgetDockWidget = qt.QDockWidget(parent=self)
        self._roisTabWidgetDockWidget.setWidget(self._roisTabWidget)
        self.addDockWidget(qt.Qt.RightDockWidgetArea,
                           self._roisTabWidgetDockWidget)

        # create Dock widgets
        self._roiStatsWindowDockWidget = qt.QDockWidget(parent=self)
        self._roiStatsWindowDockWidget.setWidget(self._statsWidget)
        # move the docker contain in the parent widget
        self.addDockWidget(qt.Qt.RightDockWidgetArea,
                           self._statsWidget._docker)
        self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                           self._roiStatsWindowDockWidget)

        # expose API
        self.setUpdateMode = self._statsWidget.setUpdateMode

    def setRois(self, rois1D=None, rois2D=None):
        rois1D = rois1D or ()
        rois2D = rois2D or ()
        self._curveRoiWidget.setRois(rois1D)
        for roi1D in rois1D:
            self._statsWidget.registerROI(roi1D)

        for roi2D in rois2D:
            self._regionManager.addRoi(roi2D)
            self._statsWidget.registerROI(roi2D)

        # update manage tab visibility
        if len(rois2D) > 0:
            self._roisTabWidget.addTab(self._2DRoiWidget, '2D roi(s)')
        if len(rois1D) > 0:
            self._roisTabWidget.addTab(self._curveRoiWidget, '1D roi(s)')

    def setStats(self, stats):
        self._statsWidget.setStats(stats=stats)

    def addItem(self, item, roi):
        self._statsWidget.addItem(roi=roi, plotItem=item)
        

def roi_radon_transform_score(arr):
    """ a stats function that takes the ROI and returns the radon transform score """
    # print(f'np.shape(arr): {np.shape(arr)}')
    # return np.nanmean(arr, axis=1)
    # print(f'np.sum(np.isnan(arr)): {np.sum(np.isnan(arr))}')
    column_medians = np.nanmedian(arr, axis=0)
    filled_arr = [arr[:,i].filled(column_medians[i]) for i in np.arange(np.shape(arr)[1])]
    return np.nanmean(filled_arr)

