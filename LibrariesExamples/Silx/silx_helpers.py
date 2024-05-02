import functools

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
        self.addDockWidget(qt.Qt.TopDockWidgetArea, self._docker)
        self.setWindowFlags(qt.Qt.Widget)

        # connect signal / slot
        self._updateModeControl.sigUpdateModeChanged.connect(self._roiStatsWindow._setUpdateMode)
        callback = functools.partial(self._roiStatsWindow._updateAllStats, is_request=True)
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
        self.plot.getDefaultColormap().setName('viridis')
        self.plot.setKeepDataAspectRatio(True)

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
        self.addDockWidget(qt.Qt.TopDockWidgetArea, self._roisTabWidgetDockWidget)

        # create Dock widgets
        self._roiStatsWindowDockWidget = qt.QDockWidget(parent=self)
        self._roiStatsWindowDockWidget.setWidget(self._statsWidget)
        # move the docker contain in the parent widget
        # self.addDockWidget(qt.Qt.BottomDockWidgetArea, self._statsWidget._docker) # worthless. Just asks how to refresh
        self.addDockWidget(qt.Qt.BottomDockWidgetArea, self._roiStatsWindowDockWidget)

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
        
