"""
This example demonstrates the creation of a plot with 
DateAxisItem and a customized ViewBox. 
"""


import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.CustomGraphicsLayoutWidget import CustomViewBox
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.DraggableGraphicsWidgetMixin import DragUpdateAction
# class CustomViewBox(pg.ViewBox):
#     sigLeftDrag = QtCore.Signal(object)  # optional custom signal
    
#     def __init__(self, *args, **kwds):
#         kwds['enableMenu'] = False
#         pg.ViewBox.__init__(self, *args, **kwds)
#         self.setMouseMode(self.RectMode)
        
#     ## reimplement right-click to zoom out
#     def mouseClickEvent(self, ev):
#         # if ev.button() == QtCore.Qt.MouseButton.RightButton:
#         #     ## this mode enables right-mouse clicks to reset the plot range
#         #     self.autoRange()            
#         print(f'.mouseClickEvent(ev: {ev})')
#         # Custom logic
#         ev.accept()  # or ev.ignore() if you want default handling
#         # Optionally call super() if desired:
#         # super().mousePressEvent(ev)
        
    
#     ## reimplement mouseDragEvent to disable continuous axis zoom
#     def mouseDragEvent(self, ev, axis=None):
#         print(f'.mouseDragEvent(ev: {ev}, axis={axis})')
#         if (ev.button() == QtCore.Qt.MouseButton.RightButton): # (axis is not None) and
#             ev.accept()
#             # ev.ignore()
#         elif (ev.button() == QtCore.Qt.MouseButton.LeftButton):
#             # axis is not None and 
#             # Emit a signal or directly update the slider here
#             new_start_point = self.mapSceneToView(ev.pos()) # PyQt5.QtCore.QPointF
#             new_start_t = new_start_point.x()
#             print(f'new_start_t: {new_start_t}')
#             self.sigLeftDrag.emit(new_start_t)
#             ev.accept()
#         else:
#             # pg.ViewBox.mouseDragEvent(self, ev, axis=axis)            
#             # Custom dragging logic
#             ev.accept()
#             # super().mouseDragEvent(ev, axis=axis)  # only if you want default drag/pan
        

# .mouseDragEvent
app = pg.mkQApp()

# axis = pg.DateAxisItem(orientation='bottom')
axis = pg.AxisItem(orientation='bottom')
update_mode: DragUpdateAction = DragUpdateAction.TRANSLATE
# _drag_start_point = None
vb = CustomViewBox()

pw = pg.PlotWidget(viewBox=vb, axisItems={'bottom': axis}, enableMenu=False, title="PlotItem with DateAxisItem, custom ViewBox and markers on x axis<br>Menu disabled, mouse behavior changed: left-drag to zoom, right-click to reset zoom")


dates = np.arange(8) # * (3600*24*356)
a_plot = pw.plot(x=dates, y=[1,6,2,4,3,5,6,8], symbol='o')


# Add the linear region overlay:
scroll_window_region = CustomLinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=a_plot) # bound the LinearRegionItem to the plotted data
scroll_window_region.setObjectName('scroll_window_region')
scroll_window_region.setZValue(10)
# Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this item when doing auto-range calculations.
# pw.plotItem.layout.addItem(scroll_window_region, ignoreBounds=True)
pw.plotItem.addItem(scroll_window_region, ignoreBounds=True)
# scroll_window_region.sigRegionChanged.connect(self._Render2DScrollWindowPlot_on_linear_region_item_update)

# @QtCore.pyqtSlot(float, float)
# def on_window_update(new_start=None, new_end=None):
# def on_window_update(new_start=None):
def on_window_update(new_value=None):
    """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
    print(f'on_window_update(new_value: {new_value})')
    # Make sure that the scroller isn't too tiny to grab.
    min_x, max_x = scroll_window_region.getRegion() # get the current region
    print(f'\tmin_x, max_x = scroll_window_region.getRegion(): (min_x: {min_x}, max_x: {max_x})')
    # print(f'on_window_update(new_start: {new_start}, new_end: {new_end})')

    if update_mode.value == DragUpdateAction.ALIGN_START.value:
        new_start: float = new_value
        fixed_width: float = abs(max(max_x, min_x) - min(max_x, min_x))
        new_end = new_start + fixed_width
        scroll_window_region.setRegion([new_start, new_end]) # adjust scroll control
        
    elif update_mode.value == DragUpdateAction.TRANSLATE.value:
        # do not replace start, just translate the existing
        # min_x, max_x
        # vb._
        new_change_x: float = new_value
        shifted_min_x = min_x + new_change_x
        shifted_max_x = max_x + new_change_x
        print(f'\t(shifted_min_x: {shifted_min_x}, shifted_max_x: {shifted_max_x})')
        scroll_window_region.setRegion([shifted_min_x, shifted_max_x])
        
    else:
        raise NotImplementedError(f'update_mode: {update_mode}')
    
    
    # scroll_window_region.setRegion([new_start, None])
    # _drag_start_point = 

_conn = vb.sigLeftDrag.connect(on_window_update)

# scroll_window_region.setRegion(
# min_x, max_x = scroll_window_region.getRegion() # get the current region

pw.show()
pw.setWindowTitle('PHO pyqtgraph example: customPlot')

# r = pg.PolyLineROI([(0,0), (10, 10)])
# pw.addItem(r)

if __name__ == '__main__':
    pg.exec()
