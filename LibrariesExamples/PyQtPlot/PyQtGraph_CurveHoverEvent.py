# import pyqtgraph as pg
# from pyqtgraph import QtCore, QtGui
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets


""" 
`HoverableCurveItem` from https://stackoverflow.com/questions/68847398/hover-tool-for-plots-in-pyqtgraph

# How to show cursor position in pyqtgrapth embedded in pyqt5:
    https://www.pythonguis.com/faq/how-to-show-cursor-position-in-pyqtgrapth-embedded-in-pyqt5/


https://stackoverflow.com/questions/46166205/display-coordinates-in-pyqtgraph

self.plot.setMouseTracking(True)
self.plot.scene().sigMouseMoved.connect(self.mouseMoved)

def mouseMoved(self, evt):
        pos = evt
        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.plotItem.vb.mapSceneToView(pos)
            self.mousecoordinatesdisplay.setText("<span style='font-size: 15pt'>X=%0.1f, <span style='color: black'>Y=%0.1f</span>" % (mousePoint.x(),mousePoint.y()))
        self.plot.plotItem.vLine.setPos(mousePoint.x())
        self.plot.plotItem.hLine.setPos(mousePoint.y()

def MouseMoved(self, evt):
    pos = evt
    if self.graphWidget.sceneBoundingRect().contains(pos):
        mousePoint = self.graphWidget.plotItem.vb.mapSceneToView(pos)
        x = float("{0:.3f}".format(mousePoint.x()))
        y = float("{0:.3f}".format(mousePoint.y()))
        self.xylabel.setText(f"Cursor Position: {x, y}")


https://stackoverflow.com/questions/63184407/trying-to-get-cursor-with-coordinate-display-within-a-pyqtgraph-plotwidget-in-py?rq=1
def mouseMoved(self, evt):
        pos = evt
        if self.plotWidget.sceneBoundingRect().contains(pos):
            mousePoint = self.plotWidget.plotItem.vb.mapSceneToView(pos)
            mx = abs(np.ones(len(self.plotx))*mousePoint.x() - self.plotx)
            index = mx.argmin()
            if index >= 0 and index < len(self.plotx):
                self.cursorlabel.setHtml(
                    "<span style='font-size: 12pt'>x={:0.1f}, \
                     <span style='color: red'>y={:0.1f}</span>".format(
                     self.plotx[index], self.ploty[index])
                     )
            self.vLine.setPos(self.plotx[index])
            self.hLine.setPos(self.ploty[index])


# PyQtGraph – Getting Points Object at Specific Position in Scatter Plot Graph:
    https://www.geeksforgeeks.org/pyqtgraph-getting-points-object-at-specific-position-in-scatter-plot-graph/
    https://www.geeksforgeeks.org/pyqtgraph-getting-points-object-in-scatter-plot-graph/

    ```python
        # getting points object at specific location of the scatter plot
        value = scatter.pointsAt(QPoint(1, 5))
        # setting text to the value
        label.setText("Points at 1, 5: " + str(value))

        # setting tool tip to the scatter plot
        scatter.setToolTip("This is tip")

        # getting tool tip of scatter plot
        value = scatter.toolTip()
        # setting text to the value
        label.setText("Tool tip : " + str(value))

    ```


    https://www.pythonguis.com/faq/how-to-show-cursor-position-in-pyqtgrapth-embedded-in-pyqt5/


"""

class HoverableCurveItem(pg.PlotCurveItem):
    """ 
    From https://stackoverflow.com/questions/68847398/hover-tool-for-plots-in-pyqtgraph

    """
    sigCurveHovered = QtCore.Signal(object, object)
    sigCurveNotHovered = QtCore.Signal(object, object)

    def __init__(self, hoverable=True, *args, **kwargs):
        super(HoverableCurveItem, self).__init__(*args, **kwargs)
        self.hoverable = hoverable
        self.setAcceptHoverEvents(True)

    def hoverEvent(self, ev):
        if self.hoverable:
            if self.mouseShape().contains(ev.pos()):
                self.sigCurveHovered.emit(self, ev)
            else:
                self.sigCurveNotHovered.emit(self, ev)


class MainWindow(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.view = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.view)
        self.makeplot()

    def makeplot(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        plot = self.view.addPlot()
        self.plotitem = HoverableCurveItem(x, y, pen=pg.mkPen('w', width=10))
        self.plotitem.setClickable(True, width=10)
        self.plotitem.sigCurveHovered.connect(self.hovered)
        self.plotitem.sigCurveNotHovered.connect(self.leaveHovered)
        plot.addItem(self.plotitem)

    def hovered(self, curve_item, hover_event):
        print(f"cursor entered curve: ev: {hover_event}")
        self.plotitem.setPen(pg.mkPen('b', width=10))

    def leaveHovered(self, ev):
        self.plotitem.setPen(pg.mkPen('w', width=10))


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())