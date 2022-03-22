
import sys
import qdarkstyle
import os
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout,QHBoxLayout,QPushButton
import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import Qt


""" 
Allows the user to interactively add/delete new scatterplot points by clicking on the axes.

"""
class CustomPoint(pg.PlotDataItem):
    def __init__(self, mainWindow, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mainWindow = mainWindow
        self.sigClicked.connect(lambda: CustomPoint.pointClicked(self))

    @staticmethod
    def pointClicked(obj):
        if obj.mainWindow.btnDelete.tag == "plotMode":
            return
        obj.mainWindow.plot.removeItem(obj)
        x, y = obj.getData()
        obj.mainWindow.removePoint([x[0], y[0]])

class GraphView(QWidget):
    def __init__(self):
        super().__init__()

        self.data = np.empty((0, 2), float)

        layout = QGridLayout()
        layout.setVerticalSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        panelToobar = QWidget()
        layoutToolbar = QHBoxLayout()
        panelToobar.setLayout(layoutToolbar)

        self.btnDelete = QPushButton("Delete")
        layoutToolbar.addWidget(self.btnDelete)
        layoutToolbar.setContentsMargins(0, 0, 0, 0)

        layoutToolbar.addStretch()
        self.btnDelete.clicked.connect(self.btnDeleteClicked)
        self.btnDelete.tag = "plotMode"

        layout.addWidget(panelToobar, 0, 0)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        graphicLayout = pg.GraphicsLayoutWidget()
        graphicLayout.setStyleSheet("padding : 0px")
        layout.addWidget(graphicLayout, 1, 0)
        
        self.plot = graphicLayout.addPlot()
        self.plot.setCursor(Qt.CrossCursor)
        self.plot.disableAutoRange()
        self.plot.scene().setClickRadius(4)

        self.click_signal = pg.SignalProxy(self.plot.scene().sigMouseClicked, rateLimit = 60, slot = self.plotMouseClicked)

    def btnDeleteClicked(self):
        if self.btnDelete.tag == "deleteMode":
            self.btnDelete.tag = "plotMode"
            self.btnDelete.setText("Delete")
            self.plot.setCursor(Qt.CrossCursor)
        else:
            self.btnDelete.tag = "deleteMode"
            self.btnDelete.setText("Plot")
            self.plot.setCursor(Qt.PointingHandCursor)
        #QApplication.restoreOverrideCursor()

    def plotMouseClicked(self, evt):
        if self.btnDelete.tag == "deleteMode":
            return

        event = evt[0]

        x = self.plot.getViewBox().mapSceneToView(event.scenePos()).x()
        y = self.plot.getViewBox().mapSceneToView(event.scenePos()).y()

        if event.double() or event.currentItem == None or event.buttons().__int__() != 1:
            return

        point = CustomPoint(self, [x], [y], symbolBrush= (0, 255, 0), symbolSize=16)
        self.plot.addItem(point)
        self.data = np.append(self.data, [[x, y]], axis = 0)

    def removePoint(self, point):
        self.data = np.delete(self.data, point, axis = 0)

    def plotMouseMoved(self, evt):
        pos = evt[0]
        mousePoint = self.plot.vb.mapSceneToView(pos)
        points = self.scatter.pointsAt(mousePoint)
        if len(points) == 0:
            return

def printMethods(obj):
    list = dir(obj)
    for i in range(int((len(list) / 5)) + 1):
        start = i * 5
        end = ((i + 1) * 5) - 1
        if end > len(list):
            end = len(list) - 1
        
        for ele in list[start: end + 1]:
            print(ele + ",  ", end = "")
        print()

if __name__ == "__main__":

    os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_from_environment(is_pyqtgraph=True))

    win = GraphView()
    win.show()

    app.exec_()



