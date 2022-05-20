import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5 import QtGui
import pyphoplacecellanalysis.External.pyqtgraph as pg
import numpy as np

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        layout = QGridLayout()
        self.setLayout(layout)

        pg.setConfigOptions(antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        graph = pg.GraphicsLayoutWidget()

        layout.addWidget(graph)
        layout.setContentsMargins(0, 0, 0, 0)

        

        self.plot = graph.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideButtons()

        x = np.mgrid[-10: 10: .1]
        y = np.sin(x)
        self.plot.plot(x, y, pen = pg.mkPen((211, 120, 117, 255), width = 3))
        y = np.cos(x)
        self.plot.plot(x, y, pen = pg.mkPen((118, 160, 203, 255), width = 3))
        y = 2 * np.cos(x)
        self.plot.plot(x, y, pen = pg.mkPen((121, 182, 131, 255), width = 3))
        y = 2 * np.sin(x)
        self.plot.plot(x, y, pen = pg.mkPen((153, 134, 199, 255), width = 3))
        y = 2 * np.sin(x) + 4 * np.cos(x)
        pp = self.plot.plot(x, y, pen = pg.mkPen((78, 78, 78, 255), width = 3))
        print(type(pp))
        pp.name = "klsfkds"

        self.plot.addLegend()

        xAxis = pg.InfiniteLine((0, 0), angle = 0, pen = pg.mkPen((125, 125, 120), width = 2))
        xAxis.setZValue(-1)
        self.plot.addItem(xAxis)
        
        self.yAxis = pg.InfiniteLine((0, 0), angle = 90, pen = pg.mkPen((125, 125, 120), width = 2))
        self.yAxis.setMovable(True)
        self.yAxis.setHoverPen(pg.mkPen((0, 255, 0), width = 3))
        self.yAxis.setZValue(-1)
        self.plot.addItem(self.yAxis)

        #self.pp = pg.SignalProxy(self.plot.sigRangeChanged,  slot = self.handleRangeChanged)

        self.x_axis = self.plot.getAxis("bottom")
        self.x_axis.setGrid(100)

        self.y_axis = self.plot.getAxis("left")
        self.y_axis.setGrid(100)
        self.y_axis.linked_axis = self.x_axis
        
        self.plot.setContentsMargins(-84, -45, -48, -71)
        
        self.ticks = []
        self.lines = []

    def handleRangeChanged(self):

        return

        for tick in self.ticks:
            self.plot.removeItem(tick)
        self.ticks.clear()
        for line in self.lines:
            self.plot.removeItem(line)
        self.lines.clear()

        #printMethods(self.plot)

        range_x = self.plot.viewRange()[0]
        range_y = self.plot.viewRange()[1]

        print("range_x", range_x)
        print("range_y", range_y)



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
    app = QApplication(sys.argv)

    win = MainWindow()
    win.show()
    app.exec_()

