import sys
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore  # (the example applies equally well to PySide2)
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg

## Always start by initializing Qt (only once per application)
app = QtGui.QApplication([])

## Define a top-level widget to hold everything
w = QtGui.QWidget()

## Create some widgets to be placed inside
btn = QtGui.QPushButton('press me')
text = QtGui.QLineEdit('enter text')
listw = QtGui.QListWidget()
plot = pg.PlotWidget()

""" Plots a single diagonal line on the plot """
test_x = np.arange(9)
test_y = 2*test_x + 4
line_plot = plot.plot(x=test_x, y=test_y)

""" Plots colored scatter points in ascending color order along the line """
brush_list = [pg.mkColor(c) for c in "rgbcmykwrg"]
for i in range(len(test_x)):
    s = pg.ScatterPlotItem([test_x[i]], [test_y[i]], size=10, pen=pg.mkPen(None))  # brush=pg.mkBrush(255, 255, 255, 120))
    s.setBrush(QtGui.QBrush(QtGui.QColor(QtCore.qrand() % 256, QtCore.qrand() % 256, QtCore.qrand() % 256)))
    plot.addItem(s)



## Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)

## Add widgets to the layout in their proper positions
layout.addWidget(btn, 0, 0)   # button goes in upper-left
layout.addWidget(text, 1, 0)   # text edit goes in middle-left
layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows

## Display the widget as a new window
w.show()

## Start the Qt event loop
# app.exec_()
sys.exit(QtGui.QApplication.exec_())

