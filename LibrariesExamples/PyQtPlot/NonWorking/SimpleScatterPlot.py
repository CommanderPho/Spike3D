# importing Qt widgets
from PyQt5.QtWidgets import *

# importing system
import sys

# importing numpy as np
import numpy as np

# importing pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph as pg
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from collections import namedtuple


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # setting title
        self.setWindowTitle("PyQtGraph")
        # setting geometry
        self.setGeometry(100, 100, 600, 500)
        # icon
        icon = QIcon("skin.png")
        # setting icon to the window
        self.setWindowIcon(icon)
        # calling method
        self.UiComponents()
        # showing all the widgets
        self.show()

    # method for components
    def UiComponents(self):
        # creating a widget object
        widget = QWidget()
        # text
        text = "Geeksforgeeks Scatter Plot Graph with different color spots"
        # creating a label
        label = QLabel(text)
        # setting minimum width
        label.setMinimumWidth(130)
        # making label do word wrap
        label.setWordWrap(True)
        # setting configuration options
        pg.setConfigOptions(antialias=True)
        # creating a plot window
        plt = pg.plot()
        # creating scatter plot item
        ## Set pxMode=False to allow spots to transform with the view
        scatter = pg.ScatterPlotItem(pxMode=False)
        # creating empty list for spots
        spots = []
        # creating loop for rows and column
        for i in range(10):
            for j in range(10):
                # creating  spot position which get updated after each iteration
                # of color which also get updated
                spot_dic = {
                    'pos': (1e-6 * i, 1e-6 * j), 'size': 1e-6,
                    'pen': {'color': 'w', 'width': 2},
                    'brush': pg.intColor(i * 10 + j, 100)}
                # adding spot_dic in the list of spots
                spots.append(spot_dic)
        # adding spots to the scatter plot
        scatter.addPoints(spots)
        # adding scatter plot to the plot window
        plt.addItem(scatter)

        # Createing a line plot:
        line = pg.PlotDataItem(antialias=True, pen=pg.mkPen(None))

        self.addItem(line)


        # Creating a grid layout
        layout = QGridLayout()
        # minimum width value of the label
        label.setMinimumWidth(130)
        # setting this layout to the widget
        widget.setLayout(layout)
        # adding label in the layout
        layout.addWidget(label, 1, 0)
        # plot window goes on right side, spanning 3 rows
        layout.addWidget(plt, 0, 1, 3, 1)
        # setting this widget as central widget of the main window
        self.setCentralWidget(widget)

# create pyqt5 app
App = QApplication(sys.argv)
# create the instance of our Window
window = Window()
# start the app
sys.exit(App.exec())
