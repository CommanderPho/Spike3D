import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

import numpy as np
import random

class fb_plot(QDialog):
    """ Simple test widget that allows the user to right-click on the plot for a custom context menu with options, and left-click drag over the plot to highlight a range. """
 
    def __init__(self, parent=None):
        super(fb_plot, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        ax = self.figure.add_subplot(111)

        # Capture button press
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('pick_event', self.pick)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # self.ax.grid(True)

        self.canvas.draw()
        self.span = SpanSelector(ax,
                                 self.spanzoom,
                                 'horizontal',
                                 useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'),
                                 button=1)

    def on_press(self, event):
        print(event)
        if event.button == 3:
            canvasSize = event.canvas.geometry()
            Qpoint_click = event.canvas.mapToGlobal(
                QtCore.QPoint(event.x, canvasSize.height()-event.y))

            if event.inaxes != None:
                self.fbMenu = QtWidgets.QMenu()
                self.fbMenu.addSection("Front/Back")
                self.fbMenu.addSeparator()
                self.fbMenu.addAction("Select All Curves")
                self.fbMenu.addAction("Remove All Curves")
                self.fbMenu.addSeparator()
                self.fbMenu.addAction("Legend")
                self.fbMenu.addAction("Cursor Legend")
                self.fbMenu.move(Qpoint_click)
                self.fbMenu.show()

            else:

                # Y Axis Front Area
                self.ylim_menu = QtWidgets.QMenu('&Limits')
                self.ylim_menu.addAction("Fixed")
                self.ylim_menu.addAction("Free")
                self.ylim_menu.addAction("Optimized")

                # y axis main menu
                self.yMenu = QtWidgets.QMenu('&Y Axis')
                self.yMenu.setTitle('sdfj')
                self.yMenu.addMenu(self.ylim_menu)
                self.yMenu.addSeparator()
                self.yMenu.addAction("Visable")
                self.yMenu.addAction("Options...")
                self.yMenu.move(Qpoint_click)
                self.yMenu.show()

                # Y Axis Back Area
                print(self.yMenu.title())

                # X Axis Area
                # Title Area
        else:
            pass

    def pick(self, event):
        print(event)

    def spanzoom(self, xmin, xmax):
        # indmin, indmax = np.searchsorted(x, (xmin, xmax))
        # indmax = min(len(x) - 1, indmax)
        # thisx = x[indmin:indmax]
        # thisy = y[indmin:indmax]
        # fb1.set_data(thisx, thisy)
        self.ax.set_xlim(xmin, xmax)

        #ax1.set_ylim(thisy.min(), thisy.max())
        self.canvas.draw()

    # def plot(self, xdata, ydata):
    #    self.ax.plot(xdata, ydata)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = fb_plot()

    # generate data
    np.random.seed(19680801)
    x = np.arange(0.0, 5.0, 0.01)
    y = np.sin(2*np.pi*x) + 0.5*np.random.randn(len(x))
    fb1, = main.plot(x, y, Label='data')

    main.show()
    sys.exit(app.exec_())