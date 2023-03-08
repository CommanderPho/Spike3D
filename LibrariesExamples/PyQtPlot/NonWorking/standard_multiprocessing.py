# -*- coding: utf-8 -*-
"""
https://gist.github.com/Overdrivr/efea3d363556c0dcf4b6

This demo is similar to Pyqtgraph remote plotting examples (https://github.com/pyqtgraph/pyqtgraph/blob/develop/examples/RemoteGraphicsView.py)
Except that it does not use Pyqtgraph's buggy multiprocess module. Instead, it relies on standard python 3+ multiprocessing (https://docs.python.org/3.5/library/multiprocessing.html).
In this example, function f is executed in a second process, which liberates immediately the main process.
"""
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg
from multiprocessing import Process, Manager

def f(name):
    app2 = QtWidgets.QApplication([])

    win2 = pg.GraphicsWindow(title="Basic plotting examples")
    win2.resize(1000,600)
    win2.setWindowTitle('pyqtgraph example: Plotting')
    p2 = win2.addPlot(title="Updating plot")
    curve = p2.plot(pen='y')

    def updateInProc(curve):
        t = np.arange(0,3.0,0.01)
        s = np.sin(2 * np.pi * t + updateInProc.i)
        curve.setData(t,s)
        updateInProc.i += 0.1

    updateInProc.i = 0

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: updateInProc(curve))
    timer.start(50)

    QtWidgets.QApplication.instance().exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    manager = Manager()
    data = manager.list()

    p = Process(target=f, args=('bob',))
    p.start()
    input("Type any key to quit.")
    print("Waiting for graph window process to join...")
    p.join()
    print("Process joined successfully. C YA !")