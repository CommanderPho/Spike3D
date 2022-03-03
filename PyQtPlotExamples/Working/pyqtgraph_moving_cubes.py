# -*- coding: utf-8 -*-

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys


class Visualizer(object):
    def __init__(self):
        self.cubes = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.setCameraPosition(distance=35, elevation=20, azimuth=180)
        self.w.setWindowTitle('pyqtgraph: moving cubees')
        self.w.setGeometry(0, 110, 960, 540)
        self.w.show()
        # add axis
        ax = gl.GLAxisItem()
        ax.setSize(1, 1, 1)
        self.w.addItem(ax)
        # create the background grids
        gx = gl.GLGridItem()
        gx.setSize(200, 16, 10)
        self.w.addItem(gx)

        self.n = 10
        self.y = np.linspace(-10, 10, self.n)
        self.x = np.linspace(-10, 100, self.n)
        self.z = np.linspace(0,0,self.n)
        self.length = 4
        self.width = 2
        self.height = 2
        self.faces = np.array([[1, 0, 7], [1, 3, 7],
                              [1, 2, 4], [1, 0, 4],
                              [1, 2, 6], [1, 3, 6],
                              [0, 4, 5], [0, 7, 5],
                              [2, 4, 5], [2, 6, 5],
                              [3, 6, 5], [3, 7, 5]])
        self.colors = np.array([[0.7, 0, 0, 0] for i in range(12)])

        for i in range(self.n):
            vertexes = self.vertices(self.x[i], self.y[i], self.z[i],
                                   self.length, self.width, self.height)
            self.cubes[i] = gl.GLMeshItem(vertexes=np.array(vertexes),
                                          faces=self.faces,
                                          faceColors=self.colors,
                                          drawEdges=True,
                                          edgeColor=(1, 0, 0, 1),
                                          antialias=True)
            self.w.addItem(self.cubes[i])

    def vertices(self, x, y, z, length, width, height):
        vertex = [[x + length, y, z],  # 0
                  [x, y, z],  # 1
                  [x, y + width, z],  # 2
                  [x, y, z + height],  # 3
                  [x + length, y + width, z],  # 4
                  [x + length, y + width, z + height],  # 5
                  [x, y + width, z + height],  # 6
                  [x + length, y, z + height]]  # 7

        return vertex

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, vertexes):
        self.cubes[name].setMeshData(vertexes=vertexes,
                                     faces=self.faces,
                                     faceColors=self.colors,
                                     drawEdges=True, edgeColor=(0.7, 0, 0, 0))

    def update(self):
        for i in range(self.n):
            vertexes = self.vertices(self.x[i], self.y[i], self.z[i],
                                     self.length, self.width, self.height)
            self.set_plotdata(
                name=i, vertexes=np.array(vertexes))
            #self.cubes[i].translate(self.x[i], 0, 0)
            self.x[i] -= 0.2


    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    v = Visualizer()
    v.animation()