# -*- coding: utf-8 -*-
"""
    Animated 3D sinc function
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys


class Visualizer(object):
    def __init__(self):
        self.traces = []
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 40
        self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.w.setGeometry(0, 110, 800, 600)
        self.w.show()

        # create the background grids
        # gx = gl.GLGridItem()
        # gx.rotate(90, 0, 1, 0)
        # gx.translate(-10, 0, 0)
        # self.w.addItem(gx)
        # gy = gl.GLGridItem()
        # gy.rotate(90, 1, 0, 0)
        # gy.translate(0, -10, 0)
        # self.w.addItem(gy)
        # gz = gl.GLGridItem()
        # gz.translate(0, 0, -10)
        # self.w.addItem(gz)

        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 0)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        self.w.addItem(gz)
        
        # self.n = 50
        self.N = 128
        # self.y = np.linspace(-10, 10, self.N + 2)
        # self.x = np.linspace(-10, 10, self.N)
        # self.phase = 0

        t_end = 8
        t = np.linspace(0, t_end, self.N)
        rate = self.N / t_end
        c0 = 0.4 * np.cos(2*np.pi * 5 * t)
        s0 = 0.3 * np.sin(2*np.pi * 3 * t)
        signal = c0 + s0
        spectrum = np.fft.rfft(signal)
        f = np.fft.rfftfreq(self.N, d=1/rate)
        cos_amp       =  np.real(spectrum)       / (self.N/2)
        sin_amp       = -np.imag(spectrum)       / (self.N/2)
        cos_amp[0]    =  np.real(spectrum[0])    / self.N
        cos_amp[self.N//2] =  np.real(spectrum[self.N//2]) / self.N
        cos_amp = cos_amp
        sin_amp = sin_amp
            

        for i in range(self.N + 2):
            x = np.empty((3, self.N))
            x[0] = t
            x[1] = np.full_like(t, f[i//2])

            if i % 2 == 0:
                x[2] = cos_amp[i//2] * np.cos(2 * np.pi * x[0] * x[1])
            else:
                x[2] = sin_amp[i//2] * np.sin(2 * np.pi * x[0] * x[1])

            self.traces.append(gl.GLLinePlotItem(
                pos=x.T, 
                color=pg.glColor((i, self.N * 1.3)), 
                width=3, 
                antialias=True
            ))
            self.w.addItem(self.traces[i])

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec()

    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)

    def update(self):
        t_end = 8
        t = np.linspace(0, t_end, self.N)
        rate = self.N / t_end
        c0 = 0.4 * np.cos(2*np.pi * 5 * t)
        s0 = 0.3 * np.sin(2*np.pi * 3 * t)
        signal = c0 + s0
        spectrum = np.fft.rfft(signal)
        f = np.fft.rfftfreq(self.N, d=1/rate)
        cos_amp       =  np.real(spectrum)       / (self.N/2)
        sin_amp       = -np.imag(spectrum)       / (self.N/2)
        cos_amp[0]    =  np.real(spectrum[0])    / self.N
        cos_amp[self.N//2] =  np.real(spectrum[self.N//2]) / self.N
        cos_amp = cos_amp
        sin_amp = sin_amp

        for i in range(self.N + 2):

            # N = 7
            # i = 2
            x = np.empty((3, self.N))
            x[0] = t
            x[1] = np.full_like(t, f[i//2])

            if i % 2 == 0:
                x[2] = cos_amp[i//2] * np.cos(2 * np.pi * x[0] * x[1])
            else:
                x[2] = sin_amp[i//2] * np.sin(2 * np.pi * x[0] * x[1])

            # plt.grid()
            # plt.plot(t, x[2])
            # print(f[i])
            # yi = np.array([self.y[i]] * self.m)
            # d = np.sqrt(self.x ** 2 + yi ** 2)
            # z = 10 * np.cos(d + self.phase) / (d + 1)
            # pts = np.vstack([self.x, yi, z]).transpose()
            # print(pts.shape)
            self.set_plotdata(
                name=i, 
                points=x.T,
                color=pg.glColor((i, self.N * 1.3)),
                width=3
            )
            # self.phase -= .003

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        # timer.start(20)
        timer.start(0)
        self.start()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    v = Visualizer()
    v.animation()
