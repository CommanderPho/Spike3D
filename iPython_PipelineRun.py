""" 

jupyter qtconsole 
# change directory to PhoPy3DPositionAnalysis2021 repo
cd 'C:\\Users\\pho\\repos\\PhoPy3DPositionAnalysis2021'

"""
%load_ext autoreload
%autoreload 2
import sys
import importlib
from pathlib import Path

import numpy as np

# required to enable non-blocking interaction:
# %gui qt
# $env:QT_API="pyqt6"
%gui qt5
# from PyQt5.Qt import QApplication
# # start qt event loop
# _instance = QApplication.instance()
# if not _instance:
#     _instance = QApplication([])
# app = _instance
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui



from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.pyqtplot_Flowchart import plot_flowchartWidget
pipeline_flowchart_window, pipeline_flowchart_app = plot_flowchartWidget(title='PhoMainPipelineFlowchartApp'); pg.exec()

