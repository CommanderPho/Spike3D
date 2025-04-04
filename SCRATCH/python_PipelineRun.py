""" 

Can run via:

# change directory to PhoPy3DPositionAnalysis2021 repo
cd 'C:\\Users\\pho\\repos\\PhoPy3DPositionAnalysis2021'
# Call iPython:
ipython iPython_PipelineRun.ipy

jupyter qtconsole 
# change directory to PhoPy3DPositionAnalysis2021 repo
cd 'C:\\Users\\pho\\repos\\PhoPy3DPositionAnalysis2021'


pip install pyqt6 qt6-applications qt6-tools pyqt6-plugins pyqt6-sip pyqt6-qt6 pyqt6-sip

pip install PyQt6

"""
import sys
import importlib
from pathlib import Path
import numpy as np

# required to enable non-blocking interaction:
# %gui qt
# $env:QT_API="pyqt6"
# %gui qt5
# %gui qt6
# from PyQt5.Qt import QApplication
# # start qt event loop
# _instance = QApplication.instance()
# if not _instance:
#     _instance = QApplication([])
# app = _instance
import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# # Must be called before any figures are created:
# import matplotlib
# matplotlib.use('Qt5Agg')

from pyphoplacecellanalysis.GUI.PyQtPlot.Flowchart.pyqtplot_Flowchart import plot_flowchartWidget

# Pipeline:
# pipeline_flowchart_window, pipeline_flowchart_app = plot_flowchartWidget(title='PhoMainPipelineFlowchartApp'); pg.exec()

if __name__ == '__main__':
    # app = QtWidgets.QApplication(sys.argv)
    pipeline_flowchart_window, pipeline_flowchart_app = plot_flowchartWidget(title='PhoMainPipelineFlowchartApp')
    pipeline_flowchart_window.show()
    pg.exec()
    # app.exec()