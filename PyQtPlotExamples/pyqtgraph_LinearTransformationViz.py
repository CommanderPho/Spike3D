import sys
import os
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QHBoxLayout, QSlider,QCheckBox 
from PyQt5 import QtCore,  QtGui

import qdarkstyle
import numpy as np
import time

class Graph(pg.GraphItem):
    def __init__(self, slot):
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.slot_update = slot
        
    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()
        
    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)
        
    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i,item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        
        if ev.isStart():
            self.setCursor(QtCore.Qt.BlankCursor)
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                self.setCursor(QtCore.Qt.ArrowCursor)
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]

            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            self.dragPoint = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                self.setCursor(QtCore.Qt.ArrowCursor)
                return

        ind = self.dragPoint.data()[0]
        new_pos =  ev.pos() + self.dragOffset
        snapped_x = self.snap(new_pos[0])
        snapped_y = self.snap(new_pos[1])
        self.data['pos'][ind] = pg.Point(snapped_x, snapped_y)
        self.updateGraph()
        self.slot_update(snapped_x, snapped_y, ind)
        ev.accept()

    def snap(self, num):
        rouned_num = round(num, 4)
        rounded = self.near(rouned_num)
        delta = round(abs(rounded - rouned_num), 4)
        if delta <= 0.04 :
            return rounded
        return num

    def near(self, num):
        rounded = round(num)
        sign = np.sign(num - rounded)
        mid = rounded + (sign * 0.5)
        return rounded if abs(num - rounded) <= 0.25 else mid

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.slidebar_val = 0

        self.v1_x = 1
        self.v1_y = 0
        self.v2_x = 0
        self.v2_y = 1
        self.input_x = 1
        self.input_y = 1
        self.output_x, self.output_y = self.computeTransform(self.input_x, self.input_y)

        pg.setConfigOptions(antialias = True)
        pg.setConfigOption('background', "#1B1B1B")
        pg.setConfigOption('foreground', "#727272")

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setVerticalSpacing(0)
        layout.setHorizontalSpacing(0)
        self.setStyleSheet("background : #1B1B1B; color : #727272")

####    Top-Left Panel ####
        panel_top_left = QWidget()
        panel_top_left.setStyleSheet("padding : 0px")
        panel_top_left.setMinimumHeight(100)
        panel_top_left.setMaximumHeight(100)
        layout_top_left = QGridLayout()
        layout_top_left.setContentsMargins(0, 0, 0, 0)

        panel_top_left.setLayout(layout_top_left)

        graph_top_left = pg.GraphicsLayoutWidget()
        layout_top_left.addWidget(graph_top_left)

        vb_top_left = graph_top_left.addViewBox()
        txt_top_left_open = pg.LabelItem("<span style = 'font-size : 75px; color : rgb(50, 65, 75);' >&#10647;</span>")
        txt_top_left_open.setPos(0, -27)
        txt_top_left_close = pg.LabelItem("<span style = 'font-size : 75px; color : rgb(50, 65, 75); '>&#10648;</span>")
        txt_top_left_close.setPos(150, -27)
        self.txt_top_left_matrix = pg.LabelItem("<span style = 'font-size : 20px;'><table ><tr ><td width=70 style = 'text-align:right; color : #ff0000; padding : 5px;'>%0.2f</td><td width=70 style = 'text-align:right; color : #00ff00;padding : 5px;'>%0.2f</td></tr><tr><td width=70 style = 'text-align:right; color : #ff0000;padding : 5px;'>%0.2f</td><td width=70 style = 'text-align:right; color : #00ff00;padding : 5px;'>%0.2f</td></tr></table></span>" % (self.v1_x , self.v2_x,self.v1_y, self.v2_y))
        self.txt_top_left_matrix.setPos(10, -3)

        txt_top_left_open.setParentItem(vb_top_left)
        txt_top_left_close.setParentItem(vb_top_left)
        self.txt_top_left_matrix.setParentItem(vb_top_left)

        ### Input Vector ###
        self.txt_top_left_input_open = pg.LabelItem("<span style = 'font-size : 75px; color : rgb(50, 65, 75);' >&#10647;</span>")
        self.txt_top_left_input_open.setPos(166, -27)
        self.txt_top_left_input_close = pg.LabelItem("<span style = 'font-size : 75px; color : rgb(50, 65, 75); '>&#10648;</span>")
        self.txt_top_left_input_close.setPos(248, -27)
        txt_top_left_equal = pg.LabelItem("<span style = 'font-size : 42px; color : rgb(50, 65, 75); '>=</span>")
        txt_top_left_equal.setPos(270, 4)

        self.txt_top_left_input = pg.LabelItem("<span style = 'font-size : 20px;'><table>    <tr><td width=50 style = 'text-align:right; color : rgb(190, 190, 190); padding : 5px;'>%0.2f</td></tr>    <tr><td width=50 style = 'text-align:right; color : rgb(190, 190, 190); padding : 5px;'>%0.2f</td></tr>     </table></span>" % (self.input_x, self.input_y))
        self.txt_top_left_input.setPos(188, -3)

        txt_top_left_res_open = pg.LabelItem("<span style = 'font-size : 75px; color : rgb(50, 65, 75);' >&#10647;</span>")
        txt_top_left_res_open.setPos(302, -27)

        txt_top_left_res_close = pg.LabelItem("<span style = 'font-size : 75px; color : rgb(50, 65, 75); '>&#10648;</span>")
        txt_top_left_res_close.setPos(383, -27)

        self.txt_top_left_output = pg.LabelItem("<span style = 'font-size : 20px;'><table>    <tr><td width=50 style = 'text-align:right; color : rgb(190, 190, 190); padding : 5px;'>%0.2f</td></tr>    <tr><td width=50 style = 'text-align:right; color : rgb(190, 190, 190); padding : 5px;'>%0.2f</td></tr>     </table></span>" % (self.output_x, self.output_y))
        self.txt_top_left_output.setPos(327, -3)

        self.txt_top_left_input_open.setParentItem(vb_top_left)
        self.txt_top_left_input_close.setParentItem(vb_top_left)
        txt_top_left_equal.setParentItem(vb_top_left)
        self.txt_top_left_input.setParentItem(vb_top_left)
        txt_top_left_res_open.setParentItem(vb_top_left)
        txt_top_left_res_close.setParentItem(vb_top_left)
        self.txt_top_left_output.setParentItem(vb_top_left)

        panel_checkboxes = QWidget()
        layout_checkboxes = QGridLayout()
        layout_checkboxes.setHorizontalSpacing(10)
        layout_checkboxes.setVerticalSpacing(0)

        panel_checkboxes.setLayout(layout_checkboxes)
        self.chk_eigen = QCheckBox("Eigen Vector")
        self.chk_eigen.setCheckState(2)
        self.chk_eigen.stateChanged.connect(self.chk_eigen_changed)
        self.chk_input = QCheckBox("Show In/Out")
        self.chk_input.setCheckState(2)
        self.chk_input.stateChanged.connect(self.chk_show_in_out)
        self.chk_circle = QCheckBox("Show Circle")
        self.chk_circle.setCheckState(2)
        self.chk_circle.stateChanged.connect(self.chk_show_circle)
        self.chk_det = QCheckBox("Show Det")
        self.chk_det.setCheckState(2)
        self.chk_det.stateChanged.connect(self.chk_det_changed)

        layout_checkboxes.addWidget(self.chk_eigen, 0, 1)
        layout_checkboxes.addWidget(self.chk_input, 1, 1)
        layout_checkboxes.addWidget(self.chk_circle, 0, 2)
        layout_checkboxes.addWidget(self.chk_det, 1, 2)
        layout_top_left.addWidget(panel_checkboxes, 0, 1)

#####   Top-Right Panel ####
        panel_top_right = QWidget()
        panel_top_right.setStyleSheet("padding : 0px")
        panel_top_right.setMinimumHeight(100)
        panel_top_right.setMaximumHeight(100)

        layout_top_right = QGridLayout()
        layout_top_right.setContentsMargins(0, 0, 0, 0)

        panel_top_right.setLayout(layout_top_right)
        graph_top_right = pg.GraphicsLayoutWidget()
        layout_top_right.addWidget(graph_top_right)

#####   Left Panel #############
        panelLeft = QWidget()
        panelLeft.setStyleSheet("padding : 0px")
        layoutPanelLeft = QGridLayout()
        layoutPanelLeft.setContentsMargins(0, 0, 0, 0)

        panelLeft.setLayout(layoutPanelLeft)
        graphLeft = pg.GraphicsLayoutWidget()
        layoutPanelLeft.addWidget(graphLeft)

        plotLeft = graphLeft.addPlot()
        plotLeft.hideButtons()
        axis_left_left = plotLeft.getAxis('left')
        axis_left_bottom = plotLeft.getAxis('bottom')
        axis_left_left.setTickSpacing(1, 1)
        axis_left_bottom.setTickSpacing(1, 1)

        axis_left_left.setStyle(showValues = False)
        axis_left_bottom.setStyle(showValues = False)

        plotLeft.showGrid(x = True, y = True)
        viewBoxLeft = plotLeft.getViewBox()
        
        plotLeft.setYRange(-4, 4)
        plotLeft.setXRange(-7, 7)

        graph_matrix_input = Graph(self.updateMatrix)
        #viewBoxLeft.addItem(graph_matrix_input)
        plotLeft.addItem(graph_matrix_input)
        viewBoxLeft.setAspectLocked()

        pos = np.array([
            [1, 0],
            [0, 1]], dtype=float)
            
        symbols = ['o', 'o']
        brushes = [pg.mkBrush(255, 0, 0), pg.mkBrush(0, 255, 0)]

        ## Update the graph
        graph_matrix_input.setData(pos=pos, size=16, symbolBrush = brushes, symbol=symbols )
        graph_matrix_input.setZValue(3)
        viewBoxLeft.disableAutoRange()

        self.arrow_head_x = pg.ArrowItem(tipAngle=30, baseAngle=25, headLen=25, tailLen=None, pen = None, brush=(255, 0, 0))
        self.arrow_line_x = pg.PlotDataItem(pen=pg.mkPen('r', width=2.5))
        self.arrow_line_x.setZValue(2)
        self.arrow_line_x.setZValue(1)
        self.arrow_line_x.setData([0,1],[0,0])

        self.arrow_head_y = pg.ArrowItem(tipAngle=30, baseAngle=25, headLen=25, tailLen=None, pen = None, brush=(0, 255, 0))
        self.arrow_head_y.setZValue(2)
        self.arrow_line_y = pg.PlotDataItem(pen=pg.mkPen('g', width=2.5))
        self.arrow_line_y.setZValue(1)
        self.arrow_line_y.setData([0,0],[0,1])

        self.arrow_head_x.resetTransform()
        self.arrow_head_y.resetTransform()
        self.arrow_head_x.rotate(-180)
        self.arrow_head_y.rotate(90)
        self.arrow_head_x.setPos(1, 0)
        self.arrow_head_y.setPos(0, 1)

        plotLeft.addItem(self.arrow_head_x)
        plotLeft.addItem(self.arrow_line_x)
        plotLeft.addItem(self.arrow_head_y)
        plotLeft.addItem(self.arrow_line_y)

        #### Determinant ######
        self.plot_determinant = plotLeft.plot([0, 1, 2, 0],[0, 1, -2, 0], fillLevel=-0.3)

        #### Input Point ######
        pos_input = np.array([[1, 1]], dtype=float)
        self.graph_input_point = Graph(self.updateInput)
        self.graph_input_point.setData(pos=pos_input, size=16, symbolBrush = (190, 190, 190), symbol='o' )
        self.graph_input_point.setZValue(3)

        viewBoxLeft.addItem(self.graph_input_point)

        ###   Output Vector ####
        self.arrow_head_output = pg.ArrowItem(tipAngle=30, baseAngle=25, headLen=25, tailLen=None, pen = None, brush=(190, 190, 190))
        self.arrow_head_output.setZValue(2)
        self.arrow_head_output.setPos(1, 1)
        self.arrow_line_output = pg.PlotDataItem(pen=pg.mkPen((190, 190, 190), width=2.5))
        self.arrow_line_output.setZValue(1)
        self.arrow_line_output.setData([0,1],[0,1])

        self.arrow_head_output.resetTransform()
        self.arrow_head_output.rotate(135)

        plotLeft.addItem(self.arrow_head_output)
        plotLeft.addItem(self.arrow_line_output)

        #####  Circle  ###########
        self.circle_x = np.linspace(-1, 1, 100)
        self.circle_y_upper = np.sqrt(1-(self.circle_x**2))
        self.circle_y_lower = -np.sqrt(1-(self.circle_x**2))
        self.plot_circle_upper = plotLeft.plot(self.circle_x, self.circle_y_upper, pen = pg.mkPen((231, 219, 96), width = 2.5))
        self.plot_circle_upper.setZValue(2)
        self.plot_circle_lower = plotLeft.plot(self.circle_x, self.circle_y_lower, pen = pg.mkPen((231, 219, 96), width = 2.5))
        self.plot_circle_lower.setZValue(2)

####    Slide Bar Left #######
        panel_slide_bar = QWidget()
        layout_slide_bar = QHBoxLayout()
        layout_slide_bar.setContentsMargins(6, 3, 4, 4)

        panel_slide_bar.setLayout(layout_slide_bar)

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        self.slider.setRange(0, 100)
        self.slider.setSingleStep(2)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.slider_val_changed)

        layout_slide_bar.addWidget(self.slider)

        self.btn_slide_run = QPushButton(">")
        self.btn_slide_run.setMinimumHeight(25)
        self.btn_slide_run.setMinimumWidth(30)
        self.btn_slide_run.tag = "paused"
        self.btn_slide_run.clicked.connect(self.btn_slide_run_clicked)

        layout_slide_bar.addWidget(self.btn_slide_run)
        layout.addWidget(panel_slide_bar, 2, 0)

####### Two Axes  ######
        self.x_axis_left = pg.InfiniteLine(angle = 0, pen = pg.mkPen((190, 190, 190), width=2))
        self.y_axis_left = pg.InfiniteLine(angle = 90, pen = pg.mkPen((190, 190, 190), width=2))

        self.lines_v = []
        self.lines_h = []

        self.grid_start = -7
        self.grid_end = -self.grid_start + 1
        self.gird_count_half = -self.grid_start

        for line in range(self.grid_start, self.grid_end, 1):
            line_v = pg.PlotDataItem(pen=pg.mkPen((23, 131, 149, 200), width=2.5))
            line_v.setData([line, line],[self.grid_start, self.grid_end - 1])
            plotLeft.addItem(line_v)
            self.lines_v.append(line_v)

            line_h = pg.PlotDataItem(pen=pg.mkPen((23, 131, 149, 200), width=2.5))
            line_h.setData([self.grid_start, self.grid_end - 1],[line, line])
            plotLeft.addItem(line_h)
            self.lines_h.append(line_h)

            if line == 0:
                line_v.setPen(pg.mkPen((190, 190, 190), width=2.5))
                line_h.setPen(pg.mkPen((190, 190, 190), width=2.5))

        ### Eigenvalues and EigenVectors
        self.eigen1 = pg.PlotDataItem(pen=pg.mkPen((233, 142, 8, 150), width=2.5))
        self.eigen2 = pg.PlotDataItem(pen=pg.mkPen((233, 142, 8, 150), width=2.5))

        plotLeft.addItem(self.eigen1)
        plotLeft.addItem(self.eigen2)

        self.eg_v1_x = 1
        self.eg_v1_y = 0
        self.eg_v2_x = 0
        self.eg_v2_y = 1

        self.arrow_head_eigens_v1 = []
        self.arrow_head_eigens_v1_neg = []
        self.arrow_head_eigens_v2 = []
        self.arrow_head_eigens_v2_neg = []

        for i in range(1, self.gird_count_half + 1):
            arrow_head = pg.ArrowItem(tipAngle=30, baseAngle=25, headLen=18, tailLen=None, pen = None, brush=(233, 142, 8, 200))
            plotLeft.addItem(arrow_head)
            self.arrow_head_eigens_v1.append(arrow_head)

            arrow_head = pg.ArrowItem(tipAngle=30, baseAngle=25, headLen=18, tailLen=None, pen = None, brush=(233, 142, 8, 200))
            plotLeft.addItem(arrow_head)
            self.arrow_head_eigens_v1_neg.append(arrow_head)

            arrow_head = pg.ArrowItem(tipAngle=30, baseAngle=25, headLen=18, tailLen=None, pen = None, brush=(233, 142, 8, 200))
            plotLeft.addItem(arrow_head)
            self.arrow_head_eigens_v2.append(arrow_head)

            arrow_head = pg.ArrowItem(tipAngle=30, baseAngle=25, headLen=18, tailLen=None, pen = None, brush=(233, 142, 8, 200))
            plotLeft.addItem(arrow_head)
            self.arrow_head_eigens_v2_neg.append(arrow_head)

######  Right Panel ##########
        panelRight = QWidget()
        panelRight.setStyleSheet("padding : 0px")
        layoutPanelRight = QGridLayout()
        layoutPanelRight.setContentsMargins(0, 0, 0, 0)
        panelRight.setLayout(layoutPanelRight)

        graphRight = pg.GraphicsLayoutWidget()
        layoutPanelRight.addWidget(graphRight)

        plotRight = graphRight.addPlot()
        viewBoxRight = plotRight.getViewBox()

        layout.addWidget(panel_top_left, 0, 0)
        layout.addWidget(panelLeft, 1, 0)

        self.setLayout(layout)

        #####  Setup States that trigger events ###
        self.chk_circle.setCheckState(0)
        self.chk_input.setCheckState(0)
        self.chk_eigen.setCheckState(0)
        self.chk_det.setCheckState(0)

        self.sliderThread = SliderRunner()
        self.sliderThread.update_signal.connect(self.increase_slider_val)

    def chk_det_changed(self):
        self.update_determinant()

    def chk_eigen_changed(self):
        self.updateEigenVectors()

    def update_determinant(self):
        if self.chk_det.checkState():
            v1_x, v1_y = self.computeTransform(1, 0)
            v2_x, v2_y = self.computeTransform(0, 1)
            det = (v1_x * v2_y) -(v2_x * v1_y) 

            if det > 0:
                self.plot_determinant.setBrush((255, 255, 0, 70))
            else:
                self.plot_determinant.setBrush((253, 0, 253, 70))
            self.plot_determinant.setData([0, v1_x, v1_x + v2_x, v2_x, 0],[0, v1_y, v1_y + v2_y, v2_y, 0])
        else:
            self.plot_determinant.setData([],[])

    def btn_slide_run_clicked(self):
        if self.btn_slide_run.tag == "paused" or self.slidebar_val == 1:
            if self.slidebar_val == 1:
                self.slider.setValue(0)
            
            self.btn_slide_run.setText("||")
            self.btn_slide_run.tag = "running"
            self.sliderThread.start()

        elif self.btn_slide_run.tag == "running":
            self.btn_slide_run.setText(">")
            self.btn_slide_run.tag = "paused"
            self.sliderThread.terminate()

    def increase_slider_val(self):
        slider_val = self.slider.value()
        if slider_val < 100:
            self.slider.setValue(slider_val + 1)
        else:
            print("thread ended..")
            self.btn_slide_run.setText(">")
            self.btn_slide_run.tag = "paused"
            self.sliderThread.terminate()

    def slider_val_changed(self, val):
        self.slidebar_val = val / 100

        #self.updateMatrix()
        self.updateVector1(self.v1_x, self.v1_y)
        self.updateVector2(self.v2_x, self.v2_y)

        self.updateGrid()
        self.updateCircle()
        self.updateOutput()

    def updateEigenVectors(self):

        if (self.eg_v1_x == 0 and self.eg_v1_y == 0) or not self.chk_eigen.checkState():
            self.eigen1.setData([0],[0])
            for i in range(self.gird_count_half):
                self.arrow_head_eigens_v1[i].hide()
                self.arrow_head_eigens_v1_neg[i].hide()
        else:
            self.eigen1.setData([self.eg_v1_x * 7,self.eg_v1_x * -7],[self.eg_v1_y * 7, self.eg_v1_y * -7])

            for i in range(1, self.gird_count_half + 1):
                arrow_head = self.arrow_head_eigens_v1[i - 1]
                arrow_head.setVisible(self.chk_eigen.checkState())
                arrow_head.resetTransform()
                arrow_head.rotate(self.getAngleToDraw(self.eg_v1_x * i, self.eg_v1_y * i))
                arrow_head.setPos(self.eg_v1_x * i, self.eg_v1_y * i)

                arrow_head = self.arrow_head_eigens_v1_neg[i - 1]
                arrow_head.setVisible(self.chk_eigen.checkState())
                arrow_head.resetTransform()
                arrow_head.rotate(self.getAngleToDraw(-self.eg_v1_x * i, -self.eg_v1_y * i))
                arrow_head.setPos(-self.eg_v1_x * i, -self.eg_v1_y * i)

        if (self.eg_v2_x == 0 and self.eg_v2_y == 0) or not self.chk_eigen.checkState():
            self.eigen2.setData([0],[0])
            for i in range(self.gird_count_half):
                self.arrow_head_eigens_v2[i].hide()
                self.arrow_head_eigens_v2_neg[i].hide()
        else:
            self.eigen2.setData([self.eg_v2_x * 7, self.eg_v2_x * -7],[self.eg_v2_y * 7, self.eg_v2_y * -7])

            for i in range(1, self.gird_count_half + 1):
                arrow_head = self.arrow_head_eigens_v2[i - 1]
                arrow_head.setVisible(self.chk_eigen.checkState())
                arrow_head.resetTransform()
                arrow_head.rotate(self.getAngleToDraw(self.eg_v2_x * i, self.eg_v2_y * i))
                arrow_head.setPos(self.eg_v2_x * i, self.eg_v2_y * i)

                arrow_head = self.arrow_head_eigens_v2_neg[i - 1]
                arrow_head.setVisible(self.chk_eigen.checkState())
                arrow_head.resetTransform()
                arrow_head.rotate(self.getAngleToDraw(-self.eg_v2_x * i, -self.eg_v2_y * i))
                arrow_head.setPos(-self.eg_v2_x * i, -self.eg_v2_y * i)

    def computeTransform(self, x, y, t = None):
        if t == None:
            v1_x = (1 * (1 - self.slidebar_val)) + (self.v1_x * self.slidebar_val)
            v1_y = (0 * (1 - self.slidebar_val)) + (self.v1_y * self.slidebar_val)

            v2_y = (1 * (1 - self.slidebar_val)) + (self.v2_y * self.slidebar_val)
            v2_x = (0 * (1 - self.slidebar_val)) + (self.v2_x * self.slidebar_val)
        else:
            v1_x = self.v1_x
            v1_y = self.v1_y
            v2_x = self.v2_x
            v2_y = self.v2_y
        return ((v1_x * x) + (v2_x * y), (v1_y * x) + (v2_y * y))

    def getAngleToDraw(self, x, y):
        if x == 0:
            if y >= 0:
                angle = 90
            else:
                angle = 270
        else:
            angle = (np.arctan(y / x) * 360) / (2 * np.pi)
        if x < 0 and y < 0:
            angle += 180
        elif x < 0 and y >= 0:
            angle = 180 + angle
        elif x > 0 and y < 0:
            angle = 360 + angle
        return  180 - angle

    def updateInput(self, x, y, ind = -1):
        self.input_x = x
        self.input_y = y
        self.txt_top_left_input.setText("<span style = 'font-size : 20px;'><table>    <tr><td width=50 style = 'text-align:right; color : rgb(190, 190, 190); padding : 5px;'>%0.2f</td></tr>    <tr><td width=50 style = 'text-align:right; color : rgb(190, 190, 190); padding : 5px;'>%0.2f</td></tr>     </table></span>" % (self.input_x, self.input_y))
        self.updateOutput()

    def updateOutput(self):
        self.output_x, self.output_y = self.computeTransform(self.input_x, self.input_y, t = 1)
        self.txt_top_left_output.setText("<span style = 'font-size : 20px;'><table>    <tr><td width=50 style = 'text-align:right; color : rgb(190, 190, 190); padding : 5px;'>%0.2f</td></tr>    <tr><td width=50 style = 'text-align:right; color : rgb(190, 190, 190); padding : 5px;'>%0.2f</td></tr>     </table></span>" % (self.output_x, self.output_y))

        self.output_x, self.output_y = self.computeTransform(self.input_x, self.input_y)

        if self.output_x == 0 and self.output_y == 0:
            self.arrow_line_output.setData([0], [0])
            self.arrow_head_output.hide()
        else:
            angle = self.getAngleToDraw(self.output_x, self.output_y)
            self.arrow_line_output.setData([0,self.output_x],[0,self.output_y])
            self.arrow_head_output.setPos(self.output_x, self.output_y)
            self.arrow_head_output.setVisible(self.chk_input.checkState())
            self.arrow_head_output.resetTransform()
            self.arrow_head_output.rotate(angle)

    def chk_show_circle(self):
        self.plot_circle_upper.setVisible(self.chk_circle.checkState())
        self.plot_circle_lower.setVisible(self.chk_circle.checkState())
        self.updateCircle()

    def chk_show_in_out(self):
        print("chk_input....")
        self.arrow_line_output.setVisible(self.chk_input.checkState())
        self.graph_input_point.setVisible(self.chk_input.checkState())
        self.updateOutput()

    def updateCircle(self):
        transformed_x, transformed_y = self.computeTransform(self.circle_x, self.circle_y_upper)
        self.plot_circle_upper.setData(transformed_x, transformed_y)
        transformed_x, transformed_y = self.computeTransform(self.circle_x, self.circle_y_lower)
        self.plot_circle_lower.setData(transformed_x, transformed_y)

    def updateGrid(self):
        for index, (line_v, line_h) in enumerate(zip(self.lines_v, self.lines_h)):
            x1, y1 = self.computeTransform(index - self.gird_count_half, -self.gird_count_half)
            x2, y2 = self.computeTransform(index - self.gird_count_half, self.gird_count_half)
            if x1 == x2 and y1 == y2:
                line_v.setData([0], [0])
            else:
                line_v.setData([x1, x2], [y1, y2])

            x1, y1 = self.computeTransform(-self.gird_count_half, index - self.gird_count_half)
            x2, y2 = self.computeTransform(self.gird_count_half, index - self.gird_count_half)
            if x1 == x2 and y1 == y2:
                line_h.setData([0], [0])
            else:
                line_h.setData([x1, x2], [y1, y2])
        self.update_determinant()

    def updateVector1(self, x, y):

        x = (1 * (1 - self.slidebar_val)) + (x * self.slidebar_val)
        y = (0 * (1 - self.slidebar_val)) + (y * self.slidebar_val)

        if x == 0 and y == 0:
            self.arrow_line_x.setData([0], [0])
            self.arrow_head_x.hide()
        else:
            angle = self.getAngleToDraw(x, y)
            self.arrow_line_x.setData([0,x],[0,y])
            self.arrow_head_x.setPos(x, y)
            self.arrow_head_x.show()
            self.arrow_head_x.resetTransform()
            self.arrow_head_x.rotate(angle)

    def updateVector2(self, x, y):

        y = (1 * (1 - self.slidebar_val)) + (y * self.slidebar_val)
        x = (0 * (1 - self.slidebar_val)) + (x * self.slidebar_val)

        if x == 0 and y == 0:
            self.arrow_line_y.setData([0], [0])
            self.arrow_head_y.hide()
        else:
            angle = self.getAngleToDraw(x, y)
            self.arrow_line_y.setData([0,x],[0,y])
            self.arrow_head_y.setPos(x, y)
            self.arrow_head_y.show()
            self.arrow_head_y.resetTransform()
            self.arrow_head_y.rotate(angle)

    def updateMatrix(self, x, y, pt_no = -1):

        if pt_no == 0:
            self.updateVector1(x, y)
            self.v1_x = x
            self.v1_y = y
        elif pt_no == 1:
            self.updateVector2(x, y)
            self.v2_x = x
            self.v2_y = y
        if pt_no != -1:
            self.txt_top_left_matrix.setText("<span style = 'font-size : 20px;'><table ><tr ><td width=70 style = 'text-align:right; color : #ff0000; padding : 5px;'>%0.2f</td><td width=70 style = 'text-align:right; color : #00ff00;padding : 5px;'>%0.2f</td></tr><tr><td width=70 style = 'text-align:right; color : #ff0000;padding : 5px;'>%0.2f</td><td width=70 style = 'text-align:right; color : #00ff00;padding : 5px;'>%0.2f</td></tr></table></span>" % (self.v1_x , self.v2_x,self.v1_y, self.v2_y))

        values, vectors = np.linalg.eig([[self.v1_x, self.v2_x],[self.v1_y, self.v2_y]])
        if type(values[0]) == np.complex128 or type(values[1]) == np.complex128:
            self.eg_v1_x = 0
            self.eg_v1_y = 0
            self.eg_v2_x = 0
            self.eg_v2_y = 0
        else:
            self.eg_v1_x = vectors[0, 0] * values[0]
            self.eg_v1_y = vectors[1, 0] * values[0]
            self.eg_v2_x = vectors[0, 1] * values[1]
            self.eg_v2_y = vectors[1, 1] * values[1]

        self.updateGrid()
        self.updateCircle()
        self.updateOutput()
        self.updateEigenVectors()

class SliderRunner(QtCore.QThread):
        update_signal = QtCore.pyqtSignal()

        def __init__(self):
            QtCore.QThread.__init__(self)

        def run(self):
            while(True):
                self.update_signal.emit()
                time.sleep(.03)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    app.exec_()