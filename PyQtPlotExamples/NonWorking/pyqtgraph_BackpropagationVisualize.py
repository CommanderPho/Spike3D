from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout,QDialog, QVBoxLayout,QHBoxLayout,QGroupBox,QPushButton,QSizePolicy, QLabel,QLineEdit 
from PyQt5 import QtCore, QtGui

import numpy as np
import sys

import pyqtgraph as pg

import NnNetworkGrapher
from tvtk.api import tvtk

import vtk
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

import time

################################################################################
#The actual visualization
class Visualization(HasTraits):

	scene = Instance(MlabSceneModel, ())
	scene.background = (0,0,0)
	view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),height=250, width=500, show_label=False),resizable=True)
	
	@on_trait_change('scene.activated')
	def update_plot(self):
		self.scene.mlab.figure(1, size = (500, 500), bgcolor  = (0,0,0))
		
		cc = np.power(np.linspace(0, 10, 13), 4)
		mm = np.max(cc)
		cc = (cc * (max / mm)).tolist() 
		
		axis_extent = (MIN_X_AXIS, MAX_X_AXIS, MIN_Y_AXIS, MAX_Y_AXIS, 0, max)
		
		self.surf_cat2 = self.scene.mlab.surf(X_W111, Y_W211, costFunction  , colormap='PiYG')

		
		self.surf_cat1 = self.scene.mlab.contour_surf(X_W111, Y_W211,C,  contours = cc, color = (172/255, 172/255, 172/255), line_width = 1.2)
		
		self.inititalPoint = self.scene.mlab.points3d([x_0[0]], [x_0[1]], [initial_cost], color = (1, 1, 0), scale_factor = 1, mode = 'point')
		self.inititalPoint.actor.property.point_size = 8
		self.points = self.scene.mlab.points3d([], [], [], 	color = (1, 1, 0),scale_factor = 1,  mode = 'point', line_width = 2)
		self.leadingPoint = self.scene.mlab.points3d([], [], [], color = (0, 1, 0), scale_factor = 1, mode = 'point')
		self.leadingPoint.actor.property.point_size = 7
		
		n_mer, n_long = 6, 11
		dphi = np.pi / 1000.0
		phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
		mu = phi * n_mer
		xxx = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
		yyy = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
		zzz = np.sin(n_long * mu / n_mer) * 0.5
		
		list_of_edges = []
		self.points.actor.property.point_size = 5
		self.points.mlab_source.dataset.lines = np.array(list_of_edges)
		
		tube = self.scene.mlab.pipeline.streamline(self.points)
		self.scene.mlab.pipeline.surface(tube, color=(1,1,1))

		
		####
		self.axis = self.scene.mlab.axes(self.surf_cat1, color=(.9, .9, .9), line_width = 1	,
            ranges=(MIN_X_AXIS, MAX_X_AXIS, MIN_Y_AXIS, MAX_Y_AXIS, 0, max), xlabel='w111', ylabel='w211',zlabel='',
            x_axis_visibility=True, z_axis_visibility=True, extent = axis_extent)

	def cat(self,x, y):
		return np.sin(x) + np.cos( y)
		
#### Algorithm.....	 ###############################
	
x = np.linspace(-2, 2, 41)

w111 = 10; b11 = 5
w121 = 2; b12 = -4

w211 = 10; w212 = 6; b21 = -1

def nominalFunction(xx, W111, W211):
	
	N11 = neuron([W111], [xx], b11)
	
	N12 = neuron([w121], [xx], b12)
	N21 = neuron([W211, w212], [N11, N12], b21)
	
	return N21

def neuron(ws, xs, b):
	temp = 0
	for i in range(len(ws)):
		temp += (ws[i] * xs[i])
	temp += b
	return 1 / (1 + np.exp(-temp))
	
y = nominalFunction(x, w111, w211)

def costFunction(W111, W211):
	temp = np.zeros((len(W111), len(W111)))
	for i in range(len(x)):
		Y = np.full((len(W111), len(W111)), y[i])
		temp += np.power(Y - nominalFunction(x[i], W111, W211), 2)
		
	return temp
	
def gradientFunction(X):
	temp_0 = 0
	temp_1 = 0
	
	for i in range(len(x)):
		nf = nominalFunction(x[i], X[0], X[1])
		neuron11 = neuron([X[0]], [x[i]], b11)
		temp_0 += 2 * (y[i] - nf) * -( nf * (1 - nf) * X[1] * neuron11 * (1 - neuron11) * x[i])
		temp_1 += 2 * (y[i] - nf) * -( nf * (1 - nf) * neuron11) 

	return np.array((temp_0, temp_1))


MIN_X_AXIS = -5
MAX_X_AXIS = 15

MIN_Y_AXIS = -5
MAX_Y_AXIS = 15

X_W111, Y_W211 = np.mgrid[MIN_X_AXIS:MAX_X_AXIS:0.2, MIN_Y_AXIS:MAX_Y_AXIS:0.2]

C = costFunction(X_W111, Y_W211)

min = 0
max = 0

def findMinMax():

	I_Min = 0
	J_Min = 0

	I_Max = 0
	J_Max = 0
	global min, max, C
	min = C[0][0]   ## must update
	max = C[0][0]   ## must update
	for i in range(len(C)):
		for j in range(len(C[0])):
			if C[i][j] < min:
				min = C[i][j]
				I_Min= i
				J_Min = j
			elif C[i][j] > max:
				max =C[i][j]
				I_Max = i
				J_Max = j
	
	print("result min", min, "max", max)

				
findMinMax()

#####  Algorithm ####

x_0 = np.array((14, -4))
original_alpha = 0.03
original_gamma = 0.2
eta = 1.5
rho = 0.5
zeta = 0.05   # 5 percent

initial_cost = costFunction(np.array([[x_0[0]]]), np.array([[x_0[1]]]))[0][0]
#################################################
	
class MainWindow(QWidget):
	
	def __init__(self):
		super().__init__()
		
		layoutMain = QGridLayout()
		layoutMain.setContentsMargins(0, 0, 0, 0)
		layoutMain.setVerticalSpacing(0)

		panelToolbar = QWidget()
		layoutToolbar = QHBoxLayout()
		layoutToolbar.setContentsMargins(5, 8, 0, 9)

		btn1 = QPushButton("Run")
		layoutToolbar.addWidget(btn1, alignment = QtCore.Qt.AlignLeft)
		btn1.clicked.connect(self.runAlgorithm)
		
		btnSetting = QPushButton("Setting")
		layoutToolbar.addWidget(btnSetting, alignment = QtCore.Qt.AlignLeft)
		btnSetting.clicked.connect(self.openSetting)
		
		layoutToolbar.addStretch()
		
		#### Performance Surface ########
		self.visualization = Visualization()
		self.ui = self.visualization.edit_traits(parent=self,kind='subpanel').control
		
		### Function Graph
		pg.setConfigOptions(antialias=True)
		pg.setConfigOption('foreground', 'w')

		self.funcGraph = pg.GraphicsLayoutWidget()
		self.funcGraph.setBackground((127, 127, 127))
		
		self.funcGraphPlot = self.funcGraph.addPlot()
		self.funcGraphPlot.enableAutoRange(pg.graphicsItems.ViewBox.ViewBox.YAxis, enable = False)
		self.funcGraphPlot.setYRange(np.min(y), np.max(y))
		
		lblOriginalFunction = pg.LabelItem()
		lblOriginalFunction.setText("<span style = 'font-size: 14px; color : #00ff00'><b>&bull; True Function</b></span><br/><span style = 'font-size: 14px; color : #ffaa00'><b>&bull; Estimated Function</b></span>")
		lblOriginalFunction.setPos(2, -5)
		lblOriginalFunction.setParentItem(self.funcGraphPlot.getViewBox())
		
		self.iteErrGraph = pg.GraphicsLayoutWidget()
		self.iteErrGraph.setBackground((127, 127, 127))

		self.iteErrPlot = self.iteErrGraph.addPlot()
		self.iteErrPlot.setXRange(0, 20)
		self.iteErrPlot.setYRange(0, initial_cost)
		self.iteErrPlot.enableAutoRange(pg.graphicsItems.ViewBox.ViewBox.YAxis, enable = False)
		
		layoutMain.addWidget(panelToolbar, 0, 0)
		
		panelStatus = QWidget()
		panelStatus.setStyleSheet(" background-color:#808080;");
		layoutStatus = QHBoxLayout()
		layoutStatus.setContentsMargins(-5, 0, -5, 0)
		panelStatus.setLayout(layoutStatus)
		
		self.lblIsRunning = QLabel("<i><span style = 'color : #ffff00;font-size: 25px;'>Ready...</span></i>")
		
		self.lblTrueW111 = QLabel("<span style='font-size: 19px; color : #ffc01f'>True <i>w <sup>1</sup><sub>11</i></sub>:      <b>%0.2f</b></span>" % (w111))
		self.lblTrueW211 = QLabel("<span style='font-size: 19px; color : #ffc01f'>True <i>w <sup>2</sup><sub>11</i></sub>:      <b>%0.2f</b></span>" % (w211))
		
		spacing = QLabel("        ")
		
		self.lblEstW111 = QLabel("<span style='font-size: 19px; color : #ffc01f'>Est: <i>w <sup>1</sup><sub>11</i></sub>:      <b>%0.3f</b></span>" % (0))
		self.lblEstW211 = QLabel("<span style='font-size: 19px; color : #ffc01f'>Est: <i>w <sup>2</sup><sub>11</i></sub>:      <b>%0.3f</b></span>" % (0))
		
		layoutStatus.addWidget(self.lblIsRunning)
		layoutStatus.addStretch()
		layoutStatus.addWidget(self.lblTrueW111)
		layoutStatus.addWidget(self.lblTrueW211)
		layoutStatus.addWidget(spacing)
		layoutStatus.addWidget(self.lblEstW111)
		layoutStatus.addWidget(self.lblEstW211)
		
		layoutMain.addWidget(self.ui, 0, 1,  3, 2) 
		layoutMain.addWidget(self.funcGraph, 1, 0)
		layoutMain.addWidget(self.iteErrGraph, 2, 0, 2, 1)
		layoutMain.addWidget(panelStatus, 3, 1)
		layoutMain.setColumnStretch(0, 3)
		layoutMain.setColumnStretch(1, 5)
		
		layoutMain.setRowStretch(1, 3)
		layoutMain.setRowStretch(2, 3)
		
		panelToolbar.setLayout(layoutToolbar)
		self.setLayout(layoutMain)
		
		self.drawFunctionGraph(x_0[0], x_0[1])
		self.ui.setParent(self)
	
	#Logistic Function Graph #
	def drawFunctionGraph(self, curr_w111, curr_w211):
		yy = self.L((w211 * self.L((w111 * x) + b11)) + (w212 * self.L((w121 * x) + b12)) + b21)
		
		self.funcGraphPlot.plot(x, yy, pen = pg.mkPen((0, 255, 0), width = 2), clear = True)
		yye = self.L((curr_w211 * self.L((curr_w111 * x) + b11)) + (w212 * self.L((w121 * x) + b12)) + b21)
		self.funcGraphPlot.plot(x, yye, pen = pg.mkPen((255, 170, 0), width = 2))
	
	def L(self, x):
		return 1 / (1 + np.exp(-x))
		
	def runAlgorithm(self):
		self.resetAllGraphs()
		self.thread = AlgorithmRunner(self)
		self.thread.update_signal.connect(self.updateGraphs)
		self.thread.finish_signal.connect(self.finish)
		self.thread.start()
	
	def openSetting(self):
		dialog = InputDialog(self)
		dialog.setWindowModality(QtCore.Qt.WindowModal)
		dialog.setWindowFlags(dialog.windowFlags() |
                              QtCore.Qt.WindowSystemMenuHint |
                              QtCore.Qt.WindowMinMaxButtonsHint)
		dialog.exec_()
	
	errFont=QtGui.QFont()
	errFont.setPixelSize(20)
	def updateGraphs(self, iteration_indexs, errors, curr_w111, curr_w211, surface_xs, surface_ys, surface_zs, list_of_edges):
		#update graphs
		
		Running = "Running"
		dot_count = iteration_indexs[len(iteration_indexs) - 1] % 4
		for i in range(dot_count):
			Running += "."

		self.lblIsRunning.setText("<i><span style = 'color : #ffff00;font-size: 25px;'>" + Running + "</span></i>")
		
		plotErr = self.iteErrPlot.plot(iteration_indexs, errors,pen = pg.mkPen((255, 60,60), width=2), clear = True)
		self.lblError = pg.TextItem(anchor=(1, 1.0), color = (255, 0, 0))
		self.lblError.setFont(self.errFont)
		#self.lblError.setText("<span style='font-size: 10pt; color : #ff3c3c'>Error %0.4f</span>" % (2))
		self.lblError.setText("%.5f" %  errors[len(errors) - 1])
		self.lblError.setPos(iteration_indexs[len(iteration_indexs) - 1], errors[len(errors) - 1])	
		
		self.lblError.setParentItem(plotErr)
		if iteration_indexs[len(iteration_indexs) - 1] > 20:
			self.iteErrPlot.enableAutoRange(pg.graphicsItems.ViewBox.ViewBox.XAxis, enable = True)

		#########################
		
		#gist_earth
		self.visualization.points.mlab_source.reset(x = surface_xs, y = surface_ys, z= surface_zs)
		self.visualization.leadingPoint.mlab_source.reset(x = surface_xs[len(surface_xs) - 1], y = surface_ys[len(surface_ys) - 1], z= surface_zs[len(surface_zs) - 1])
	
		self.drawFunctionGraph(curr_w111, curr_w211)
		temp_list = list_of_edges[:]
		del(temp_list[0])
		
		self.visualization.points.mlab_source.dataset.lines = np.array(temp_list)
		self.visualization.points.mlab_source.update()
		
		self.lblEstW111.setText("<span style='font-size: 19px; color : #ffc01f'>Est: <i>w <sup>1</sup><sub>11</i></sub>:      <b>%0.3f</b></span>" % (curr_w111))
		self.lblEstW211.setText("<span style='font-size: 19px; color : #ffc01f'>Est: <i>w <sup>2</sup><sub>11</i></sub>:      <b>%0.3f</b></span>" % (curr_w211))
		
	def resetAllGraphs(self):
		
		self.visualization.points.mlab_source.dataset.lines = np.array([])
		self.visualization.points.mlab_source.update()
		self.funcGraphPlot.clear()
		
		self.iteErrPlot.clear()
		self.visualization.points.mlab_source.reset(x = [], y = [], z = [])
		self.visualization.leadingPoint.mlab_source.reset(x = [], y = [], z = [])
		
		self.iteErrPlot.setXRange(0, 20)
		self.iteErrPlot.setYRange(0, initial_cost)
		self.iteErrPlot.enableAutoRange(pg.graphicsItems.ViewBox.ViewBox.XAxis, enable = False)
		
		self.lblIsRunning.setText("<i><span style = 'color : #ffff00;font-size: 25px;'>Ready...</span></i>")

		self.lblTrueW111.setText("<span style='font-size: 19px; color : #ffc01f'>True <i>w <sup>1</sup><sub>11</i></sub>:      <b>%0.2f</b></span>" % (w111))
		self.lblTrueW211.setText("<span style='font-size: 19px; color : #ffc01f'>True <i>w <sup>2</sup><sub>11</i></sub>:      <b>%0.2f</b></span>" % (w211))
	
		self.lblEstW111.setText("<span style='font-size: 19px; color : #ffc01f'>Est: <i>w <sup>1</sup><sub>11</i></sub>:      <b>%0.3f</b></span>" % (0))
		self.lblEstW211.setText("<span style='font-size: 19px; color : #ffc01f'>Est: <i>w <sup>2</sup><sub>11</i></sub>:      <b>%0.3f</b></span>" % (0))
	def updateCostFunctionGraph(self):
		print("Updating cost function")
		
		self.visualization.surf_cat2.mlab_source.set(x = X_W111, y = Y_W211, scalars = C)
		self.visualization.surf_cat1.mlab_source.set(x = X_W111, y = Y_W211, scalars = C)
		
		self.visualization.scene.mlab.axes(ranges = (MIN_X_AXIS, MAX_X_AXIS, MIN_Y_AXIS, MAX_Y_AXIS, 0, max), extent = (MIN_X_AXIS, MAX_X_AXIS, MIN_Y_AXIS, MAX_Y_AXIS, 0, max))
		
	def drawInitialPoint(self, x_, y_):
		initial_cost = costFunction(np.array([[x_]]), np.array([[y_]]))[0][0]
		
		self.visualization.inititalPoint.mlab_source.reset(x=[x_], y=[y_], z=[initial_cost])
		self.iteErrPlot.setXRange(0, 20)
		self.iteErrPlot.setYRange(0, initial_cost)
		
	def finish(self, curr_w111, curr_w211):
		self.lblIsRunning.setText("<i><span style = 'color : #00ff00;font-size: 25px;font-style: bold;'>Converged!</span></i>")
		self.lblEstW111.setText("<span style='font-size: 19px; color : #ffc01f'>Est: <i>w <sup>1</sup><sub>11</i></sub>:      </span><span style = 'color : #00ff00;font-size: 19px;'><b>%0.3f</b></span>" % (curr_w111))
		self.lblEstW211.setText("<span style='font-size: 19px; color : #ffc01f'>Est: <i>w <sup>2</sup><sub>11</i></sub>:      </span><span style = 'color : #00ff00;font-size: 19px;'><b>%0.3f</b></span>" % (curr_w211))

		
class AlgorithmRunner(QtCore.QThread):

	iteration_indexs = []
	errors = []
	update_signal = QtCore.pyqtSignal(list, list, float, float, list, list, list, list)
	finish_signal = QtCore.pyqtSignal(float, float)
	
	surface_xs = []
	surface_ys = []
	surface_zs = []
	
	def __init__(self, obj):
		super().__init__()
		self.obj = obj
		
	def run(self):
		old_x = x_0
		old_delta_x = np.array((0,0))
		new_delta_x = 0
		old_y = costFunction(np.array([[old_x[0]]]), np.array([[old_x[1]]]))
		
		gamma = original_gamma
		alpha = original_alpha
		
		iteration_index = 0
		list_of_edges = []
		
		while True:
			g = gradientFunction(old_x)
			new_delta_x = (gamma * old_delta_x) - ((1 - gamma) * alpha * g)
			new_x = old_x + new_delta_x	
				
			new_y = costFunction(np.array([[new_x[0]]]), np.array([[new_x[1]]]))[0][0]
				
			
			if (new_y - old_y) < (old_y * 0.05):
				old_x = new_x
				old_y = new_y
				old_delta_x = new_delta_x
				alpha = eta * alpha
				
				gamma = original_gamma
					
				self.iteration_indexs.append(iteration_index)
				self.errors.append(new_y)
				
				self.surface_xs.append(new_x[0])
				self.surface_ys.append(new_x[1])
				self.surface_zs.append(new_y)
				
				list_of_edges.append([len(self.surface_xs) - 2, len(self.surface_xs) - 1])
				
				self.update_signal.emit(self.iteration_indexs, self.errors, old_x[0], old_x[1], self.surface_xs, self.surface_ys, self.surface_zs, list_of_edges)
				
				if abs(g[0]) < 0.001 and abs(g[1]) < 0.001:
					break
				
				time.sleep(0.5)
				
			else:
				alpha = rho * alpha
				gamma = 0
				
			iteration_index += 1
			
		self.surface_xs.clear()
		self.surface_ys.clear()
		self.surface_zs.clear()
		self.iteration_indexs.clear()
		self.errors.clear()
		
		self.finish_signal.emit(old_x[0], old_x[1])
		
class InputDialog(QDialog):
	
	def __init__(self, parent):
		super().__init__()
		
		self.parent = parent
		
		mainLayout = QGridLayout()
		mainLayout.setContentsMargins(0, 0, 0, 0)
		
		network_graph = NnNetworkGrapher.NnNetworkGrapher()
		
		panelInputs = QWidget()
		layoutInputs = QGridLayout()
		
		panelInputs.setLayout(layoutInputs)
		
		lb_w111 = QLabel('w111');self.txt_w111 = QLineEdit();self.txt_w111.setText(str(w111));self.txt_w111.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		self.txt_w111.setStyleSheet('background-color: #ffc01f')
		lb_b11 = QLabel('b11');self.txt_b11 = QLineEdit();self.txt_b11.setText(str(b11));self.txt_b11.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		lb_w121 = QLabel('w121');self.txt_w121 = QLineEdit();self.txt_w121.setText(str(w121));self.txt_w121.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		lb_b12 = QLabel('b12');self.txt_b12 = QLineEdit();self.txt_b12.setText(str(b12));self.txt_b12.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		lb_w211 = QLabel('w211');self.txt_w211 = QLineEdit();self.txt_w211.setText(str(w211));self.txt_w211.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		self.txt_w211.setStyleSheet('background-color: #ffc01f')
		lb_w212  = QLabel('w212');self.txt_w212 = QLineEdit();self.txt_w212.setText(str(w212));self.txt_w212.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		lb_b21 = QLabel('b21');self.txt_b21 = QLineEdit();self.txt_b21.setText(str(b21));self.txt_b21.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		
		panel11 = QWidget()
		layout11 = QHBoxLayout()
		panel11.setLayout(layout11)
		
		layout11.addWidget(lb_w111)
		layout11.addWidget(self.txt_w111)
		layout11.addWidget(lb_b11)
		layout11.addWidget(self.txt_b11)
		
		panel12 = QWidget()
		layout12 = QHBoxLayout()
		panel12.setLayout(layout12)
		
		layout12.addWidget(lb_w121)
		layout12.addWidget(self.txt_w121)
		layout12.addWidget(lb_b12)
		layout12.addWidget(self.txt_b12)
		
		panel21 = QWidget()
		layout21 = QHBoxLayout()
		panel21.setLayout(layout21)
		
		layout21.addWidget(lb_w211)
		layout21.addWidget(self.txt_w211)
		layout21.addWidget(lb_w212)
		layout21.addWidget(self.txt_w212)
		layout21.addWidget(lb_b21)
		layout21.addWidget(self.txt_b21)
		
		panelGraphLimit = QGroupBox("Graph Axis Limits")
		layoutGraphLimit = QHBoxLayout()
		panelGraphLimit.setLayout(layoutGraphLimit)
		
		lblXMin = QLabel("W111")
		self.txtXMin = QLineEdit(str(MIN_X_AXIS))
		self.txtXMin.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		
		lblXMax = QLabel("")
		self.txtXMax = QLineEdit(str(MAX_X_AXIS))
		self.txtXMax.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		
		lblYMin = QLabel("W211")
		self.txtYMin = QLineEdit(str(MIN_Y_AXIS))
		self.txtYMin.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		
		lblYMax = QLabel("")
		self.txtYMax = QLineEdit(str(MAX_Y_AXIS))
		self.txtYMax.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		
		layoutGraphLimit.addStretch()
		layoutGraphLimit.addWidget(lblXMin)
		layoutGraphLimit.addWidget(self.txtXMin)
		layoutGraphLimit.addWidget(lblXMax)
		layoutGraphLimit.addWidget(self.txtXMax)
		layoutGraphLimit.addWidget(lblYMin)
		layoutGraphLimit.addWidget(self.txtYMin)
		layoutGraphLimit.addWidget(lblYMax)
		layoutGraphLimit.addWidget(self.txtYMax)
		layoutGraphLimit.addStretch()
		
		panel22 = QGroupBox("Initial Point")
		layout22 = QHBoxLayout()
		panel22.setLayout(layout22)
		
		lblInitialX = QLabel("Initial W111")
		
		self.txtInitialX = QLineEdit(str(x_0[0]));self.txtInitialX.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		lblInitialY = QLabel("Initial W211")
		self.txtInitialY = QLineEdit(str(x_0[1]));self.txtInitialY.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		
		layout22.addWidget(lblInitialX)
		layout22.addWidget(self.txtInitialX)
		layout22.addWidget(lblInitialY)
		layout22.addWidget(self.txtInitialY)
		
		panel13 = QWidget()
		layout13 = QVBoxLayout()
		
		lblAlpha = QLabel("Alpha")
		self.txtAlpha = QLineEdit(str(original_alpha));self.txtAlpha.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		lblGamma = QLabel("Gamma")
		self.txtGamma = QLineEdit(str(original_gamma));self.txtGamma.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		lblEta = QLabel("Eta")
		self.txtEta = QLineEdit(str(eta));self.txtEta.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		lblRho = QLabel("Rho")
		self.txtRho = QLineEdit(str(rho));self.txtRho.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		lblZeta = QLabel("Zeta")
		self.txtZeta = QLineEdit(str(zeta));self.txtZeta.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		
		panelCtrlVar = QGroupBox("Parameters")
		layoutCtrlVar = QGridLayout()
		panelCtrlVar.setLayout(layoutCtrlVar)
		
		layoutCtrlVar.addWidget(lblAlpha, 0, 0)
		layoutCtrlVar.addWidget(self.txtAlpha, 0, 1)
		layoutCtrlVar.addWidget(lblGamma, 1, 0)
		layoutCtrlVar.addWidget(self.txtGamma, 1, 1)
		layoutCtrlVar.addWidget(lblEta, 2, 0)
		layoutCtrlVar.addWidget(self.txtEta, 2, 1)
		layoutCtrlVar.addWidget(lblRho, 3, 0)
		layoutCtrlVar.addWidget(self.txtRho, 3, 1)
		layoutCtrlVar.addWidget(lblZeta, 4, 0)
		layoutCtrlVar.addWidget(self.txtZeta, 4, 1)
		
		layoutInputs.addWidget(panel11, 0, 0)
		layoutInputs.addWidget(panel12, 1, 0)
		layoutInputs.addWidget(panel21, 0, 1)
		layoutInputs.addWidget(panel22, 1, 1)
		
		layoutInputs.addWidget(panelCtrlVar, 0, 2, 2, 1)
		
		layoutInputs.addWidget(panelGraphLimit, 2, 0, 1, 2)
		
		layoutInputs.setColumnStretch(0, 3)
		layoutInputs.setColumnStretch(1, 3)
		layoutInputs.setColumnStretch(2, 2)
		
		panelBtns = QWidget()
		layoutBtns = QHBoxLayout()
		layoutBtns.setDirection(QHBoxLayout.RightToLeft)
		panelBtns.setLayout(layoutBtns)
		
		
		btn_update = QPushButton("Update")
		btn_cancel = QPushButton("Cancel")
		btn_update.clicked.connect(self.btnUpdateClicked)
		btn_cancel.clicked.connect(self.btnCancelClicked)
		
		layoutBtns.addWidget(btn_cancel)
		layoutBtns.addWidget(btn_update)
		layoutBtns.addStretch()
		
		mainLayout.addWidget(network_graph, 0, 0)
		mainLayout.addWidget(panelInputs, 1, 0)
		mainLayout.addWidget(panelBtns, 2, 0, 2, 1)
		
		self.setLayout(mainLayout)
	
	
		
	def btnUpdateClicked(self):
		global original_alpha, original_gamma, eta, rho, zeta, MIN_X_AXIS,MAX_X_AXIS, MIN_Y_AXIS, X_W111,MAX_Y_AXIS, Y_W211, w111, w121, w211, w212, min, max, C, x_0, y, b11, b12, b21
		
		newAlpha = float(self.txtAlpha.text())
		newGamma = float(self.txtGamma.text())
		newEta = float(self.txtEta.text())
		newRho = float(self.txtRho.text())
		newZeta = float(self.txtZeta.text())
		
		original_alpha = newAlpha
		original_gamma = newGamma
		eta = newEta
		rho = newRho
		zeta = newZeta
		
		newXMin = float(self.txtXMin.text())
		newXMax = float(self.txtXMax.text())
		newYMin = float(self.txtYMin.text())
		newYMax = float(self.txtYMax.text())
		newW111 = float(self.txt_w111.text())
		newW121 = float(self.txt_w121.text())
		newB11 =  float(self.txt_b11.text())
		newB12 = float(self.txt_b12.text())
		newW211 = float(self.txt_w211.text())
		newW212 = float(self.txt_w212.text())
		newB21 = float(self.txt_b21.text())
		newInitialX = float(self.txtInitialX.text())
		newInitialY = float(self.txtInitialY.text())
					
		MIN_X_AXIS = newXMin
		MAX_X_AXIS = newXMax
		MIN_Y_AXIS = newYMin
		MAX_Y_AXIS = newYMax
			
		len1 = MAX_X_AXIS - MIN_X_AXIS
		len2 = MAX_Y_AXIS - MIN_Y_AXIS
			
		diff = len1 - len2
			
		if diff > 0:  # len1 is larger
			MIN_Y_AXIS -= diff / 2
			MAX_Y_AXIS += diff / 2
			self.txtYMin.setText(str(MIN_Y_AXIS))
			self.txtYMax.setText(str(MAX_Y_AXIS))
		else:
			MIN_X_AXIS -= abs(diff) / 2
			MAX_X_AXIS += abs(diff) / 2
			self.txtXMin.setText(str(MIN_X_AXIS))
			self.txtXMax.setText(str(MAX_X_AXIS))

		
		if (newXMin != MIN_X_AXIS or newXMax != MAX_X_AXIS or newYMin != MIN_Y_AXIS or newYMax != MAX_Y_AXIS) and (w111 == newW111 and w121 == newW121 and w211 == newW211 and w212 == newW212 and b11 == newB11 and b12 == newB12 and b21 == newB21):
			
			# only range is changed....
			X_W111, Y_W211 = np.mgrid[MIN_X_AXIS:MAX_X_AXIS:0.2, MIN_Y_AXIS:MAX_Y_AXIS:0.2]
			y = nominalFunction(x, w111, w211)
			C = costFunction(X_W111, Y_W211)
			
			findMinMax()
			self.parent.updateCostFunctionGraph()
			
		elif x_0[0] != newInitialX or x_0[1] != newInitialY or newXMin != MIN_X_AXIS or newXMax != MAX_X_AXIS or newYMin != MIN_Y_AXIS or newYMax != MAX_Y_AXIS or w111 != newW111 or w121 != newW121 or w211 != newW211 or w212 != newW212 or b11 != newB11 or b12 != newB12 or b21 != newB21:
			
			w111 = newW111
			w121 = newW121
			w211 = newW211
			w212 = newW212
			b11 = newB11
			b12 = newB12
			b21 = newB21

			x_0 = [float(self.txtInitialX.text()), float(self.txtInitialY.text())]
			
			X_W111, Y_W211 = np.mgrid[MIN_X_AXIS:MAX_X_AXIS:0.2, MIN_Y_AXIS:MAX_Y_AXIS:0.2]

			y = nominalFunction(x, w111, w211)
			C = costFunction(X_W111, Y_W211)
			
			self.parent.resetAllGraphs()
			self.parent.drawFunctionGraph(x_0[0], x_0[1])
			self.parent.drawInitialPoint(x_0[0], x_0[1])
			
			findMinMax()
			self.parent.updateCostFunctionGraph()
		
	def btnCancelClicked(self):
		print('Cancel')
		self.close()
		
if __name__ == '__main__':
	app = QApplication.instance()
	
	win = MainWindow()
	win.show()
	
	app.exec_()