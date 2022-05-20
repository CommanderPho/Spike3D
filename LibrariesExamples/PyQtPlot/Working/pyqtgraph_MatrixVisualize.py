
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore

import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg


X_1 = 1
Y_1 = 0
X_2 = 0
Y_2 = 1

def txtChanged():
	global X_1, Y_1, X_2, Y_2
	try:
		X_1 = (int)(line_edit1.text())
		Y_1 = (int)(line_edit2.text())
		X_2 = (int)(line_edit3.text())
		Y_2 = (int)(line_edit4.text())
		valueChanged()
		
	except:
		print("exception")
	

def resize_function():
	print("p")

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

pg.setConfigOptions(antialias=True)
win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(700, 700)
win.setWindowTitle('pyqtgraph example: Plotting')


x = np.linspace(-1, 1, 100)
y1 = np.sqrt(1-np.square(x))
y2 = -1 * np.sqrt(1-np.square(x))


widget = QtGui.QWidget()
widget.setMaximumHeight(100)

p = widget.palette()
p.setColor(widget.backgroundRole(), QtCore.Qt.red)
widget.setPalette(p)

widgetMatrix = QtGui.QWidget()
widgetMatrix.setMaximumWidth(150)

gridLayout = QtGui.QGridLayout(widgetMatrix)
line_edit1 = QtGui.QLineEdit("1");line_edit1.setFixedWidth(50);line_edit1.textChanged.connect(txtChanged);
line_edit2 = QtGui.QLineEdit("0");line_edit2.setFixedWidth(50);line_edit2.textChanged.connect(txtChanged);
line_edit3 = QtGui.QLineEdit("0");line_edit3.setFixedWidth(50);line_edit3.textChanged.connect(txtChanged);
line_edit4 = QtGui.QLineEdit("1");line_edit4.setFixedWidth(50);line_edit4.textChanged.connect(txtChanged);
gridLayout.addWidget(line_edit1, 0, 0)
gridLayout.addWidget(line_edit2, 1, 0)
gridLayout.addWidget(line_edit3, 0, 1)
gridLayout.addWidget(line_edit4, 1, 1)

boxLayout = QtGui.QHBoxLayout(widget)

boxLayout.addWidget(widgetMatrix)
boxLayout.addStretch()

pp = QtGui.QGraphicsProxyWidget()

pp.setWidget(widget)

win.nextRow()

p = win.addLayout(row = 0, col = 0)
p.addItem(pp,row=0,col=0)



p4 = win.addPlot(title="Parametric, grid enabled", row = 1, col =0)
p4.disableAutoRange()
p4.setXRange(-4, 4, padding = 0)
p4.setYRange(-4, 4, padding = 0)
win.resizeEvent(resize_function())

p4.setAspectLocked()

#roi = pg.PolyLineROI([0, 1], [2, 1], pen=(1,9))
#p4.addItem(roi)

roi = pg.RectROI([0, 20], [2, 1], pen=(0,9))
p4.addItem(roi)


p4.showGrid(x=True, y=True)

h_lines = []
v_lines = []

for i in range(10):
	line_1 = pg.InfiniteLine(pen=pg.mkPen((0,80, 110), width=2))
	line_1.setAngle(0)
	line_1.setValue((0, i))
	p4.addItem(line_1)
	v_lines.append(line_1)
		
	line_2 = pg.InfiniteLine(pen=pg.mkPen((0,80, 110), width=2))
	line_2.setAngle(0)
	line_2.setValue((0, -1 * i))
	p4.addItem(line_2)
	v_lines.append(line_2)
	
	line_3 = pg.InfiniteLine(pen=pg.mkPen((0,80, 110), width=2))
	line_3.setAngle(90)
	line_3.setValue((i, 0))
	p4.addItem(line_3)
	h_lines.append(line_3)
		
	line_4 = pg.InfiniteLine(pen=pg.mkPen((0,80, 110), width=2))
	line_4.setAngle(90)
	line_4.setValue((-i, 0))
	p4.addItem(line_4)
	h_lines.append(line_4)

	print(i)



p4.plot(x, y1, pen=pg.mkPen('g', width=2))
p4.plot(x, y2, pen=pg.mkPen('g', width=2))

vector1 = pg.ArrowItem(pos = (1,0), angle = 180, brush = (0, 255,0), pen=pg.mkPen('g', width=2))
vector1.opts['pos'] = (1,1)
print(vector1.opts['pos'])
p4.addItem(vector1)
p4.plot([0, 1],[0, 0] ,pen=pg.mkPen('g', width=3))

vector2 = pg.ArrowItem(pos = (0,1), angle = 90, brush = (255, 0,0), pen=pg.mkPen('r', width=2))
p4.addItem(vector2)
p4.plot([0,0],[0,1],pen=pg.mkPen('r', width=3))


def valueChanged():

	theta = slider.value() / 50
	
	x_1 = (X_1 * theta) + (1 - theta)
	y_1 = (Y_1 * theta)
	
	x_2 = (X_2 * theta)
	y_2 = (Y_2 * theta) + (1 - theta)
	
		
	X1 = (x_1 * x) + (x_2 * y1)
	X2 = (x_1 * x) + (x_2 * y2)
	Y1 = (y_1 * x) + (y_2 * y1)
	Y2 = (y_1 * x) + (y_2 * y2)


	
	slope1 = 0
	slope2 = 0
	slope2_inv = 0
	ang1 = 0
	ang2 = 0
	
	
	if(x_1 == 0):
		if(y_1 > 0):
			ang1 = 90	
		else:
			ang1 = 270
	elif(y_1 == 0):
		if(x_1 > 0):
			ang1 = 0
		else:
			ang1 = 180
	else:
		slope1 = y_1 / x_1
		ang1 = (np.arctan(slope1) * 180) / np.pi
		
	if(x_2 == 0):
		if(y_2 > 0):
			ang2 = 90
		else:
			ang2 = 270
	elif(y_2 == 0):
		if(x_2 > 0):
			ang2 = 0
		else:
			ang2 = 180
	else:
		slope2 = y_2 / x_2
		slope2_inv = x_2 / y_2
		
		ang2 = (np.arctan(slope2) * 180) / np.pi
	
	p4.plot([0],[0],clear = True )
	
	for i in range(10):
	
		y_intercept = y_2 - (x_2 * slope1)
		
		line_1 = h_lines[i * 2]
		line_1.setAngle(ang1)
		line_1.setValue((0, i * y_intercept))
		
		
		line_2 = h_lines[(i * 2) + 1]
		line_2.setAngle(ang1)
		line_2.setValue((0, -i * y_intercept))
		
		x_intercept = x_1 - (y_1 * slope2_inv)
		

		line_3 = v_lines[i * 2]
		line_3.setAngle(ang2)
		line_3.setValue((i * x_intercept, 0))
		
		line_4 = v_lines[(i * 2) + 1]
		line_4.setAngle(ang2)
		line_4.setValue((-i * x_intercept, 0))

		p4.addItem(line_1)
		p4.addItem(line_2)	
		p4.addItem(line_3)
		p4.addItem(line_4)
	
	#vector1 = pg.ArrowItem()


	vector1.resetTransform()
	vector1.setPos(x_1,y_1)
	vector1.rotate(180 - ang1)
	print("ang1 " , ang1)
	
	vector2.resetTransform()
	vector2.setPos(x_2,y_2)
	vector2.rotate(180 - ang2)
	print("ang2 ", ang2)
	
	p4.addItem(vector1)
	p4.addItem(vector2)	
	
	
	p4.plot([0, x_1],[0, y_1], pen=pg.mkPen('g', width=3))
	p4.plot([0, x_2],[0, y_2], pen=pg.mkPen('r', width=3))
	
	p4.plot(X1, Y1,  pen=pg.mkPen('g', width=2))
	p4.plot(X2, Y2,  pen=pg.mkPen('g', width=2))


win.nextRow()
slider_max = 50
slider = QtGui.QSlider(QtCore.Qt.Horizontal)
slider.setMinimum(0)
slider.setMaximum(slider_max)
slider.setTickInterval(1)
slider.valueChanged.connect(valueChanged)

proxy = QtGui.QGraphicsProxyWidget()

proxy.setWidget(slider)

p3 = win.addLayout(row=2, col=0)
p3.addItem(proxy,row=0,col=0)



if __name__ == '__main__':
	import sys
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_() 
