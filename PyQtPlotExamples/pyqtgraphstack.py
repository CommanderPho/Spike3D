# Restructuring this module as a class
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg



"""
Author : *Yatharth Bhasin* (Github → yatharthb97)
License: *MIT open-source license* (https://opensource.org/licenses/mit-license.php)
This piece of software was released on GistHub : https://gist.github.com/yatharthb97/f3748ef894627748bacccf092648aa59

PyQtGraphStack is a wrapper around the pyqtgraph API for rapid prototyping, focused around creation of multiple plots with multiple curves. 

As the name implies, it creates a stack of graps, one below the other, unless explicitly specified. (Passing 'samerow'=True as a keyword arguement will create the graph on the same column.)


* Generate html documentation for this file using `pydoc` ↓ 
`bash>$ pydoc -w "PyQTGraphStack.py" `

* pyqtgraph docs → https://pyqtgraph.readthedocs.io/en/latest/

* Dependencies : pyqtgraph, PyQt, numpy 
(https://pyqtgraph.readthedocs.io/en/latest/installation.html#installation) 

The object encapsulates the following entities:

→ Entities that the user manages: 
"canvas": It is the portion of the window, which holds curve/curves. 
           Each curve is identified by a unique name passed by the end user.
           Managed using `add_canvas()` and `get_canvas()` methods.

"curve" : Object that represents the data as lines on the screen. Each curve is placed on a canvas.
		  Each curve inside a canvas is identified with a unique name passed by the end user.
          Managed using `add_curve()` and `get_curve()` methods.


→ Entities the user has no need to care about:
"window" : The object that holds the layout, the window title and the icon.
"app"    : The application object that controls the thread.

•••• Mini Tutorial ••••

# Create Graph and Add a canvas
>> PyQTGraphWrapper graph;
>> graph.add_canvas('data1', curvename = 'curve1')

# Set name for canvas 'data1':
>> canvas = graph.get_canvas('data1')
>> canvas.setTitle(title="Data-1 Graph")

# Set data for curve 'curve1' on 'data1':
>> curve = graph.get_curve('curve1', data1')
>> curve.setData(xData, yData)

# Launch Plot Window
>> graph.launch()

# TODO
 * kwargs forwarding to pyqtgraph functions.
"""


class PyQtGraphStack:

	def __init__(self, **kwargs):
		"""
		Initalization function (constructor):

		kwargs:
		------

		* resolution : list of two values [x_resolution, y_resolution].
		* icon : adds window icon if the passed string path is a png, ignores it otherwise.
		* title : sets the window title

		"""

		# Resolution Resources → Fixed Aspect Ratio of 16:9 enforced
		self.resolution = [192, 108]
	
		# Rows & Colums
		self.rows = 1
		self.cols = 4

		if 'resolution' in kwargs:
			self.current_res = kwargs['resolution']

		# Set Application
		self.app = QtGui.QApplication([])
		
		# Set Window
		title = "PyQtGraphStack"
		if 'title' in kwargs:
			title = kwargs['title']

		self.window = pg.GraphicsLayoutWidget(show=True, title=title)
		self.window.resize(*self.resolution)

		self.window.setWindowTitle(self.title)

		if 'icon' in kwargs:
			if kwargs['icon'].endswith('.png'):
				# Set icon
				icon = QtGui.QIcon(kwargs['icon'])
				self.window.setWindowIcon(icon)

		# Enable antialiasing and OpenGL use
		pg.setConfigOptions(antialias=True, useOpenGL=True)

	def add_canvas(self, name, rowspan = 4, colspan = 1, pen = 'y', symbolBrush = 'c', symbolSize = 4,  xLog = False, yLog = False, blankcanvas = True, **kwargs):
		
		"""
		Add a canvas object to the graph.

		Positional Arguements:
		----------------------
		* name : Name of the canvas object and its unique reference
	
		Optional Arguements:
		--------------------
		* rowspan : Numbers of rows occupied
		* colspan :  Numbers of columns occupied
		* pen : pyqtgraph option
		* rowspan : Span of canvas across rows
		* colspan : Span of canvas across columns
		* symbolBrush : pyqtgraph option
		* symbolSize : pyqtgraph option
		* xLog(bool) : sets x-axis to log scale
		* yLog(bool) : sets y-axis to log scale
		* blankcanvas(bool) : Creates a blank canvas is set to True, 
							  False creates a canvas with one curve
		
		kwargs:
		-------
		* samerow(bool) : Forces this curve on the same row instead of stacking down.
		* title : Title of the canvas, default is same as `name`
		* port : Port name, relavant for real time input and update
		* x_label : x-axis label
		* x_units : x-axis label units
		* y_label : y-axis label
		* y_units : y-axis label units
		* legend(bool) : activate/deactivate legend 
		* xRange : [low, high] - x-range
		* yRange : [low, high] - y-range
		* curvename : Override the name of the default curve of the canvas
					  (default is 'curve')

		"""
		if name in canvaslist:
			raise Exception(f"Curve name must be a unique identifier! {name} already exists.")

		# Decide whether to use the next column or row
		if 'samerow' in kwargs:
			if not kwargs['samerow']:
				self.window.nextRow()
		else:
			self.window.nextRow()

		title = "Unnammed Plot!"
		port = ""
		x_label = "X axis →"
		x_units = ""
		y_label = "Y axis →"
		y_units = ""

		if 'title' in kwargs:
			title = kwargs['title']
		
		if 'port' in kwargs:
			port = kwargs['port']

		if 'x_label' in kwargs:
			x_label = kwargs['x_label']

		if 'x_units' in kwargs:
			x_units = kwargs['x_units']

		if 'y_label' in kwargs:
			y_label = kwargs['y_label']

		if 'y_units' in kwargs:
			y_units = kwargs['y_units']

		# Resizing
		new_rows = self.rows + rowspan
		new_cols = self.cols + colspan
		self.resize(new_rows, new_cols)

		new_canvas = window.addPlot(title=title, row = self.rows, col = self.cols, rowspan=rowspan, colspan=colspan)

		self.rows = new_rows
		self.cols = new_cols

		new_canvas.showGrid(x = True, y = True)
		new_canvas.setLabel('left', y_label, y_units)
		new_canvas.setLabel('bottom', x_label, x_units)

		if 'legend' in kwargs:
			if kwargs['legend']
				new_canvas.addLegend()

		if 'xRange' in kwargs:
			new_canvas.setRange(xRange=kwargs['xRange'], update=True, disableAutoRange=True)
		if 'yRange' in kwargs:
			new_canvas.setRange(yRange=kwargs['yRange'])

		self.canvaslist[name] = new_canvas

		#Generation of first curve
		if not blankcanvas:
			first_curve = canvas.plot(pen=pen, symbolBrush=symbolBrush, symbolSize=symbolSize)
			first_curve.setLogMode(xLog, yLog)

			curve_name = 'curve'
			if 'curvename' in kwargs:
				curve_name = kwargs['curvename']

			self.curvelist[name][curve_name] = first_curve

	def get_canvas(self, canvas_name):
		"""
		Returns the reference of the canvas object, if a valid canvas name is passed.
		sThrows an exception otherwise
		"""
		if not canvas_name in self.canvaslist:
			raise Exception(f"No such canvas exists - {canvas_name}!")
		else:
			return self.canvaslist[canvas_name]

	def add_curve(self, name, canvas_name, pen = 'y', symbolBrush = 'c', symbolSize = 4, **kwargs):
		
		"""
		This function adds a curve to a given canvas object.  

		Positional Arguements:
		----------------------
		* name : Unique key to access the curve
		* canvas_name : Unique name of a canvas entity that already exists
		
		Optional Arguements:
		--------------------
		* pen : pyqtgraph option
		* symbolBrush : pyqtgraph option
		* symbolSize : pyqtgraph option

		kwargs:
		-------
		* title : Title of curve used for legends and labelling
		* xData : Sets the x-data fields
		* yData : Sets the y-data fields
		* legend(bool) : Activate/Deactivate the legend for the particular canvas
		Note: len(xData) == len(yData) or an exception might be thrown
		"""

		# Validation
		if not canvas_name in self.canvaslist:
			raise Exception(f"No such canvas exists - {name}!")
		if not name in self.canvaslist[name]:
			raise Exception(f"A curve with this name already exists in the scope of the canvas - {canvas_name}!")

		title = name
		if 'title' in kwargs:
			title = kwargs['title']
		curve = self.canvaslist[canvas_name].plot(pen=pen, symbolBrush=symbolBrush, symbolSize=symbolSize, title=title)

		#Set data if passed
		if 'xData' and 'yData' in kwargs:
			curve.setData(x=xData, y=yData)
		elif 'yData' in kwargs:
			curve.setData(y=yData)
		
		# Append
		self.curvelist[canvas_name][name] = curve
		if 'legend' in kwargs:
			if kwargs['legend'];
				self.canvaslist[canvas_name].addLegend()

	def get_curve(self, curve_name, canvas_name):

		"""
		Returns a reference to a curve if a valid curve_name and canvas_name are passed.
		Raises an exception otherwise.
		"""

		if not canvas_name in self.canvaslist:
			raise Exception(f"No such canvas exists - {canvas_name}!")
		if not curve_name in self.curvelist[canvas_name]:
			raise Exception(f"No such curve exists - {curve_name}!")

		return self.canvaslist[canvas_name][curve_name]

	def resize(self, new_rows, new_cols):
		
		"""
		Resizes the window based on the updated rows and columns in passed.
		It also resets the class `rows`, `cols`, and `current_res` attributes.
		"""
		new_x_res = self.resolution[0] * new_rows / self.rows
		new_y_res = self.resolution[1] * new_cols / self.cols

		self.resolution = [new_x_res, new_y_res]
		self.window.resize(*self.resolution)

	def launch(self):
		"""
		Launches the GUI window and blocks the main thread
		"""
		self.app.exec_()

	@classmethod
	def get_QTimer(cls, callback_fn):
		"""
		Returns a QTimer instance with a connected callback function.
		The returned QThread (QTimer) can be started with `timer.start(time_in_ms)`.
		The thread can be stopped with `timer.stop()`.
		"""
		timer = QtCore.QTimer()
		timer.timeout.connect(callback_fn)
		return timer

	@classmethod
	def set_scrolling_data(cls, curve_, canvas_, xData, yData):
		"""
		Updates the data and updates the "canvas view" to create a scrolling canvas 
		effect.
		Warning : The curve and canvas objects are not verified.

		"""
		curve_.setData(xData, yData)
		canvas_.setPos(xData[0], 0)

	@classmethod
	def set_data(cls, curve_, xData, yData):
		"""
		Updates the data of a curve object.
		"""
		curve_.setData(xData, yData)
		