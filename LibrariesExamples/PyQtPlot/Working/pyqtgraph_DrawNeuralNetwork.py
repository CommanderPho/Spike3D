from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout

import pyphoplacecellanalysis.External.pyqtgraph as pg
import numpy as np

import sys

no_of_input = 1
neuron_counts = [2,1]

class NnNetworkGrapher(QWidget):
	def __init__(self):
		super().__init__()
		
		layoutMain = QGridLayout()
		layoutMain.setContentsMargins(0, 0, 0, 0)

		self.setLayout(layoutMain)
		
		mainGraphicPanel = pg.GraphicsLayoutWidget()
		pg.setConfigOptions(antialias = True)
		
		
		self.plot = mainGraphicPanel.addPlot()
		self.plot.invertY(True)
		self.plot.setAspectLocked()
		self.plot.hideAxis('left')
		self.plot.hideAxis('bottom')

		
		layoutMain.addWidget(mainGraphicPanel, 0, 0)
		
		self.drawNetwork()
		
	def drawNetwork(self):
		
		no_of_temp_input = no_of_input
		
		tt = no_of_input
		no_of_lines = 0
		for i in range(len(neuron_counts)):
			no_of_lines = no_of_lines + (neuron_counts[i] * (3 + tt))
			tt = neuron_counts[i]
		lines_color = np.zeros([no_of_lines, 5])
		for i in range(no_of_lines):
			lines_color[i] = (255,0,0,255,1)

		
		lines = np.zeros([no_of_lines,2])
		lines_pen = np.zeros(no_of_lines, dtype=(float,5))
		#lines_pen.fill()
		
		lines_index = 0
		lines_index_offset = 0
				
		for i in range(len(neuron_counts)):
			for j in range(neuron_counts[i]):
				for k in range(no_of_temp_input):
					lines[lines_index] = [lines_index_offset + no_of_temp_input + (4 * j), lines_index_offset + k]
					lines_pen[lines_index] = (0,255,0,255,2)
					lines_index = lines_index + 1
					
					
				ddd = lines_index_offset + no_of_temp_input + (4 * j)

				lines[lines_index] = [ddd + 1, ddd + 2]
				lines_pen[lines_index] = (0,255,0,255,2)
				lines_index = lines_index + 1	
				
					
				lines[lines_index] = [ddd + 1, ddd + 3]
				lines_pen[lines_index] = (0,255,0,255, 2)
				lines_index = lines_index + 1
					
					
				lines[lines_index] = [ddd + 3, lines_index_offset + no_of_temp_input + (4 * neuron_counts[i]) + j]
				lines_pen[lines_index] = (0,255,0,255,2)
				lines_index = lines_index + 1	
				
						
			# update offset
			lines_index_offset = lines_index_offset + (neuron_counts[i] * 4) + no_of_temp_input
			no_of_temp_input = neuron_counts[i]
				
		d_nodes = np.zeros([sum(neuron_counts) * 5 + no_of_input, 2])
		d_symbols = np.empty(sum(neuron_counts) * 5 + no_of_input, dtype=str)
		d_symbols.fill('t')
		d_nodes_index = 0
		
		v_spacing_input = 10
		v_spacing_neurons = 25
		node_spacing = 18
		pos = np.zeros([no_of_input, 2])
		symbols = []
		
		g = pg.GraphItem()
		
		for i in range(no_of_input):
			pos[i] = [0, ((v_spacing_input * (no_of_input - 1))/2 ) - v_spacing_input * i]
			symbols.append('o')
			d_nodes[d_nodes_index] = pos[i]
			
			lb = pg.LabelItem(justify = 'right' , color = 'CCFF00')
			lb.setText("<span style='font-size: 2pt; font-style:italic'>p<sub>%d</sub></span>" % (no_of_input - i))
			lb.setPos(-node_spacing / 2 + pos[i][0] - .8,-v_spacing_input + 3.5 + 	pos[i][1])
			
			self.plot.addItem(lb)
			
			d_nodes_index = d_nodes_index + 1
		
		g.setData(pos=pos, size=3, symbol=symbols,symbolBrush=(217,83,25),  pxMode=False)		
		self.plot.addItem(g)
		
		
		
		node_pos = np.zeros([sum(neuron_counts) * 2 ,2])
		node_symbols = np.empty(sum(neuron_counts) * 2 , dtype=str)
		node_symbols.fill('s')
		
		circle_node_pos = np.zeros([sum(neuron_counts) * 2 - neuron_counts[-1] ,2])
		circle_node_symbols = np.empty(sum(neuron_counts) * 2 - neuron_counts[-1], dtype=str)
		circle_node_symbols.fill('o')
		
		#for i in range(neuron_counts[-1]):
		#	circle_node_symbols[[sum(neuron_counts) * 2  - i - 1]] = ''
		
		node_index = 0
		circle_node_index = 0
		
		node_arrows_head_index = 0
		node_arrows_head = np.zeros([sum(neuron_counts) + neuron_counts[-1],2])
		node_arrows_symbols = []
		
		g1 = pg.GraphItem()
		g2 = pg.GraphItem()
		gd = pg.GraphItem()
		gt = pg.GraphItem()

		for i in range(len(neuron_counts)):
			for j in range(neuron_counts[i]):
				node_pos[node_index] = [node_spacing + (i * 3 * node_spacing), (v_spacing_neurons * (neuron_counts[i] - 1))/2 - (v_spacing_neurons * j)]
				
				d_nodes[d_nodes_index] = [node_pos[node_index][0] - 3,node_pos[node_index][1]]
				d_nodes_index = d_nodes_index + 1
				d_nodes[d_nodes_index] = [node_pos[node_index][0],node_pos[node_index][1]]
				
				lb = pg.LabelItem(justify = 'right' , color = 'CCFF00')
				lb.setText("<span style='font-size: 2pt; font-style: italic;'>w <sup>%d</sup><sub>%d</sub></span>" % (i + 1, neuron_counts[i] - j))
				lb.setPos(node_pos[node_index][0] - 6,node_pos[node_index][1] - 6 - 6)
				
				
				lb5 = pg.LabelItem(justify = 'right' , color = 'CCFF00')
				lb5.setText("<span style='font-size: 2.2pt;'>&sum;</span>")
				lb5.setPos(node_pos[node_index][0] - 5.3,node_pos[node_index][1]- 7)
				lb5.setZValue(2)
				
				self.plot.addItem(lb5)
				self.plot.addItem(lb)
				
				d_nodes_index = d_nodes_index + 1
				node_index = node_index + 1
				###
				
				d_nodes[d_nodes_index] = [node_spacing + (i * 3 * node_spacing), (v_spacing_neurons * (neuron_counts[i] - 1))/2 - (v_spacing_neurons * j) + 0.35 * v_spacing_neurons]
				circle_node_pos[circle_node_index] = d_nodes[d_nodes_index]
					
				lb1 = pg.LabelItem(justify = 'right' , color = 'CCFF00')
				lb1.setText("<span style='font-size: 2pt;'>1</span>")
				lb1.setPos(circle_node_pos[circle_node_index][0] - 4.5,circle_node_pos[circle_node_index][1] - 6 + 3)
				
				self.plot.addItem(lb1)
					
				lb2 = pg.LabelItem(justify = 'right' , color = 'CCFF00')
				lb2.setText("<span style='font-size: 2pt; font-style: italic;'>b<sup>%d</sup><sub>%d</sub></span>" % (i + 1, neuron_counts[i] - j))
				lb2.setPos(circle_node_pos[circle_node_index][0] - 2,circle_node_pos[circle_node_index][1] - 7)
				
				self.plot.addItem(lb2)
					
				circle_node_index = circle_node_index + 1
				d_nodes_index = d_nodes_index + 1	

				
				###
				node_pos[node_index] = [2 * node_spacing + (i * 3 * node_spacing) + .8, (v_spacing_neurons * (neuron_counts[i] - 1))/2 - (v_spacing_neurons * j)]
				d_nodes[d_nodes_index] = node_pos[node_index]
			
				lb4 = pg.LabelItem(justify = 'right' , color = 'CCFF00')
				lb4.setText("<span style='font-size: 2pt; font-style: italic;'>a<sup>%d</sup><sub>%d</sub></span>" % (i + 1, neuron_counts[i] - j))
				lb4.setPos(node_pos[node_index][0] + 4,node_pos[node_index][1] - 8.5)
				self.plot.addItem(lb4)
				
					
				lb6 = pg.LabelItem(justify = 'right' , color = 'CCFF00')
				lb6.setText("<span style='font-size: 2.2pt;'>&int;</span>")
				lb6.setPos(node_pos[node_index][0] - 5.2,node_pos[node_index][1]- 7)
				lb6.setZValue(2)
				self.plot.addItem(lb6)
									

				lb7 = pg.LabelItem(justify = 'right' , color = 'CCFF00')
				lb7.setText("<span style='font-size: 2.2pt;'>&oline;&oline;</span>")
				lb7.setPos(node_pos[node_index][0] - 5.8,node_pos[node_index][1]- 5)
				lb7.setZValue(2)
				self.plot.addItem(lb7)
				
				
				
				node_index = node_index + 1
				d_nodes_index = d_nodes_index + 1
				
				###
				node_arrows_head[node_arrows_head_index] = [2 * node_spacing + (i * 3 * node_spacing) - 1.5 - 3 + .8, (v_spacing_neurons * (neuron_counts[i] - 1))/2 - (v_spacing_neurons * j)]
				
				lb3 = pg.LabelItem(justify = 'right' , color = 'CCFF00')
				lb3.setText("<span style='font-size: 2pt; font-style: italic;'>n<sup>%d</sup><sub>%d</sub></span>" % (i + 1, neuron_counts[i] - j))
				lb3.setPos(node_arrows_head[node_arrows_head_index][0] - 12,node_arrows_head[node_arrows_head_index][1] - 8.5)
				self.plot.addItem(lb3)
				
				node_arrows_head_index = node_arrows_head_index + 1
				node_arrows_symbols.append('t2')
				
				##
								
			for j in range(neuron_counts[i]):
				d_nodes[d_nodes_index] = [3 * node_spacing + (i * 3 * node_spacing), (v_spacing_neurons * (neuron_counts[i] - 1))/2 - (v_spacing_neurons * j)]
				
				if i != len(neuron_counts) - 1:
					circle_node_pos[circle_node_index] = d_nodes[d_nodes_index] 
					circle_node_index = circle_node_index + 1
				d_nodes_index = d_nodes_index + 1
			
			if i == len(neuron_counts) - 1:
				for j in range(neuron_counts[i]):
					node_arrows_head[node_arrows_head_index] = [3 * node_spacing + (i * 3 * node_spacing), (v_spacing_neurons * (neuron_counts[i] - 1))/2 - (v_spacing_neurons * j)]
					node_arrows_head_index = node_arrows_head_index + 1
					node_arrows_symbols.append('t2')
				
				
		print("node_arrows_head", len(node_arrows_head))
		print("node_arrows_symbols", len(node_arrows_symbols))
				
		g1.setData(pos=node_pos, size=6, symbol=node_symbols, symbolBrush=(217,83,25),  pxMode=False)		
		
		g2.setData(pos=circle_node_pos, size=2.4, symbol=circle_node_symbols, symbolBrush=(0,255,0),  pxMode=False)		
			
		gt.setData(pos=node_arrows_head, size=3, symbol=node_arrows_symbols, symbolBrush=(0,255,0),  pxMode=False)		
		gd.setData(pos=d_nodes, size=0, pen = lines_pen, adj = lines.astype(int), lines = lines_color, symbol=d_symbols,symbolBrush=(0,255,0),  pxMode=False)		
		
		g.setZValue(1)
		g1.setZValue(1)
		g2.setZValue(2)
		self.plot.addItem(g1)	
		self.plot.addItem(g2)	
		self.plot.addItem(gd)	
		self.plot.addItem(gt)	

app = QApplication(sys.argv)
win = NnNetworkGrapher()
win.show()
app.exec_()