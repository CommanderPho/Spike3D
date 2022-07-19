
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QDialog, QGridLayout,\
QTabWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QComboBox, QLineEdit,QScrollArea, QColorDialog
from PyQt5 import QtGui, QtCore
import pyphoplacecellanalysis.External.pyqtgraph as pg
import numpy as np
from functools import partial
import os
import qdarkstyle

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        view = pg.GraphicsLayoutWidget()
        view.setStyleSheet("padding : 0px")
        layout.addWidget(view)

        plot = view.addPlot()

        menuList = plot.scene().contextMenu
        action = QtGui.QAction("Settings")
        menuList.append(action)
        action.triggered.connect(self.settingMenuAction)

        plot.setAspectLocked(True)

        self.setting = Setting()
        self.setting.axisSetting.addAxis('BOTTOM', -2, 2)
        self.setting.axisSetting.addAxis('LEFT', -2, 2)

        plot.setLabel('bottom', self.setting.axisSetting.getName(0))
        plot.setLabel('left', self.setting.axisSetting.getName(1))

        self.scatters = []

        ###### draw scatter plot for each category ########
        for i in range(2):
            self.setting.colorSetting.addClass(str(i), i)
            n = 500
            scatter = pg.ScatterPlotItem(size=self.setting.colorSetting.getSize(i), symbol = self.setting.colorSetting.getSymbol(i), pen = self.setting.colorSetting.getPen(i), brush=self.setting.colorSetting.getBrush(i))
            self.scatters.append(scatter)
            pos = np.random.normal(size=(2,n), scale=1)
            pos = pos + (i * 3)
            spots = [{'pos': pos[:,i], 'data': 1} for i in range(n)] + [{'pos': [0,0], 'data': 1}]
            scatter.addPoints(spots)
            plot.addItem(scatter)

            legendIcon = pg.PlotDataItem([15], [8 + (i * 21)], symbolSize=15, symbol = self.setting.colorSetting.getSymbol(i), symbolPen=self.setting.colorSetting.getPen(i), symbolBrush=self.setting.colorSetting.getBrush(i))
            legendIcon.setZValue(2)
            legendIcon.setParentItem(plot.getViewBox())

            self.scatters[i].tag = legendIcon

            legendLabel = pg.LabelItem("<b>" + self.setting.colorSetting.getLabel(i)+ "</b>")
            legendLabel.setPos(85, -4 +  (i * 21) )
            legendLabel.setParentItem(plot)

    def settingMenuAction(self):
        dialog = SettingDialog(self, self.setting)
        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        #dialog.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        dialog.setWindowFlags(QtCore.Qt.WindowTitleHint)
        
        dialog.setWindowOpacity(.95)
        dialog.show()
        dialog.exec_()

class SettingDialog(QDialog):
    def __init__(self, parentGui, setting):
        super().__init__()

        self.settingManager = SettingManager(parentGui, setting)
        self.setting = setting
        self.setting_copy = setting.copy()

        ### Init UI ####
        layout = QGridLayout()
        layout.setContentsMargins( 8, 8, 8, 8)

        self.setLayout(layout)
        self.resize(640, 270)

        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        scroll = QScrollArea()

        panelPlot = QWidget()
        layoutPlot = QVBoxLayout()
        panelPlot.setLayout(layoutPlot)
        scroll.setWidget(panelPlot)

        scroll.setWidgetResizable(True)

        tab2 = QWidget()
        tabs.addTab(scroll, "           Colors           ")
        tabs.addTab(tab2, "            Axis            ")

        for i in range(2):
            panelClass1 = QWidget()
            layoutPlot.addWidget(panelClass1)
            layoutPlot.setContentsMargins(0, 0, 0, 0)

            layoutClass1 = QHBoxLayout()
            panelClass1.setLayout(layoutClass1)

            layoutClass1.addStretch()

            lblClass1 = QLabel(self.setting.colorSetting.getLabel(i))
            lblClass1.setMaximumWidth(150)
            lblClass1.setStyleSheet("font-weight : bold")
            layoutClass1.addWidget(lblClass1)

            lblPen = QLabel("   Pen")
            layoutClass1.addWidget(lblPen)

            btnPen = QPushButton()
            btnPen.setFocusPolicy(QtCore.Qt.NoFocus)

            btnPen.clicked.connect(partial(self.btnPenClicked, btnPen, i))

            layoutClass1.addWidget(btnPen)
            
            btnPen.setStyleSheet("background-color : #19232d; border : 2px solid #" + self.setting.colorSetting.getPenHex(i) + " ; border-radius : 12px; width : 15px; height : 15px");

            lblBrush = QLabel("   Brush")
            layoutClass1.addWidget(lblBrush)

            btnBrush = QPushButton()
            btnBrush.setFocusPolicy(QtCore.Qt.NoFocus)

            layoutClass1.addWidget(btnBrush)
            btnBrush.clicked.connect(partial(self.btnBrushClicked, btnBrush, i))
            btnBrush.setStyleSheet("background-color : #" + self.setting.colorSetting.getBrushHex(i) + " ; border: none; border-radius : 12px; width : 19px; height : 19px");

            lblSymbol = QLabel("   Symbol")
            layoutClass1.addWidget(lblSymbol)

            comboSymbol = QComboBox()
            comboSymbol.setFocusPolicy(QtCore.Qt.NoFocus)
            comboSymbol.addItem("o");comboSymbol.addItem("t");comboSymbol.addItem("t1");comboSymbol.addItem("t2");comboSymbol.addItem("t3");comboSymbol.addItem("s");comboSymbol.addItem("p");comboSymbol.addItem("h");comboSymbol.addItem("star");comboSymbol.addItem("+");comboSymbol.addItem("d")

            comboSymbol.setCurrentText(self.setting.colorSetting.getSymbol(i))
            comboSymbol.currentTextChanged.connect(partial(self.on_combobox_changed, comboSymbol, i))

            layoutClass1.addWidget(comboSymbol)

            lblSize = QLabel("   Size")
            layoutClass1.addWidget(lblSize)
            txtSize = QLineEdit(str(self.setting.colorSetting.getSize(i)))
            txtSize.returnPressed.connect(partial(self.txtSizeEnterPressed, txtSize, i))
            txtSize.setFixedWidth(55)

            txtSize.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            layoutClass1.addWidget(txtSize)
            layoutClass1.addStretch()

        layoutPlot.addStretch()

        #### OK / Cancel Buttons.. #####
        panelButtons = QWidget()
        layoutButtons = QHBoxLayout()
        panelButtons.setLayout(layoutButtons)

        btnOK = QPushButton("     OK     ")
        btnOK.setFocusPolicy(QtCore.Qt.NoFocus)
        btnOK.clicked.connect(self.btnOkClicked)
        btnCancel = QPushButton("  Cancel  ")
        btnCancel.clicked.connect(self.btnCancelClicked)
        btnCancel.setFocusPolicy(QtCore.Qt.NoFocus)
        layoutButtons.addStretch()
        layoutButtons.setContentsMargins(0, 0, 0, 0)
        
        layoutButtons.addWidget(btnOK)
        layoutButtons.addWidget(btnCancel)
        layout.addWidget(panelButtons, 1, 0)

    def btnOkClicked(self):
        self.done(QtCore.Qt.WA_DeleteOnClose)

    def btnCancelClicked(self):
        self.done(QtCore.Qt.WA_DeleteOnClose)

    def txtSizeEnterPressed(self, sender, index):
        text = sender.text()

        try:
            newSize = int(text)
        except:
            return

        if self.setting.colorSetting.getSize(index) != newSize:
            self.settingManager.updateSize(index, newSize)

    def on_combobox_changed(self, sender, index):
        symbol = sender.currentText()

        ### Update Model ###
        self.settingManager.updateSymbol(index, symbol)

    def btnBrushClicked(self, sender, index):
        ## Check if it really needs to update ###
        currentColor = self.setting.colorSetting.getBrush(index)
        selectedColor = QColorDialog().getColor(initial = currentColor.color())
        if not selectedColor.isValid():
            return
        ########################################

        if selectedColor != currentColor.color():
            self.settingManager.updateBrush(index, sender,pg.mkBrush(selectedColor))
        else:
            print("same..")

    def btnPenClicked(self, sender, index):

        ## Check if it really needs to update ###
        currentColor = self.setting.colorSetting.getPen(index)
        selectedColor = QColorDialog().getColor(initial = currentColor.color())
        if not selectedColor.isValid():
            return
        ########################################

        if selectedColor != currentColor.color():
            self.settingManager.updatePen(index, sender,pg.mkPen(selectedColor))
        else:
            print("same..")

class Setting:
    def __init__(self):
        self.colorSetting = ColorSetting()
        self.axisSetting = AxisSetting()
    def copy(self):
        newObj = Setting()
        newObj.colorSetting = self.colorSetting.copy()
        newObj.axisSetting = self.axisSetting.copy()

class AxisSetting:
    def __init__(self):
        self.names = []
        self.mins = []
        self.maxs = []
        self.is_grid_enabled = False

    def addAxis(self, name, minimum, maximum):
        self.names.append(name)
        self.mins.append(minimum)
        self.maxs.append(maximum)

    def getName(self, index):
        return self.names[index]

    def getMin(self, index):
        return self.mins[index]

    def getMax(self, index):
        return self.maxs[index]

    def copy(self):
        newObj = AxisSetting()
        return newObj

default_pen = pg.mkPen({'color': "#000000", "width": 1})
default_brushes = [pg.mkBrush(0, 255, 0), pg.mkBrush(255, 0, 0), pg.mkBrush(0, 0, 255), pg.mkBrush(255, 167, 26), pg.mkBrush(85, 0, 127)]
default_symbols = ["o", "+", "p", "t", "t1", "t2", "t3", "s", "p", "h", "star", "+", "d"]
default_size = 14

class ColorSetting:
    def __init__(self):
        self.labels = []
        self.pens = []
        self.brushes = []
        self.symbols = []
        self.sizes = []

    def addClass(self, label, index):
        self.labels.append(label)
        self.pens.append(default_pen)
        self.brushes.append(default_brushes[index])
        self.symbols.append(default_symbols[index])
        self.sizes.append(default_size)

    def getLabel(self, index):
        return self.labels[index]

    def getPen(self, index):
        return self.pens[index]

    def getPenHex(self, index):
        return self.convertToHex(self.pens[index].color())

    def getBrushHex(self, index):
        return self.convertToHex(self.brushes[index].color())
        
    def convertToHex(self, color):
        red = hex(color.red())[2:]
        if len(red) == 1:
            red = str(0) + red
        green = hex(color.green())[2:]
        if len(green) == 1:
            green = str(0) + green
        blue = hex(color.blue())[2:]
        if len(blue) == 1:
            blue = str(0) + blue
        return  red + green + blue

    def getBrush(self, index):
        return self.brushes[index]

    def getSize(self, index):
        return self.sizes[index]

    def getSymbol(self, index):
        return self.symbols[index]

    def setPen(self, index, pen):
        self.pens[index] = pen

    def setBrush(self, index, brush):
        self.brushes[index] = brush

    def setSymbol(self, index, symbol):
        self.symbols[index] = symbol

    def setSize(self, index, size):
        self.sizes[index] = size

    def copy(self):
        newObj = ColorSetting()

        newObj.labels = self.labels[:]
        newObj.pens = self.pens[:]
        newObj.brushes = self.brushes[:]
        newObj.symbols = self.symbols[:]
        newObj.sizes = self.sizes[:
        return newObj

class SettingManager:
    def __init__(self, parentGui, setting):
        self.parentGui = parentGui
        self.setting = setting

    def updateBrush(self, index, sender, brush):
        # Model Update ##
        self.setting.colorSetting.setBrush(index, brush)

        # Button UI update ##
        sender.setStyleSheet("background-color : #" + self.setting.colorSetting.getBrushHex(index) + " ; border: none; border-radius : 12px; width : 19px; height : 19px");

        ## Scatter Plot Update ##
        self.parentGui.scatters[index].setBrush(brush)

        ## Update Legend ##
        self.parentGui.scatters[index].tag.setData(symbolBrush=brush, symbol = self.setting.colorSetting.getSymbol(index))
        self.parentGui.scatters[index].tag.updateItems()

    def updatePen(self, index, sender, pen):

        # Model Update ###
        self.setting.colorSetting.setPen(index, pen)

        ## Button UI update ###
        sender.setStyleSheet("background-color : #19232d; border : 2px solid #" + self.setting.colorSetting.getPenHex(index) + " ; border-radius : 12px; width : 15px; height : 15px");

        ## Scatter Plot Update ###
        self.parentGui.scatters[index].setPen(pen)

        ## Legend Update ###
        self.parentGui.scatters[index].tag.setData(symbolPen=pen, symbol = self.setting.colorSetting.getSymbol(index))
        self.parentGui.scatters[index].tag.updateItems()

    def updateSymbol(self, index, symbol):

        ## Update Model ####
        self.setting.colorSetting.setSymbol(index, symbol)

        ## Scatter Plot Update ###
        self.parentGui.scatters[index].setSymbol(symbol)

        ## Legend Update ###
        self.parentGui.scatters[index].tag.setData(symbol = symbol)
        self.parentGui.scatters[index].tag.updateItems()

    def updateSize(self, index, size):
        ### Update Model ####
        self.setting.colorSetting.setSize(index, size)

        ## Scatter Plot Update ##
        self.parentGui.scatters[index].setSize(size)

def plot():
    os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    win = MainWindow()
    win.showMaximized()
    app.exec_()

plot()

