#https://pythonprogramminglanguage.com/pyqt-checkbox/
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QCheckBox, QWidget
from PyQt5.QtCore import QSize    

class ExampleWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(140, 40))    
        self.setWindowTitle("Checkbox") 

        self.b = QCheckBox("Awesome?",self)
        self.b.stateChanged.connect(self.clickBox)
        self.b.move(20,20)
        self.b.resize(320,40)

    def clickBox(self, state):
        if state == QtCore.Qt.Checked:
            print('Checked')
        else:
            print('Unchecked')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = ExampleWindow()
    mainWin.show()
    sys.exit( app.exec_() )