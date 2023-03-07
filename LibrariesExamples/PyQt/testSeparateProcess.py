import sys
import threading
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


def main():
      app = QtGui.QApplication(sys.argv)
      ex = Start_GUI()
      app.exec_()  #<---------- code blocks over here !

#After running the GUI, continue the rest of the application task

t = threading.Thread(target=main)
t.daemon = True
t.start()

doThis = do_Thread("doThis")
doThis.start()
doThat = do_Thread("doThat")
doThat.start()
