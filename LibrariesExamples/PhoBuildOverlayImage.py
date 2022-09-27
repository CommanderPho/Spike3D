import sys
from qtpy import QtCore, QtGui, QtWidgets
# from qtpy.QtCore import *
# from qtpy.QtGui import *

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    if len(app.arguments()) < 2:
        sys.stderr.write("Usage: %s <image file> <overlay file>\n" % sys.argv[0])
        sys.exit(1)
    
    image = QtGui.QImage(app.arguments()[1])
    if image.isNull():
        sys.stderr.write("Failed to read image: %s\n" % app.arguments()[1])
        sys.exit(1)
    
    overlay = QtGui.QImage(app.arguments()[2])
    if overlay.isNull():
        sys.stderr.write("Failed to read image: %s\n" % app.arguments()[2])
        sys.exit(1)
    
    if overlay.size() > image.size():
        overlay = overlay.scaled(image.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    
    painter = QtGui.QPainter()
    painter.begin(image)
    painter.drawImage(0, 0, overlay)
    painter.end()
    
    label = QtWidgets.QLabel()
    label.setPixmap(QtGui.QPixmap.fromImage(image))
    label.show()
    
    sys.exit(app.exec_())
    
    