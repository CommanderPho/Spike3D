import sys
import pathlib
from qtpy import QtCore, QtGui, QtWidgets
from qtpy import QtSvg
# from qtpy.QtCore import *
# from qtpy.QtGui import *


def build_svg_viewer():
    """ from https://www.reddit.com/r/learnpython/comments/byurin/i_need_to_display_a_svg_image_in_pyqt5_or_convert/
    """
    svg_viewer = QtSvg.QSvgWidget()
    svg_viewer.load('C:/Windows/Temp/tubesheetsvgpreview.svg')
    svg_viewer.setGeometry(QtCore.QRect(0,0,600,600))
    # svg_viewer.setAlignment(self.centralwidget)
    
    # set the layout to centralWidget
    # lay = QtWidgets.QVBoxLayout(self.centralwidget)
    # add the viewer to the layout
    # lay.addWidget(svg_viewer)
    return svg_viewer
    
    
def build_overlay_viewer(image_path, overlay_path):
    """ 
    
    Usage:
        image, overlay, painter, label = build_overlay_viewer(image_path=app.arguments()[1], overlay_path=app.arguments()[2])
    """
    image = QtGui.QImage(image_path)
    if image.isNull():
        sys.stderr.write("Failed to read image: %s\n" % image_path)
        sys.exit(1)
    
    overlay = QtGui.QImage(overlay_path)
    if overlay.isNull():
        sys.stderr.write("Failed to read image: %s\n" % overlay_path)
        sys.exit(1)

    if overlay.size().width() > image.size().width():
        overlay = overlay.scaled(image.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    
    painter = QtGui.QPainter()
    painter.begin(image)
    painter.drawImage(0, 0, overlay)
    painter.end()
    
    label = QtWidgets.QLabel()
    label.setPixmap(QtGui.QPixmap.fromImage(image))
    label.show()
    
    return image, overlay, painter, label
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    if len(app.arguments()) < 2:
        sys.stderr.write("Usage: %s <image file> <overlay file>\n" % sys.argv[0])
        sys.exit(1)
    
    
    image, overlay, painter, label = build_overlay_viewer(image_path=app.arguments()[1], overlay_path=app.arguments()[2])
    # image = QtGui.QImage(app.arguments()[1])
    # if image.isNull():
    #     sys.stderr.write("Failed to read image: %s\n" % app.arguments()[1])
    #     sys.exit(1)
    
    # overlay = QtGui.QImage(app.arguments()[2])
    # if overlay.isNull():
    #     sys.stderr.write("Failed to read image: %s\n" % app.arguments()[2])
    #     sys.exit(1)

    # if overlay.size().width() > image.size().width():
    #     overlay = overlay.scaled(image.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    
    # painter = QtGui.QPainter()
    # painter.begin(image)
    # painter.drawImage(0, 0, overlay)
    # painter.end()
    
    # label = QtWidgets.QLabel()
    # label.setPixmap(QtGui.QPixmap.fromImage(image))
    # label.show()
    
    sys.exit(app.exec_())
    
    
# "c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Icons\actions\chart-up-color.png"

## Overlays:

### .svg versions:
# "C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\EXTERNAL\Design\Icons\Potential\Overlays\svg"
# 'noise-control-off-add.svg'

### .png versions:
# "C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\EXTERNAL\Design\Icons\Potential\Overlays\png\1x"
# 'noise-control-off-add.png'


# "C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\EXTERNAL\Design\Icons\Potential\Overlays\svg\noise-control-off-add.svg"
# 'noise-control-off-new.svg'
# 'noise-control-off-delete.svg'
# 'noise-control-off-add.svg'
# 'noise-control-off-remove.svg'

# C:\Users\pho\repos\PhoPy3DPositionAnalysis2021>C:/Users/pho/miniconda3/envs/phoviz_ultimate/python.exe c:/Users/pho/repos/PhoPy3DPositionAnalysis2021/LibrariesExamples/PhoBuildOverlayImage.py "c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Icons\actions\chart-up-color.png" "C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\EXTERNAL\Design\Icons\Potential\Overlays\svg\noise-control-off-add.svg"


### Overlays:
# "C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Icons\overlay_modifiers"
# ['exclamation.png'
# 'exclamation-red.png'
# 'exclamation-small.png'
# 'exclamation-small-red.png'
# 'eye.png'
# 'eye--arrow.png'
# 'eye-close.png'
# 'eye--exclamation.png'
# 'eye--minus.png'
# 'eye--pencil.png'
# 'eye--plus.png'
# 'minus.png'
# 'minus-button.png'
# 'minus-small.png'
# 'new.png'
# 'plus.png'
# 'plus-button.png'
# 'plus-small.png']


# "C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Resources\Icons\overlay_modifiers\new.png"
# exclamation.png
# exclamation-red.png
# exclamation-small.png
# exclamation-small-red.png
# eye.png
# eye--arrow.png
# eye-close.png
# eye--exclamation.png
# eye--minus.png
# eye--pencil.png
# eye--plus.png
# minus.png
# minus-button.png
# minus-small.png
# new.png
# plus.png
# plus-button.png
# plus-small.png





