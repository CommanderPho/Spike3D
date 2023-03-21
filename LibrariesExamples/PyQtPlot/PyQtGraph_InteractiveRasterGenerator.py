# PyQtGraph_InteractiveRasterGenerator

import sys
# import pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class RasterPlotWidget(QtWidgets.QWidget):
    def __init__(self, num_rows):
        super().__init__()
        self.num_rows = num_rows
        self.color_map = pg.ColorMap(
            [0, num_rows], pg.mkColor(0, 0, 0), pg.mkColor(255, 255, 255)
        )

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground((255, 255, 255))
        self.plot_widget.show()

        self.plots = []
        for row in range(self.num_rows):
            p = self.plot_widget.addPlot(row=row, col=0)
            p.showGrid(x=True, y=True)
            p.setLabel('left', f'Row {row}')
            p.setXRange(0, 10)
            p.setYRange(-0.5, 0.5)
            p.plot([], [], pen=None, symbol='o', symbolSize=10,
                   symbolBrush=self.color_map.map(row, 'qcolor'))

            self.plots.append(p)

        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_click)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def on_mouse_click(self, event):
        if not event.double():
            pos = self.plot_widget.getViewBox().mapSceneToView(event.pos())
            x, y = pos.x(), pos.y()
            for plot in self.plots:
                if plot.sceneBoundingRect().contains(event.pos()):
                    row = plot.plotItem.row
                    symbol = pg.QtGui.QGraphicsEllipseItem(
                        x - 0.1, row - 0.025, 0.2, 0.05
                    )
                    symbol.setPen(pg.mkPen(None))
                    symbol.setBrush(pg.mkBrush(self.color_map.map(row, 'qcolor')))
                    plot.addItem(symbol)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = RasterPlotWidget(5)
    window.show()
    sys.exit(app.exec_())
