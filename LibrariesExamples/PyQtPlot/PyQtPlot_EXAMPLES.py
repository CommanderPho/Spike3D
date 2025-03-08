# Set PyQt5 as the preferred binding before importing pyqtgraph
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

# Now import pyqtgraph
import pyphoplacecellanalysis.External.pyqtgraph as pg
# import pyphoplacecellanalysis.External.pyqtgraph.examples
from pyphoplacecellanalysis.External.pyqtgraph import examples
examples.run()
pg.exec()
