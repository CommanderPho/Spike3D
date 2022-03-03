import numpy as np
import matplotlib as mpl
# from OpenGL.GL import *  # noqa
from pyqtgraph.Qt import QtCore, QtGui
# %gui qt5 
import pyqtgraph as pg
from matplotlib import cm
pg.mkQApp()

## make a widget for displaying 3D objects
import pyqtgraph.opengl as gl
view = gl.GLViewWidget()


## create three grids, add each to the view
xgrid = gl.GLGridItem()
ygrid = gl.GLGridItem()
zgrid = gl.GLGridItem()

view.addItem(xgrid)
view.addItem(ygrid)
view.addItem(zgrid)

## rotate x and y grids to face the correct direction
xgrid.rotate(90, 0, 1, 0)
ygrid.rotate(90, 1, 0, 0)

## scale each grid differently
xgrid.scale(0.2, 0.1, 0.1)
ygrid.scale(0.2, 0.1, 0.1)
zgrid.scale(0.1, 0.2, 0.1)

PlotItem = gl.GLLinePlotItem(pos=np.array([[0,0,0],[1,0,0]]), color=pg.glColor((255, 140, 140, 255)), width=5, antialias=True)
view.addItem(PlotItem)
PlotItem = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,1,0]]), color=pg.glColor((140, 255, 140, 255)), width=5, antialias=True)
view.addItem(PlotItem)
PlotItem = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,0,1]]), color=pg.glColor((140, 140, 255, 255)), width=5, antialias=True)
view.addItem(PlotItem)

# Example 4:
# wireframe
 
md = gl.MeshData.sphere(rows=20, cols=20)
m4 = gl.GLMeshItem(meshdata=md, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0.3,0.3,0.3,1))
m4.translate(0,0,0)
view.addItem(m4)

# view.show()
class _3DAlignmentImageItem():
    def __init__(self, rho, thetas, phis, view):
        self.thetas = thetas
        self.phis = phis
        self.d_th = abs(thetas[1]-thetas[0])/2
        self.d_phi = abs(phis[1]-phis[0])/2
        self.rho = rho
        self.parent_view = view
        self.items = []
        
    def del_items(self):
        for item in self.items:
            try:
                self.parent_view.removeItem(item)
            except:
                print('No items to remove')
                return
    
    def make_face(self, index):
        theta, phi = self.thetas[index[0]], self.phis[index[1]]
        vertices = []
        for i in [1,-1]:
            for j in [1,-1]:
                _theta = theta + self.d_th*i
                _phi = phi + self.d_phi*j
                x1 = self.rho * np.sin(_theta) * np.cos(_phi)
                y1 = self.rho * np.sin(_theta) * np.sin(_phi)
                z1 = self.rho * np.cos(_theta)
                vertices.append([x1,y1,z1])
        faces = [[0,1,2],
                 [1,2,3]]
        
        return vertices, faces
                
    def setImage(self, matrix, levels):
        self.image = matrix
        cb_min, cb_max = (levels)
        norm = mpl.colors.Normalize(vmin=cb_min, vmax=cb_max)
        cmap = cm.viridis
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        thet = self.thetas
        ph = self.phis
        mi_vertices = []
        mi_faces = []
        mi_colors = []
        
        n = 0
        for i in range(len(thet)):
            for j in range(len(ph)):
                verts, faces = self.make_face((i,j))
                c = m.to_rgba(matrix[i,j])
                colors = np.array([[*c],[*c]])
                
                mi_vertices.extend(verts)
                mi_faces.extend(np.asarray(faces)+4*n)
                mi_colors.extend(colors)
                n += 1
                ## Mesh item will automatically compute face normals.
        m1 = gl.GLMeshItem(vertexes=np.asarray(mi_vertices), faces=np.asarray(mi_faces), faceColors=np.asarray(mi_colors), smooth=False, computeNormals=False)
        m1.translate(0, 0, 0)
        m1.setGLOptions('additive')
        self.parent_view.addItem(m1)
        self.items.append(m1)

class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, X, Y, Z, text):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()
        
    # def renderText(self,  x, y, z, text, font):
        # pg.
        # GL.glGetError()
        # qt.QGLWidget.renderText(self, x, y, z, text, font)
        # GL.glGetError()

    def paint(self):
        self.GLViewWidget.setBackgroundColor('w')
        # self.GLViewWidget.qglColor(QtCore.Qt.white)
        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text, QtGui.QFont('Arial', 12, QtGui.QFont.Medium))

class _3DAxisItem():
    def __init__(self, rho, thetas, phis, n_ticks, view, axis=False):
        self.n_ticks = n_ticks
        self.thetas = thetas
        self.phis = phis
        self.rho = rho
        self.parent_view = view
        self.items = []
        if axis:
            self.make_axis()
        self.make_ticks()
    
    def del_items(self):
        for item in self.items:
            try:
                self.parent_view.removeItem(item)
            except:
                print('No items to remove')
                return
                
    def make_axis(self):
        thet = np.linspace(self.thetas[0], self.thetas[-1], 50)
        ph = np.linspace(self.phis[0], self.phis[-1], 50)
        v = self.rho
        d_th = abs(self.thetas[0] - self.thetas[1])*0.5
        d_ph = abs(self.phis[0] - self.phis[1])*0.75
        
        thet -= d_th
        ph -=d_ph

        x0 = v * np.sin(thet[0]) * np.cos(ph[0])
        y0 = v * np.sin(thet[0]) * np.sin(ph[0])
        z0 = v * np.cos(thet[0])
        for j in range(len(ph)):
            theta = thet[0]
            phi = ph[j]

            x = v * np.sin(theta) * np.cos(phi)
            y = v * np.sin(theta) * np.sin(phi)
            z = v * np.cos(theta)

            PlotItem = gl.GLLinePlotItem(pos=np.array([[x0,y0,z0],[x,y,z]]), color=pg.glColor((255, 255, 255, 255)), width=5, antialias=True)
            self.parent_view.addItem(PlotItem)
            self.items.append(PlotItem)
            x0,y0,z0 = x,y,z

        x0 = v * np.sin(thet[0]) * np.cos(ph[0])
        y0 = v * np.sin(thet[0]) * np.sin(ph[0])
        z0 = v * np.cos(thet[0])
        for i in range(len(thet)):
            theta = thet[i]
            phi = ph[0]

            x = v * np.sin(theta) * np.cos(phi)
            y = v * np.sin(theta) * np.sin(phi)
            z = v * np.cos(theta)

            PlotItem = gl.GLLinePlotItem(pos=np.array([[x0,y0,z0],[x,y,z]]), color=pg.glColor((255, 255, 255, 255)), width=5, antialias=True)
            self.parent_view.addItem(PlotItem)
            self.items.append(PlotItem)
            x0,y0,z0 = x,y,z
    
    def make_ticks(self):
        thet = np.linspace(self.thetas[0], self.thetas[-1], self.n_ticks)
        ph = np.linspace(self.phis[0], self.phis[-1], self.n_ticks)
        v = self.rho
        d_th = (self.thetas[1] - self.thetas[0])*2
        d_ph = (self.phis[1] - self.phis[0])*2.5
        x0 = v * np.sin(thet[0]) * np.cos(ph[0])
        y0 = v * np.sin(thet[0]) * np.sin(ph[0])
        z0 = v * np.cos(thet[0])
        for j in range(len(ph)):
            theta = thet[0]
            phi = ph[j]

            dx = v * np.sin(theta-d_th) * np.cos(phi)
            dy = v * np.sin(theta-d_th) * np.sin(phi)
            dz = v * np.cos(theta-d_th)

            txt = CustomTextItem(dx,dy,dz,f'{phi/np.pi:.2f}π')
            txt.setGLViewWidget(view)
            self.parent_view.addItem(txt)
            self.items.append(txt)

        x0 = v * np.sin(thet[0]) * np.cos(ph[0])
        y0 = v * np.sin(thet[0]) * np.sin(ph[0])
        z0 = v * np.cos(thet[0])
        for i in range(len(thet)):
            theta = thet[i]
            phi = ph[0]

            dx = v * np.sin(theta) * np.cos(phi-d_ph)
            dy = v * np.sin(theta) * np.sin(phi-d_ph)
            dz = v * np.cos(theta)

            txt = CustomTextItem(dx,dy,dz,f'{theta/np.pi:.2f}π')
            txt.setGLViewWidget(view)
            self.parent_view.addItem(txt)
            self.items.append(txt)

size = 30
thet = np.linspace(0,np.pi/4,size)
ph = np.linspace(0,np.pi,size)
matrix = np.random.random((size,size))
# matrix = makeGaussian(size, size/3)
rho = 0.7
try:
    axis.del_items()
    image.del_items()
except:
    print('Oh')
axis = _3DAxisItem(rho, thet, ph, 5, view, False)
image = _3DAlignmentImageItem(rho, thet, ph, view)
image.setImage(matrix, (0,1))
view.show()