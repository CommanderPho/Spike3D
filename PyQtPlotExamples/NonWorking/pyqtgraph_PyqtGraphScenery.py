#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Javier Martinez Garcia, Marzo 2014


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import math as mt
import numpy as np

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('Scenery')
w.setCameraPosition(distance=100)
g = gl.GLGridItem()
g.scale(10.0,10.0,0.1)
w.addItem(g)

#Create Origin
Origin=np.array([
        (4,0,0),
        (0,0,0),
        (0,4,0),
        (0,0,0),
        (0,0,4)
        ])
orgn=gl.GLLinePlotItem(pos=Origin)
w.addItem(orgn)

#Create Ground ("suelo" in spanish :) )

lsuelo=100.0
Suelovertex=np.array([
    [lsuelo,lsuelo,0],
    [-lsuelo,lsuelo,0],
    [-lsuelo,-lsuelo,0],
    [lsuelo,-lsuelo,0]
    ])
Sueloface=np.array([
  [0,1,2],
  [2,3,0]
  ])
r=210.0/256.0
g=180.0/256.0
b=140.0/256.0
s=0.0
Suelocolors=np.array([
    (r,g,b,s),
    (r,g,b,s),
    (r,g,b,s),
    (r,g,b,s)
    ])
Suelo=gl.GLMeshItem(vertexes=Suelovertex, faces=Sueloface, faceColors=Suelocolors, smooth=False)
w.addItem(Suelo)

def Buildings(VE):                      #Creates "Buildings" given by VE
  for i in range(len(VE)):
    
    Name=str(VE[i][0])
    
    B=float(VE[i][1])
    P=float(VE[i][2])
    A=float(VE[i][3])
    
    Excero=float(VE[i][4][0])
    Eycero=float(VE[i][4][1])
    Ezcero=float(VE[i][4][2])
    
    #Eax=float(VE[i][5][0])
    #Eay=float(VE[i][5][1])
    #Eaz=float(VE[i][5][2])
    
    r=float(VE[i][5][0])/256.0
    g=float(VE[i][5][1])/256.0
    b=float(VE[i][5][2])/256.0
    s=float(VE[i][5][3])/256.0
    
    Edvertex=np.array([
        (Excero,Eycero,Ezcero),
        (B+Excero,Eycero,Ezcero),
        (B+Excero,P+Eycero,Ezcero),
        (Excero,P+Eycero,Ezcero),
        (Excero,Eycero,A+Ezcero),
        (B+Excero,Eycero,A+Ezcero),
        (B+Excero,P+Eycero,A+Ezcero),
        (Excero,P+Eycero,A+Ezcero),
        ])
    
    Edface=np.array([
        (1,0,5),
        (5,4,0),
        (4,7,0),
        (3,7,0),
        (7,6,3),
        (6,2,3),
        (5,1,6),
        (1,6,2),
        (5,6,4),
        (6,7,4)
        ])
    
    Edcolors=np.array([
        (r,g,b,s),
        (r,g,b,s),
        (r,g,b,s),
        (r,g,b,s),
        (r,g,b,s),
        (r,g,b,s),
        (r,g,b,s),
        (r,g,b,s),
        (r-(10.0/256.0),g-(10.0/256.0),b-(10.0/256.0),s),
        (r-(10.0/256.0),g-(10.0/256.0),b-(10.0/256.0),s)
        ])
    
    Name=gl.GLMeshItem(vertexes=Edvertex, faces=Edface, faceColors=Edcolors, smooth=False, drawEdges=True)
    w.addItem(Name)




    
def Captadores(VC):                     #Creates "solar panel" given by VC
  for i in range(len(VC)):
    
    Name=str(VC[i][0])
    
    B=float(VC[i][1])
    A=float(VC[i][2])
    
    xcero=float(VC[i][3][0])
    ycero=float(VC[i][3][1])
    zcero=float(VC[i][3][2])
    
    ax=mt.radians(float(VC[i][4][0]))
    ay=mt.radians(float(VC[i][4][1]))
    az=mt.radians(float(VC[i][4][2]))
    
    r=float(VC[i][5][0])
    g=float(VC[i][5][1])
    b=float(VC[i][5][2])
    s=float(VC[i][5][3])
    
    Cvertex=np.array([
      [0,0,0,0],
      [0,mt.cos(ax)*A,mt.sin(ax)*A,0],
      [B,mt.cos(ax)*A,mt.sin(ax)*A,0],
      [B,0,0,0]
      ])
    
    Cposvector=np.array([
      [xcero,ycero,zcero,0],
      [xcero,ycero,zcero,0],
      [xcero,ycero,zcero,0],
      [xcero,ycero,zcero,0]
      ])
    
    Rz=np.array([
      [mt.cos(az),mt.sin(az),0,0],
      [-mt.sin(az),mt.cos(az),0,0],
      [0,0,1,0],
      [0,0,0,1]
      ])
    Cvertex=np.dot(Cvertex,Rz) # Rotacion en Z  
    
    Cvertex=Cvertex+Cposvector
    
    Cvertex=np.delete(Cvertex,3,1)
    
    Cface=np.array([
      [0,1,2],
      [2,3,0]
      ])
    
    Colors=np.array([
	(r,g,b,s),
	(r,g,b,s),
	(r,g,b,s),
	(r,g,b,s)])
    
    Name=gl.GLMeshItem(vertexes=Cvertex, faces=Cface, faceColors=Colors, smooth=False)
    w.addItem(Name)


#Example VE (name, length, width, heigh, (positionx, y, z),(colorr, g, b, s))

VE=(('E1',10,10,7,(0,0,0),(192,192,192,200)),
('E2',20,20,14,(-10,18,0),(150,180,192,200)),
('E2.1',5,5,3,(-8,24,14),(192,192,192,250)),
('E3',12,12,22,(20,-10,0),(192,192,192,100)),
('E4',30,30,6,(20,18,0),(192,192,192,100)),
('E5',15,7,5,(-5,-17,0),(110,192,192,100)),
('E6',5,5,4,(-5,-17,5),(192,192,192,100)),
('E7',10,20,25,(-20,-10,0),(192,192,192,100)),
('E8',10,30,15,(40,-15,0),(120,120,120,100))
)

#Example VC    (name, Length, Width, (posx,posy,posz),(alphax,alphay,alphaz),(colorr,g,b,s))

VC=(('LOL', 3,1,(4.5,2,7),(110,0,0),(0.5,1,0.5,0.2)),
  ('LOL1', 3,1,(4.5,4,7),(120,0,20),(0.5,1,0.5,0.2)),
  ('LOL2', 3,1,(4.5,6,7),(130,0,30),(0.5,1,0.5,0.2)),
  ('LOL3', 3,1,(4.5,8,7),(140,0,90),(0.5,1,0.5,0.2)))


#Create the objects:

Buildings(VE)
Captadores(VE)

