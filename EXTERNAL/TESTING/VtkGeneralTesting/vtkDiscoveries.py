import ipywidgets as widgets
# from PyQt5 import QtWidgets, uic
from pyvistaqt import QtInteractor, MainWindow
# from pyqt6 import QApplication
from IPython.external.qt_for_kernel import QtGui
from PyQt5.QtWidgets import QApplication



# from PhoGui import vtk_ui
# from PhoGui.vtk_ui.manager import get_manager, manager_exists, delete_manager
from pyvista import _vtk
# import vtk
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkFiltersCore import vtkElevationFilter
from vtkmodules.vtkFiltersModeling import vtkBandedPolyDataContourFilter
from vtkmodules.vtkFiltersSources import vtkConeSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
nc = vtkNamedColors()

from pyvista.plotting.tools import parse_color, FONTS
## Add placemap toggle checkboxes GUI:
# for i, an_extant_checkbox_widget_actor in enumerate(checkboxWidgetActors):
#     # print(an_extant_checkbox_widget_actor.GetRepresentation().GetBalloon()) # vtkBalloonRepresentation
    
#     # print(an_extant_checkbox_widget_actor.GetRepresentation().GetBalloon().GetImageProperty()) # vtkProperty2D 
#     print(an_extant_checkbox_widget_actor.GetRepresentation().GetBalloon().BalloonImage) # vtkProperty2D 
    
#     # print(an_extant_checkbox_widget_actor.GetRepresentation()) # vtkTexturedButtonRepresentation2D 
#     # print(an_extant_checkbox_widget_actor.GetRepresentation().GetBounds()) # RETURNS NONE
#     # print(an_extant_checkbox_widget_actor.GetRepresentation().GetProperty()) # vtkProperty2D 
#     # print(an_extant_checkbox_widget_actor.GetRepresentation().GetActors()) # vtkProperty2D 
#     # print(an_extant_checkbox_widget_actor.GetInteractor()) # vtkGenericRenderWindowInteractor   


# Camera:
def debug_print_camera(camera):
    print('camera: position: {}'.format(camera.position))
    
# camera = pv.Camera()
camera = p.camera
# print('camera: {}'.format(camera))

debug_print_camera(camera) # (419.82010625260347, 397.37481091874116, 278.0261261309909)
# top: (142.89398002624512, 120.44868469238281, 480.75012047191734)
# pl.camera.position

camera.position = np.append(midpoint, [0.0])

# actor = p.add_points(np.append(centroid_point, [0.0]), render_points_as_spheres=True, point_size=10.0, color='cyan')
actor = p.add_points(np.append(midpoint, [0.0]), render_points_as_spheres=True, point_size=10.0, color='cyan')
p.remove_actor(actor)