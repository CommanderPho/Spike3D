"""Module dedicated to custom VTK widgets."""

import numpy as np
import collections

import pyvista
from pyvista import _vtk
from pyvista.utilities import (NORMALS, generate_plane, get_array,
                               try_callback, get_array_association)
from pyvista.plotting.tools import parse_color, FONTS
# from ConfigurationFunctions import SIGNAL

# from PhoGui import vtk_ui
# import vtk_ui


class MultilineTextBuffer:
    """ A fixed-length circular text buffer class which allows the user to add lines to the end of the buffer and loses the oldest ones once full.
        Useful for implementing a scrolling/overflowing text console or printing debug messages within a fixed space.
        Usage:
            test_buffer = MultilineTextBuffer()
            print(test_buffer)
            test_buffer.add_lines_to_buffer(['line 1', 'line 2', 'line 3', 'line 4', 'line 5', 'line 7', 'line 8'])
            print(test_buffer)
            test_buffer.add_lines_to_buffer(['line 9'])
            print(test_buffer)
            test_buffer.add_lines_to_buffer(['line 10'])
            print(test_buffer)
            print('test_buffer.joined_text: {}'.format(test_buffer.joined_text))
    """
    def __init__(self, max_num_lines=5, is_debug=False):
        self.max_num_lines = max_num_lines # the maximum number of lines to store in the buffer
        self.is_debug=is_debug        
        # Allocate the circular buffer that indicates which element is active
        self._circular_buffer = collections.deque(np.full([max_num_lines,], ''), maxlen=self.max_num_lines)
    
    @property
    def buffer_text_strings(self):
        """The buffer_text_strings property."""
        return [item for item in self._circular_buffer]
    @property    
    def joined_text(self):
        controls_helper_text = '\n'.join(self._circular_buffer)
        # controls_helper_text = '\n'.join(self._circular_buffer)
        return controls_helper_text
    
    def add_line_to_buffer(self, new_line):
        """Adds the new_line to the end of the circular buffer"""
        self._circular_buffer.append(new_line)

    def add_lines_to_buffer(self, iterable_lines):
        """Adds the iterable_lines to the end of the circular buffer"""
        for a_line in iterable_lines:
            self.add_line_to_buffer(a_line)        
            
    def __repr__(self) -> str:
        return f"<MultilineTextBuffer: max_num_lines: {self.max_num_lines}>: {self.buffer_text_strings}"
    def __str__(self) -> str:
        return f"<MultilineTextBuffer: max_num_lines: {self.max_num_lines}>: {self.buffer_text_strings}"
    
    

class MultilineTextConsoleWidget(object):
    """ A text widget that renders in the viewport to display a fixed-length circular text buffer class which allows the user to add lines to the end of the buffer and loses the oldest ones once full.
        Useful for implementing a scrolling/overflowing text console or printing debug messages within a fixed space.
        Usage:
            # Adds a multi-line debug console to the GUI for output logging:
            debug_console_widget = MultilineTextConsoleWidget(pActiveTuningCurvesPlotter)
            debug_console_widget.add_line_to_buffer('test log')
    """
    def __init__(self, p, max_num_lines=5, name='lblDebugLoggingConsole', is_debug=False):
        self.is_debug=is_debug        
        # Allocate the internal text buffer object
        self._text_buffer = MultilineTextBuffer(max_num_lines=max_num_lines, is_debug=is_debug)
        self._debug_logging_console_label_actor = PhoWidgetHelper.perform_add_button_text_label(p, self._text_buffer.joined_text, (0.5, 0.0), font_size=6, color=[1, 1, 1], shadow=False, name=name, viewport=True)
        
    @property
    def max_num_lines(self):
        return self._text_buffer.max_num_lines
    @property
    def buffer_text_strings(self):
        return self._text_buffer.buffer_text_strings
    @property    
    def joined_text(self):
        return self._text_buffer.joined_text
    
    def add_line_to_buffer(self, new_line):
        """Adds the new_line to the end of the circular buffer"""
        self._text_buffer.add_line_to_buffer(new_line)
        self.on_update_buffer()
        
    def add_lines_to_buffer(self, iterable_lines):
        """Adds the iterable_lines to the end of the circular buffer"""
        for a_line in iterable_lines:
            self._text_buffer.add_lines_to_buffer(a_line)      
        self.on_update_buffer()
            
    def on_update_buffer(self):
        """Called when the internal buffer object changes to update the label actor with the new text"""
        self._debug_logging_console_label_actor.SetInput(self.joined_text)
        
    def __repr__(self) -> str:
        return f"<MultilineTextConsoleWidget: max_num_lines: {self.max_num_lines}>: {self.buffer_text_strings}"
    def __str__(self) -> str:
        return f"<MultilineTextConsoleWidget: max_num_lines: {self.max_num_lines}>: {self.buffer_text_strings}"
    
    
 
 
class PhoWidgetHelper:
    """An internal class to manage widgets.

    It also manages and other helper methods involving widgets.

    """
    
    def add_custom_button_widget(self, callback, value=False,
                                   position=(10., 10.), size=50, border_size=5,
                                   color_on='blue', color_off='grey',
                                   background_color='white'):
        # if not hasattr(self, "button_widgets"):
        #     self.button_widgets = []
        # call the static method:
        button_widget = PhoWidgetHelper.perform_add_custom_button_widget(self, callback, value=value,
                            position=position, size=size, border_size=border_size,
                            color_on=color_on, color_off=color_off,
                            background_color=background_color)
        
        # self.button_widgets.append(button_widget)
        return button_widget
        
    
        
    @staticmethod
    def perform_add_custom_button_widget(p, callback, value=False,
                            position=(10., 10.), size=50, border_size=5,
                            color_on='blue', color_off='grey',
                            background_color='white', render=True):
        """Add a custom button widget to the scene.

        This is useless without a callback function. You can pass a callable
        function that takes a single argument, the state of this button widget
        and performs a task with that value.

        Parameters
        ----------
        callback : callable
            The method called every time the button is clicked. This should take
            a single parameter: the bool value of the button.

        value : bool, optional
            The default state of the button.

        position : tuple(float), optional
            The absolute coordinates of the bottom left point of the button.

        size : int, optional
            The size of the button in number of pixels.

        border_size : int, optional
            The size of the borders of the button in pixels.

        color_on : str or 3 item list, optional
            The color used when the button is checked. Default is ``'blue'``.

        color_off : str or 3 item list, optional
            The color used when the button is not checked. Default is ``'grey'``.

        background_color : str or sequence, optional
            The background color of the button. Default is ``'white'``.

        Returns
        -------
        vtk.vtkButtonWidget
            The VTK button widget configured as a checkbox button.

        """
        if not hasattr(p, "button_widgets"):
            p.button_widgets = []
        
        def _create_button(color1, color2, color3, dims=[size, size, 1]):
            color1 = np.array(parse_color(color1)) * 255
            color2 = np.array(parse_color(color2)) * 255
            color3 = np.array(parse_color(color3)) * 255

            n_points = dims[0] * dims[1]
            button = pyvista.UniformGrid(dims)
            arr = np.array([color1] * n_points).reshape(dims[0], dims[1], 3)  # fill with color1
            arr[1:dims[0]-1, 1:dims[1]-1] = color2  # apply color2
            arr[
                border_size:dims[0]-border_size,
                border_size:dims[1]-border_size
            ] = color3  # apply color3
            button.point_data['texture'] = arr.reshape(n_points, 3).astype(np.uint8)
            return button
        

        # def create_button_text_label(text, position, font_size=18, color=None, font=None, shadow=False, name=None, viewport=False):
        # 	# Create the TextActor
        # 	text_actor = _vtk.vtkTextActor()
        # 	text_actor.SetInput(text)
        # 	text_actor.SetPosition(position)
        # 	if viewport:
        # 		text_actor.GetActualPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        # 		text_actor.GetActualPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        # 	text_actor.GetTextProperty().SetFontSize(int(font_size * 2))
            
        # 	text_actor.GetTextProperty().SetColor(color)
        # 	text_actor.GetTextProperty().SetFontFamily(FONTS[font].value)
        # 	text_actor.GetTextProperty().SetShadow(shadow)
        # 	self.add_actor(text_actor, reset_camera=False, name=name, pickable=False)
        
        # 	# ## Viewport Stuff:
        # 	# # Create the text representation. Used for positioning the text_actor
        # 	# text_representation = _vtk.vtkTextRepresentation()
        # 	# # text_representation.GetPositionCoordinate().SetValue(0.15, 0.15)
        # 	# # text_representation.GetPosition2Coordinate().SetValue(0.7, 0.2)
        # 	# text_representation.GetPositionCoordinate().SetValue(0.15, 0.15)
        # 	# text_representation.GetPosition2Coordinate().SetValue(0.7, 0.2)


        # 	# # Create the TextWidget
        # 	# # Note that the SelectableOff method MUST be invoked!
        # 	# # According to the documentation :
        # 	# #
        # 	# # SelectableOn/Off indicates whether the interior region of the widget can be
        # 	# # selected or not. If not, then events (such as left mouse down) allow the user
        # 	# # to 'move' the widget, and no selection is possible. Otherwise the
        # 	# # SelectRegion() method is invoked.
        # 	# text_widget = _vtk.vtkTextWidget()
        # 	# text_widget.SetRepresentation(text_representation)

        # 	# text_widget.SetInteractor(self.iren.interactor)
        # 	# text_widget.SetTextActor(text_actor)
        # 	# # text_widget.SetCurrentRenderer(self.renderer) # maybe needed?
        # 	# text_widget.SelectableOff()
        
        button_on = _create_button(color_on, background_color, color_on)
        button_off = _create_button(color_on, background_color, color_off)

        bounds = [
            position[0], position[0] + size,
            position[1], position[1] + size,
            0., 0.
        ]
        
        half_size = size / 2
        center_point = [(position[0] + half_size), (position[1] + half_size)]
        
        button_rep = _vtk.vtkTexturedButtonRepresentation2D()
        button_rep.SetNumberOfStates(2)
        button_rep.SetState(value)
        button_rep.SetButtonTexture(0, button_off)
        button_rep.SetButtonTexture(1, button_on)
        
        button_rep.SetPlaceFactor(1)
        button_rep.PlaceWidget(bounds)

        button_widget = _vtk.vtkButtonWidget()
        button_widget.SetInteractor(p.iren.interactor)
        button_widget.SetRepresentation(button_rep)
        button_widget.SetCurrentRenderer(p.renderer)
        if render:
            button_widget.On()
        
        # # custom button widget
        # button_widget = vtk_ui.Button(self.iren.interactor, label="Font Test", font=font, top=ind * 25, size=sizes[ind])
        
        def _the_callback(widget, event):
            state = widget.GetRepresentation().GetState()
            if callable(callback):
                try_callback(callback, bool(state))

        button_widget.AddObserver(_vtk.vtkCommand.StateChangedEvent, _the_callback)
        p.button_widgets.append(button_widget) 
        return button_widget
    
    

    def clear_button_widgets(self):
        """Disable all of the button widgets."""
        if hasattr(self, 'button_widgets'):
            for widget in self.button_widgets:
                widget.Off()
            del self.button_widgets

    @staticmethod
    def perform_add_button_text_label(p, text, position, font_size=18, color=[1, 1, 1], font='courier', shadow=False, name=None, viewport=False):
        text_actor = _vtk.vtkTextActor()
        text_actor.SetInput(text)
        text_actor.SetPosition(position)
        if viewport:
            text_actor.GetActualPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            text_actor.GetActualPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor.GetTextProperty().SetFontSize(int(font_size * 2))

        text_actor.GetTextProperty().SetColor(color)
        text_actor.GetTextProperty().SetFontFamily(FONTS[font].value)
        text_actor.GetTextProperty().SetShadow(shadow)
        p.add_actor(text_actor, reset_camera=False, name=name, pickable=False)
        return text_actor        

    @staticmethod
    def perform_add_text(p, text, position='upper_left', font_size=18, color=None,
                font=None, shadow=False, name=None, viewport=False):
        """Add text to plot object in the top left corner by default.
        Parameters
        ----------
        text : str
            The text to add the rendering.

        position : str, tuple(float), optional
            Position to place the bottom left corner of the text box.
            If tuple is used, the position of the text uses the pixel
            coordinate system (default). In this case,
            it returns a more general `vtkOpenGLTextActor`.
            If string name is used, it returns a `vtkCornerAnnotation`
            object normally used for fixed labels (like title or xlabel).
            Default is to find the top left corner of the rendering window
            and place text box up there. Available position: ``'lower_left'``,
            ``'lower_right'``, ``'upper_left'``, ``'upper_right'``,
            ``'lower_edge'``, ``'upper_edge'``, ``'right_edge'``, and
            ``'left_edge'``.

        font_size : float, optional
            Sets the size of the title font.  Defaults to 18.

        color : str or sequence, optional
            Either a string, RGB list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

            Defaults to :attr:`pyvista.global_theme.font.color <pyvista.themes._Font.color>`.

        font : str, optional
            Font name may be ``'courier'``, ``'times'``, or ``'arial'``.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to ``False``.

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        viewport : bool, optional
            If ``True`` and position is a tuple of float, uses the
            normalized viewport coordinate system (values between 0.0
            and 1.0 and support for HiDPI).

        Returns
        -------
        vtk.vtkTextActor
            Text actor added to plot.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> actor = PhoWidgetHelper.perform_add_text(pl, 'Sample Text', position='upper_right', color='blue',
        ...                     shadow=True, font_size=26)
        >>> pl.show()
        
        """
        if font is None:
            font = p._theme.font.family
        if font_size is None:
            font_size = p._theme.font.size
        if color is None:
            color = p._theme.font.color
        if position is None:
            # Set the position of the text to the top left corner
            window_size = p.window_size
            x = (window_size[0] * 0.02) / p.shape[0]
            y = (window_size[1] * 0.85) / p.shape[0]
            position = [x, y]

        corner_mappings = {
            'lower_left': _vtk.vtkCornerAnnotation.LowerLeft,
            'lower_right': _vtk.vtkCornerAnnotation.LowerRight,
            'upper_left': _vtk.vtkCornerAnnotation.UpperLeft,
            'upper_right': _vtk.vtkCornerAnnotation.UpperRight,
            'lower_edge': _vtk.vtkCornerAnnotation.LowerEdge,
            'upper_edge': _vtk.vtkCornerAnnotation.UpperEdge,
            'left_edge': _vtk.vtkCornerAnnotation.LeftEdge,
            'right_edge': _vtk.vtkCornerAnnotation.RightEdge,

        }
        corner_mappings['ll'] = corner_mappings['lower_left']
        corner_mappings['lr'] = corner_mappings['lower_right']
        corner_mappings['ul'] = corner_mappings['upper_left']
        corner_mappings['ur'] = corner_mappings['upper_right']
        corner_mappings['top'] = corner_mappings['upper_edge']
        corner_mappings['bottom'] = corner_mappings['lower_edge']
        corner_mappings['right'] = corner_mappings['right_edge']
        corner_mappings['r'] = corner_mappings['right_edge']
        corner_mappings['left'] = corner_mappings['left_edge']
        corner_mappings['l'] = corner_mappings['left_edge']

        if isinstance(position, (int, str, bool)):
            if isinstance(position, str):
                position = corner_mappings[position]
            elif position is True:
                position = corner_mappings['upper_left']
            p.textActor = _vtk.vtkCornerAnnotation()
            # This is how you set the font size with this actor
            p.textActor.SetLinearFontScaleFactor(font_size // 2)
            p.textActor.SetText(position, text)
        else:
            p.textActor = _vtk.vtkTextActor()
            p.textActor.SetInput(text)
            p.textActor.SetPosition(position)
            if viewport:
                p.textActor.GetActualPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                p.textActor.GetActualPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
            p.textActor.GetTextProperty().SetFontSize(int(font_size * 2))

        p.textActor.GetTextProperty().SetColor(parse_color(color))
        p.textActor.GetTextProperty().SetFontFamily(FONTS[font].value)
        p.textActor.GetTextProperty().SetShadow(shadow)

        p.add_actor(p.textActor, reset_camera=False, name=name, pickable=False)
        return p.textActor
    
    
    
    # def getScalars( image_data, x, y ):
    #     comp_data = [ int( image_data.GetScalarComponentAsFloat ( x, y, 0, ic ) ) for ic in range( 4 ) ]
    #     return str( comp_data )


    @staticmethod
    def add_discrete_slider_widget(p, callback, range=(0.0, 1.0), **kwargs):
        # Plots a discrete slider that can be used to select from a series of discrete values, like place cells
        return p.add_slider_widget(callback, range, **({'title':'Trajectory Timestep', 'pass_widget':False, 'event_type':'always', 'style':'modern', 'pointa':(0.025, 0.1), 'pointb':(0.98, 0.1), 'fmt':'%0.2f'} | kwargs))
        
        

# class MyCustomRoutine():
#     def __init__(self, mesh):
#         self.output = mesh # Expected PyVista mesh type
#         # default parameters
#         self.kwargs = {
#             'radius': 0.5,
#             'theta_resolution': 30,
#             'phi_resolution': 30,
#         }

#     def __call__(self, param, value):
#         self.kwargs[param] = value
#         self.update()

#     def update(self):
#         # This is where you call your simulation
#         result = pv.Sphere(**self.kwargs)
#         self.output.overwrite(result)
#         return

# class ListWidget:

#     def __init__( self, interactor, **args ):
#         self.StateChangedSignal = SIGNAL('StateChanged')
#         self.buttonRepresentation = None
#         self.interactor = interactor
#         self.buttons = {}
#         self.visible = False
#         self.windowSize = self.interactor.GetRenderWindow().GetSize()

#     def processStateChangeEvent( self, button, event ):
#         button_rep = button.GetSliderRepresentation()
#         state = button_rep.GetState()
#         button_specs = self.buttons[ button ]
#         button_id = button_specs[ 0 ]
#         self.StateChangedSignal( button, [ button_id, state ] )

#     def getButton( self, **args ):
#         button_id, buttonRepresentation = self.getButtonRepresentation( **args )
#         buttonRepresentation.SetPlaceFactor( args.get( 'scale', 1 ) )
#         position = args.get( 'position', [ 1.0, 1.0 ] )
#         size = args.get( 'size', [ 100.0, 20.0 ] )
#         buttonRepresentation.PlaceWidget( self.computeBounds(position,size) )
#         buttonWidget = _vtk.vtkButtonWidget()
#         buttonWidget.SetInteractor(self.interactor)
#         buttonWidget.SetRepresentation(buttonRepresentation)
#         buttonWidget.AddObserver( 'StateChangedEvent', self.processStateChangeEvent )
#         self.buttons[ buttonWidget ] = [ button_id, position, size ]
#         return buttonWidget

#     def checkWindowSizeChange( self ):
#         new_window_size = self.interactor.GetRenderWindow().GetSize()
#         if ( self.windowSize[0] != new_window_size[0] ) or ( self.windowSize[1] != new_window_size[1] ):
#             self.windowSize = new_window_size
#             return True
#         else:
#             return False

#     def updatePositions(self):
#         if self.checkWindowSizeChange():
#             for button_item in self.buttons.items():
#                 button = button_item[0]
#                 [ button_id, position, size ] = button_item[1]
#                 brep = button.GetRepresentation()
#                 brep.PlaceWidget( self.computeBounds(position,size) )
#                 brep.Modified()
#                 button.Modified()

#     def build(self):
#         pass

#     def getButtonRepresentation(self):
#         return None, None

#     def show(self):
#         self.visible = True
#         for button in self.buttons.keys():
#             button.On()
# #            button.Render()

#     def hide(self):
#         self.visible = False
#         for button in self.buttons.keys():
#             button.Off()

#     def toggleVisibility( self, **args ):
#         state = args.get( 'state', None )
#         if state != None: self.visible = True if ( state == 0 ) else False
#         if self.visible:
#             self.hide()
#         else:
#             self.updatePositions()
#             self.show()

#     def getRenderer(self):
#         rw = self.interactor.GetRenderWindow()
#         return rw.GetRenderers().GetFirstRenderer ()

#     def computeBounds( self, normalized_display_position, size ):
#         renderer = self.getRenderer()
#         upperRight = _vtk.vtkCoordinate()
#         upperRight.SetCoordinateSystemToNormalizedDisplay()
#         upperRight.SetValue( normalized_display_position[0], normalized_display_position[1] )
#         bds = [0.0]*6
#         bds[0] = upperRight.GetComputedDisplayValue(renderer)[0] - size[0]
#         bds[1] = bds[0] + size[0]
#         bds[2] = upperRight.GetComputedDisplayValue(renderer)[1] - size[1]
#         bds[3] = bds[2] + size[1]
#         return bds

