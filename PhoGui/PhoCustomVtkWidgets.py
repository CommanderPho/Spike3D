"""Module dedicated to custom VTK widgets."""

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import (NORMALS, generate_plane, get_array,
                               try_callback, get_array_association)
from pyvista.plotting.tools import parse_color, FONTS

from PhoGui import vtk_ui
# import vtk_ui

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
                            background_color='white'):
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
        
        def create_button(color1, color2, color3, dims=[size, size, 1]):
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
        
        button_on = create_button(color_on, background_color, color_on)
        button_off = create_button(color_on, background_color, color_off)

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


    @staticmethod(function)
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