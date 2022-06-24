# Render Items:
	## TODO: Unfinished and unused


```python
class RenderClassName:
    """ custom render item class

    Allows encapsulation of everything related to an item to be visually rendered as an alternative to mixins.
        Enables composition instead of inheritance, which is often easier to debug and permits reloading from module automatically in Jupyter-Lab with importlib
        
        
    Responsibilities:
        holds the render parameters
        holds the datasource and command used to render its items.
        holds references to the items it renders so that they can be removed later.
    
    """
    
    @QtCore.pyqtSlot()
    def RenderClassName_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @QtCore.pyqtSlot()
    def RenderClassName_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass


    @QtCore.pyqtSlot()
    def RenderClassName_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass

    @QtCore.pyqtSlot()
    def RenderClassName_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        pass

    @QtCore.pyqtSlot(float, float)
    def RenderClassName_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        pass

    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.pyqtSlot(object)
    def RenderClassName_on_window_update_rate_limited(self, evt):
        self.RenderClassName_on_window_update(*evt)

```