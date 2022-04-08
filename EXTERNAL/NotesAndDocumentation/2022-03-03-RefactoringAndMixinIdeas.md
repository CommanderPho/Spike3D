
# Use a model like QtDesigner generates to add required properties to an implementor from a Mixin.

```python
""" qtDesigner auto-generated python file/class """
class Ui_rootForm(object):
    def setupUi(self, rootForm):
        rootForm.setObjectName("rootForm")
        rootForm.resize(100, 116)
        """ ... """
```

```python

from .PlacefieldVisualSelectionWidgetBase import Ui_rootForm # Generated file from .ui

class PlacefieldVisualSelectionWidget(QtWidgets.QWidget):
    """Custom python class that properly implements the qtDesigner generated ui. 
        sets self.ui and then calls self.ui.setupUi(self) to add the UI widgets and such.
      """
 
    def __init__(self, *args, **kwargs):
        super(PlacefieldVisualSelectionWidget, self).__init__(*args, **kwargs)
        self.ui = Ui_rootForm()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        """ ... """
```

        # Helper variables
        self.params = VisualizationParameters('')
        self.debug = DebugHelper('')
        