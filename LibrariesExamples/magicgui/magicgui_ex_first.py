import enum
import pathlib

from datetime import datetime
from magicgui import magicgui

class Choice(enum.Enum):
    A = 'Choice A'
    B = 'Choice B'
    C = 'Choice C'

@magicgui
def widget_demo(
    boolean=True,
    number=1,
    string="Text goes here",
    dropdown=Choice.A,
    filename=pathlib.Path.home(),
):
    """Run some computation."""
    ...

widget_demo.show()