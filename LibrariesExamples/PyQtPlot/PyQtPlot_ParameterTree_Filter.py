"""
Filter Parameters

"""
import importlib
import sys
from pathlib import Path
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np

from pyphoplacecellanalysis.External.pyqtgraph.widgets.FeedbackButton import FeedbackButton

# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core

    importlib.reload(core)
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print("neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.")
    from neuropy import core

from neuropy.core.neuron_identities import NeuronType
# Custom Param Types:
from _buildFilterParamTypes import makeAllParamTypes

from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui

app = pg.mkQApp("Parameter Tree Filter Options")
import pyphoplacecellanalysis.External.pyqtgraph.parametertree.parameterTypes as pTypes
from pyphoplacecellanalysis.External.pyqtgraph.parametertree import Parameter, ParameterTree
from pyphoplacecellanalysis.GUI.PyQtPlot.Params.SaveRestoreStateParamHelpers import add_save_restore_btn_functionality


## test subclassing parameters
## This parameter automatically generates two child parameters which are always reciprocals of each other
class ComplexParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.addChild({'name': 'A = 1/B', 'type': 'float', 'value': 7, 'suffix': 'Hz', 'siPrefix': True})
        self.addChild({'name': 'B = 1/A', 'type': 'float', 'value': 1 / 7., 'suffix': 's', 'siPrefix': True})
        self.a = self.param('A = 1/B')
        self.b = self.param('B = 1/A')
        self.a.sigValueChanged.connect(self.aChanged)
        self.b.sigValueChanged.connect(self.bChanged)

    def aChanged(self):
        self.b.setValue(1.0 / self.a.value(), blockSignal=self.bChanged)

    def bChanged(self):
        self.a.setValue(1.0 / self.b.value(), blockSignal=self.aChanged)


## test add/remove
## this group includes a menu allowing the user to add new parameters into its child list
class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        opts['addList'] = ['str', 'float', 'int']
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(
            dict(name="ScalableParam %d" % (len(self.childs) + 1), type=typ, value=val, removable=True, renamable=True))

def _save_restore_state_button_children():
    return {
        'name': 'Save/Restore functionality', 'type': 'group', 'children': [
        {'name': 'Save State', 'type': 'action'},
        {
            'name': 'Restore State', 'type': 'action', 'children': [
            {'name': 'Add missing items', 'type': 'bool', 'value': True},
            {'name': 'Remove extra items', 'type': 'bool', 'value': True},
        ]},
    ]}


# # Extra Parameters:
# params = [
#     makeAllParamTypes(),
#     {
#         'name': 'Save/Restore functionality', 'type': 'group', 'children': [
#         {'name': 'Save State', 'type': 'action'},
#         {
#             'name': 'Restore State', 'type': 'action', 'children': [
#             {'name': 'Add missing items', 'type': 'bool', 'value': True},
#             {'name': 'Remove extra items', 'type': 'bool', 'value': True},
#         ]},
#     ]},
#     {
#         'name': 'Custom context menu', 'type': 'group', 'children': [
#         {
#             'name': 'List contextMenu', 'type': 'float', 'value': 0, 'context': [
#             'menu1',
#             'menu2'
#         ]},
#         {
#             'name': 'Dict contextMenu', 'type': 'float', 'value': 0, 'context': {
#             'changeName': 'Title',
#             'internal': 'What the user sees',
#         }},
#     ]},
#     ComplexParameter(name='Custom parameter group (reciprocal values)'),
#     ScalableGroup(name="Expandable Parameter Group", tip='Click to add children', children=[
#         {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
#         {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
#     ]),
# ]

# ## Create tree of Parameter objects
# p = Parameter.create(name='params', type='group', children=params)


# Simple Example:
def _example_simple_dict_params():
    children = [
        # dict(name='a', type='slider', value=5, limits=[0, 10]),
        # dict(name='b', type='slider', value=0.1, limits=[-5, 5], step=0.1),
        # dict(name='c', type='slider', value=2, span=np.linspace(0, 2*np.pi, 1000)),
        dict(name='Included Epochs', type='checklist', value=['maze1'], limits=['pre', 'maze1', 'post1', 'maze2', 'post2']),
        dict(name='Cell Types', type='checklist', value=[NeuronType.PYRAMIDAL.longClassName], limits=NeuronType.__members__),
    ]

    # Use save/restore state buttons
    children.append(_save_restore_state_button_children())
    
    p = Parameter.create(name='Filter Options', type='group', children=children)
    return p
    # t = ParameterTree()
    # t.setParameters(p, showTop=True)


p = _example_simple_dict_params()


## If anything changes in the tree, print a message
def change(param, changes):
    print("tree changes:")
    for param, change, data in changes:
        path = p.childPath(param)
        if path is not None:
            childName = '.'.join(path)
        else:
            childName = param.name()
        print('  parameter: %s' % childName)
        print('  change:    %s' % change)
        print('  data:      %s' % str(data))
        print('  ----------')


p.sigTreeStateChanged.connect(change)


def valueChanging(param, value):
    print("Value changing (not finalized): %s %s" % (param, value))


# Too lazy for recursion:
for child in p.children():
    child.sigValueChanging.connect(valueChanging)
    for ch2 in child.children():
        ch2.sigValueChanging.connect(valueChanging)


add_save_restore_btn_functionality(p)


## Create two ParameterTree widgets, both accessing the same data
t = ParameterTree()
t.setParameters(p, showTop=False)
t.setWindowTitle('pyqtgraph example: Parameter Tree')
t2 = ParameterTree()
t2.setParameters(p, showTop=False)

win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)
layout.addWidget(QtGui.QLabel("These are two views of the same data. They should always display the same values."), 0, 0, 1, 2)
layout.addWidget(t, 1, 0, 1, 1)
layout.addWidget(t2, 1, 1, 1, 1)
win.show()
win.resize(800,900)

## test save/restore
state = p.saveState()
p.restoreState(state)
compareState = p.saveState()
assert pg.eq(compareState, state)

if __name__ == '__main__':
    pg.exec()
