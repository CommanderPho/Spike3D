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
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ParameterTreeWidget import create_parameter_tree_widget


# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
importlib.reload(core)
from neuropy.core.neurons import NeuronType

# Custom Param Types:
from _buildFilterParamTypes import makeAllParamTypes
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui
import pyphoplacecellanalysis.External.pyqtgraph.parametertree.parameterTypes as pTypes
from pyphoplacecellanalysis.External.pyqtgraph.parametertree import Parameter, ParameterTree
from pyphoplacecellanalysis.GUI.PyQtPlot.Params.SaveRestoreStateParamHelpers import default_parameters_save_restore_state_button_children, add_save_restore_btn_functionality # for adding save/restore buttons

## For qdarkstyle theming support:
import qdarkstyle
# app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

# For BreezeStylesheets support:
from qtpy import QtWidgets
from qtpy.QtCore import QFile, QTextStream
import pyphoplacecellanalysis.External.breeze_style_sheets.breeze_resources
# # set stylesheet:
# stylesheet_qss_file = QFile(":/dark/stylesheet.qss")
# stylesheet_qss_file.open(QFile.ReadOnly | QFile.Text)
# stylesheet_data_stream = QTextStream(stylesheet_qss_file)
# # app.setStyleSheet(stylesheet_data_stream.readAll())



app = pg.mkQApp("Parameter Tree Filter Options")
# app.setStyleSheet(stylesheet_data_stream.readAll())
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5()) # QDarkStyle version




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


## Define Custom Parameter types:
import pyphoplacecellanalysis.External.pyqtgraph.parametertree.parameterTypes as pTypes
from pyphoplacecellanalysis.External.pyqtgraph.parametertree import Parameter, ParameterTree
## this group includes a menu allowing the user to add new parameters into its child list
class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        opts['addList'] = ['str', 'float', 'int'] # dropdown list of items to add (shows up in the combo box)
        pTypes.GroupParameter.__init__(self, **opts)
    
    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(dict(name="ScalableParam %d" % (len(self.childs)+1), type=typ, value=val, removable=True, renamable=True))
        
class ExportHdf5KeysGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add Key"
        opts['addList'] = ['str', 'float', 'int'] # dropdown list of items to add (shows up in the combo box)
        pTypes.GroupParameter.__init__(self, **opts)
    
    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(dict(name="/filtered_sessions/YOUR_SESSION_NAME/YOUR_KEY", type=typ, value=val, removable=True, renamable=True))
        

##################################################
## FILTER TREE
##################################################/
def _build_filter_parameters_tree(parameter_names='Filter Options', include_state_save_restore_buttons=True, debug_print=False):    
    def _simple_filter_dict_params():
        children = [
            # dict(name='a', type='slider', value=5, limits=[0, 10]),
            # dict(name='b', type='slider', value=0.1, limits=[-5, 5], step=0.1),
            # dict(name='c', type='slider', value=2, span=np.linspace(0, 2*np.pi, 1000)),
            dict(name='Data Directory', type='file', winTitle='Select Input Data Directory', directory='', options=['ShowDirsOnly', 'DontResolveSymlinks']),
            dict(name='Included Epochs', type='checklist', value=['maze1'], limits=['pre', 'maze1', 'post1', 'maze2', 'post2']),
            dict(name='Cell Types', type='checklist', value=[NeuronType.PYRAMIDAL.longClassName], limits=NeuronType.__members__),
        ]

        # Use save/restore state buttons
        if include_state_save_restore_buttons:
            children.append(default_parameters_save_restore_state_button_children())

        p = Parameter.create(name=parameter_names, type='group', children=children)
        return p
        
    p = _simple_filter_dict_params()

    ## If anything changes in the tree, print a message
    def on_tree_value_change(param, changes):
        print("tree changes:")
        for param, change, data in changes:
            path = p.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')

    p.sigTreeStateChanged.connect(on_tree_value_change)

    def valueChanging(param, value):
        # called whenever a child value is changed:
        print("Value changing (not finalized): %s %s" % (param, value))

    # Only listen for changes of the 'widget' child:
    # for child in p.child(parameter_names):
    for child in p.children():
        if 'widget' in child.names:
            child.child('widget').sigValueChanging.connect(valueChanging)
        
    if include_state_save_restore_buttons:
        add_save_restore_btn_functionality(p)

    return p


##################################################/
## EXPORT TREE:
##################################################/
def _build_export_parameters_tree(parameter_names='ExportParams', finalized_output_cache_file='data/pipeline_cache_store.h5', include_state_save_restore_buttons=True, debug_print=False):
    # Define Saving/Loading Directory and paths:
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import _test_save_pipeline_data_to_h5, get_h5_data_keys, save_some_pipeline_data_to_h5, load_pipeline_data_from_h5  #ExportHelpers

    
    def _simple_export_dict_params():
        # List existing keys in the file:
        # out_keys = get_h5_data_keys(finalized_output_cache_file=finalized_output_cache_file)
        
        ## DEBUG ONLY: for debug testing, don't actually require a .h5 file to load the keys from, use these hardcoded defaults
        out_keys = ['/spikes_df', '/sess/pos_df', '/sess/spikes_df', '/filtered_sessions/maze2/pos_df', '/filtered_sessions/maze2/spikes_df', '/filtered_sessions/maze2/pos_df/meta/values_block_2/meta', '/filtered_sessions/maze2/pos_df/meta/values_block_1/meta', '/filtered_sessions/maze1/pos_df', '/filtered_sessions/maze1/spikes_df', '/filtered_sessions/maze1/pos_df/meta/values_block_2/meta', '/filtered_sessions/maze1/pos_df/meta/values_block_1/meta', '/filtered_sessions/maze/pos_df', '/filtered_sessions/maze/spikes_df', '/filtered_sessions/maze/pos_df/meta/values_block_2/meta', '/filtered_sessions/maze/pos_df/meta/values_block_1/meta']
        
        if debug_print:
            print(f'out_keys: {out_keys}')
        key_children_list = [{'name': extant_key, 'type': 'str', 'value': "<TODO>"} for extant_key in out_keys]
        if debug_print:
            print(f'key_children_list: {key_children_list}')
        children = [
                dict(name='Export Path', type='file', dialogLabel='test label', value=finalized_output_cache_file),
                dict(name='Cell Types', type='checklist', value=[], limits=['one', 'two', 'three', 'four']),
                ExportHdf5KeysGroup(name="Export Keys", tip='Click to add children', children=key_children_list),
            ]

        # Use save/restore state buttons
        if include_state_save_restore_buttons:
            children.append(default_parameters_save_restore_state_button_children())

        ## Create tree of Parameter objects
        p = Parameter.create(name=parameter_names, type='group', children=children)
        return p
    
    p = _simple_export_dict_params()
    
    ## If anything changes in the tree, print a message
    def on_tree_value_change(param, changes):
        print("tree changes:")
        for param, change, data in changes:
            path = p.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')

    p.sigTreeStateChanged.connect(on_tree_value_change)

    def valueChanging(param, value):
        # called whenever a child value is changed:
        print("Value changing (not finalized): %s %s" % (param, value))

    # Only listen for changes of the 'widget' child:
    # for child in p.child(parameter_names):
    for child in p.children():
        if 'widget' in child.names:
            child.child('widget').sigValueChanging.connect(valueChanging)
        
    if include_state_save_restore_buttons:
        add_save_restore_btn_functionality(p)

    return p



## Main Create function
def create_pipeline_parameter_tree(tree_type='filter', debug_print=False):
    ## Create two ParameterTree widgets, both accessing the same data
    if tree_type == 'filter':
        ## FILTER TREE
        p = _build_filter_parameters_tree(parameter_names='Filter Options', debug_print=debug_print)
    elif tree_type == 'export':
        ## EXPORT TREE:
        p = _build_export_parameters_tree(parameter_names='ExportParams', finalized_output_cache_file='data/pipeline_cache_store.h5', debug_print=debug_print)
    else:
        p = None
        raise NotImplementedError
    
    layout_win, param_tree = create_parameter_tree_widget(p)
    return layout_win, param_tree


## Conveninece functions:
def create_pipeline_filter_parameter_tree():
    return create_pipeline_parameter_tree(tree_type='filter')

def create_pipeline_export_parameter_tree():
    return create_pipeline_parameter_tree(tree_type='export')


if __name__ == '__main__':
    # win, param_tree = create_pipeline_filter_parameter_tree()
    win, param_tree = create_pipeline_export_parameter_tree()
    pg.exec()
