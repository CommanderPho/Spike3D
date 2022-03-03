
import pickle
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import (
    WidgetParameterItem, registerParameterType)
from pyqtgraph.Qt import QtGui, QtCore


def tree(name, params):
    """Create a parameter tree object
    """
    pg.mkQApp()
    if not isinstance(params, list):
        params = [params]
    p = Parameter.create(name=name, type='group', children=params)
    t = ParameterTree()
    t.setParameters(p, showTop=False)
    t.paramSet = p
    return t


def group(name, children, **kwargs):
    """Create a named group of parameters
    """
    return dict(name=name, type='group', children=children, **kwargs)


def item(name, value='', values=None, **kwargs):
    """Add an item to a parameter tree.

    Parameters
    ----------
    name : str
        Name of parameter
    value : object
        Default value for object.  If 'type' is not given,
        the type of this object is used to infer the 'type'
    values : list
        Allowable values.  If 'type' is not given, and 'values' are given,
        `type` will be assumed to be `list`.
    **kwargs
        Additional keyword arguments, such as

        type : str, The type of parameter (e.g. 'action' or 'text')
        step : int, The spinbox stepsize
        suffix : str, modifier suffix (e.g. 'Hz' or 'V')
        siPrefix : bool, whether to add an SI prefix ('e.g. mV or MHz')
    """
    if 'type' not in kwargs:
        if values:
            kwargs['type'] = 'list'
        elif isinstance(value, bool):
            kwargs['type'] = 'bool'
        elif isinstance(value, int):
            kwargs['type'] = 'int'
        elif isinstance(value, float):
            kwargs['type'] = 'float'
        else:
            kwargs['type'] = 'str'
    return dict(name=name, value=value, values=values, **kwargs)


def show(widget, title='', width=800, height=800):
    """Show a simple application around a widget
    """
    widget.setWindowTitle(title)
    widget.show()
    widget.resize(width, height)
    QtGui.QApplication.instance().exec_()


def register_callback(tree, cb):
    """Register a change callback to a tree
    """
    def change(param, changes):
        new_changes = []
        for param, change, data in changes:
            name = [param.name()]
            parent = param.parent()
            while parent:
                name = [parent.name()] + name
                parent = parent.parent()
            name = '.'.join(name[1:])
            new_changes.append(dict(name=name, param=param,
                                    type=change, data=data))
        cb(new_changes)

    tree.paramSet.sigTreeStateChanged.connect(change)


def save_state(tree, filename):
    """Save a tree as a pickle file
    """
    with open(filename, 'w') as fid:
        pickle.dump(fid, tree.parmSet.saveState())


def load_state(tree, filename):
    """Load a tree state to a pickle file
    """
    with open(filename) as fid:
        data = pickle.load(fid)

    tree.paramSet.restoreState(data)


def get_item_by_name(tree, name):
    """Get an item in a tree by name
    """
    items = tree.paramSet.children()
    item = None
    while True:
        first, _, name = name.partition('.')
        names = [i.name() for i in items]
        if first in names:
            item = items[names.index(first)]
            items = item.children()
        if not name:
            return item


def tabs(*args):
    """Create a set of Tabs based on a set of (name, widget) tuples"""
    nb = QtGui.QTabWidget()
    for (name, widget) in args:
        nb.addTab(widget, name)
    nb.setMovable(True)
    return nb


def _splitter(*children, ltype=QtGui.QHBoxLayout):
    widget = QtGui.QWidget()
    layout = ltype()
    for child in children:
        layout.addWidget(child)
    widget.setLayout(layout)
    return widget


def hbox(*children):
    """Create an Horizontal Box Layout"""
    return _splitter(*children)


def vbox(*children):
    """Create a Vertical Box Layout"""
    return _splitter(*children, ltype=QtGui.QVBoxLayout)


class SliderParameterItem(WidgetParameterItem):

    def makeWidget(self):
        w = QtGui.QSlider(QtCore.Qt.Horizontal, self.parent())
        w.sigChanged = w.valueChanged
        w.sigChanged.connect(self._set_tooltip)
        self.widget = w
        return w

    def _set_tooltip(self):
        self.widget.setToolTip(str(self.widget.value()))


class SliderParameter(Parameter):
    itemClass = SliderParameterItem

registerParameterType('slider', SliderParameter, override=True)


if __name__ == '__main__':
    grp = group('test', [
                item('item1', 1.5, suffix='V', siPrefix=True),
                item('item2', 'hello'),
                item('item3', False),
                item('item4', 10, type='slider'),
                item('item5', 2, [1, 2, 3])])
    tree1 = tree('my tree', grp)

    def change(changes):
        print("tree changes:")
        for change in changes:
            print('  parameter: %s' % change['name'])
            print('  change:    %s' % change['type'])
            print('  data:      %s' % change['data'])
            print('  ----------')

            if 'item1' not in change['name']:
                param = get_item_by_name(tree1, 'test.item1')
                param.setValue(param.value() + 1)

    register_callback(tree1, change)

    tree2 = tree('my tree 2', grp)
    tree3 = tree('my tree 3', grp)
    tree4 = tree('my tree 4', [item('hello', 1.5)])
    show(hbox(tree1,
              vbox(tree2,
                   tabs(('3', tree3), ('4', tree4)))))
