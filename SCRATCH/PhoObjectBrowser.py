from pathlib import Path
import dill
from attrs import define, field, Factory
from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QVariant
from PyQt5.QtWidgets import QTreeView, QApplication
import sys
import inspect
from pyphocorehelpers.assertion_helpers import Assert
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData
from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows

@define
class ObjectTreeModel(QAbstractItemModel):
    root_object: object = field(init=False)

    def __init__(self, root_object: object):
        super().__init__()
        self.root_object = root_object
        
    def rowCount(self, parent: QModelIndex) -> int:
        obj = self.getNode(parent)
        return len(inspect.getmembers(obj))

    def columnCount(self, parent: QModelIndex) -> int:
        return 2  # Name and value

    def data(self, index: QModelIndex, role: int) -> QVariant:
        if not index.isValid() or role != Qt.DisplayRole:
            return QVariant()

        obj = self.getNode(index.parent())
        members = inspect.getmembers(obj)
        row = index.row()
        column = index.column()

        if 0 <= row < len(members):
            name, value = members[row]
            if column == 0:
                return name
            elif column == 1:
                return str(value)

        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> QVariant:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                return "Name"
            elif section == 1:
                return "Value"
        return QVariant()

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        obj = self.getNode(parent)
        members = inspect.getmembers(obj)

        if 0 <= row < len(members):
            child_obj = members[row][1]
            return self.createIndex(row, column, child_obj)

        return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        return QModelIndex()  # No hierarchy; flat structure for simplicity

    def getNode(self, index: QModelIndex):
        return index.internalPointer() if index.isValid() else self.root_object
    

@define
class ObjectBrowser(QTreeView):
    object_model: ObjectTreeModel = field(init=False)

    def __init__(self, root_object):
        super().__init__()
        self.object_model = ObjectTreeModel(root_object=root_object)
        self.setModel(self.object_model)
        self.setRootIndex(self.object_model.index(0, 0, QModelIndex()))




## Testing
if __name__ == "__main__":
    debug_print: bool = True
    pkl_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-09_1-22-43\output\global_computation_results.pkl')
    
    # pkl_path = Path('W:/Data/KDIBA/gor01/one/2006-6-09_1-22-43/loadedSessPickle.pkl')
    Assert.path_exists(pkl_path)
    
    print(f'pkl_path: {pkl_path}')
    
    def _try_load_global_batch_result():
        if debug_print:
            print(f'pkl_path: {pkl_path}')
        # try to load an existing batch result:
        try:
            curr_active_pipeline = loadData(pkl_path, debug_print=debug_print)
            
        except NotImplementedError:
            # Fixes issue with pickled POSIX_PATH on windows for path.                    
            with set_posix_windows():
                curr_active_pipeline = loadData(pkl_path, debug_print=debug_print) # Fails this time if it still throws an error

        except (FileNotFoundError, TypeError):
            # loading failed
            print(f'Failure loading {pkl_path}.')
            curr_active_pipeline = None
            
        return curr_active_pipeline
    
    any_python_object = _try_load_global_batch_result()
    assert any_python_object is not None
    print(f'loaded `curr_active_pipeline`, building Spike3DRasterWindowWidget...')


    # any_python_object = dill.load(pkl_path)

    app = QApplication(sys.argv)
    root_object = any_python_object # Replace with the object you want to browse
    browser = ObjectBrowser(root_object=root_object)
    browser.show()
    sys.exit(app.exec_())


