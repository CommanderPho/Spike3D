from pathlib import Path
import dill
from attrs import define, field, Factory
from PyQt5.QtCore import Qt, QAbstractItemModel
from PyQt5.QtWidgets import QTreeView, QApplication
import sys
import inspect

@define
class ObjectTreeModel(QAbstractItemModel):
    root_object: object

    def rowCount(self, parent):
        obj = self.getNode(parent)
        return len(inspect.getmembers(obj))

    def getNode(self, index):
        return index.internalPointer() if index.isValid() else self.root_object

    # Implement other necessary methods

@define
class ObjectBrowser(QTreeView):
    object_model: ObjectTreeModel

    def __init__(self, root_object):
        super().__init__()
        self.object_model = ObjectTreeModel(root_object=root_object)
        self.setModel(self.object_model)
        self.setRootIndex(self.object_model.index(root_object))




## Testing
if __name__ == "__main__":
    test_file_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-09_1-22-43\output\global_computation_results.pkl')
    assert test_file_path.exists()
    any_python_object = dill.load(test_file_path)

    app = QApplication(sys.argv)
    root_object = any_python_object # Replace with the object you want to browse
    browser = ObjectBrowser(root_object=root_object)
    browser.show()
    sys.exit(app.exec_())


