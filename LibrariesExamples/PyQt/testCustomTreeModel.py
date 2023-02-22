import sys
from PyQt5.QtWidgets import QApplication, QTreeView
from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex

class TreeItem:
    def __init__(self, data, parent=None):
        self.parentItem = parent
        self.itemData = data
        self.childItems = []

    def appendChild(self, item):
        self.childItems.append(item)

    def child(self, row):
        return self.childItems[row]

    def childCount(self):
        return len(self.childItems)

    def columnCount(self):
        return len(self.itemData)

    def data(self, column):
        return self.itemData[column]

    def parent(self):
        return self.parentItem

    def row(self):
        if self.parentItem:
            return self.parentItem.childItems.index(self)
        return 0


class TreeModel(QAbstractItemModel):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.rootItem = TreeItem(['Name', 'Visibility', 'Color'])
        self.setupModelData(data, self.rootItem)

    def columnCount(self, parent):
        if parent.isValid():
            return parent.internalPointer().columnCount()
        return self.rootItem.columnCount()

    def data(self, index, role):
        if not index.isValid():
            return None

        if role != Qt.DisplayRole and role != Qt.EditRole:
            return None

        item = index.internalPointer()

        return item.data(index.column())

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags

        return super().flags(index) | Qt.ItemIsEditable

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.rootItem.data(section)

        return None

    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent()

        if parentItem == self.rootItem:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        return parentItem.childCount()

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole:
            return False

        item = index.internalPointer()
        item.itemData[index.column()] = value
        self.dataChanged.emit(index, index)
        return True

    def setupModelData(self, data, parent):
        for name, visibility, color in data:
            itemData = [name, visibility, color]
            parent.appendChild(TreeItem(itemData, parent))




# from tree_model import TreeModel

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Test data
    data = [('Item 1', 'visible', 'red'), ('Item 2', 'hidden', 'blue')]

    # Create the model
    model = TreeModel(data)

    # Create the tree view and set the model
    tree_view = QTreeView()
    tree_view.setModel(model)

    # # Check the data in the model
    # assert model.data(model.index(0, 0, None)) == 'Item 1'
    # assert model.data(model.index(0, 1, None)) == 'visible'
    # assert model.data(model.index(0, 2, None)) == 'red'

    # assert model.data(model.index(1, 0, None)) == 'Item 2'
    # assert model.data(model.index(1, 1, None)) == 'hidden'
    # assert model.data(model.index(1, 2, None)) == 'blue'

    # # Check that the data can be modified
    # model.setData(model.index(0, 1, None), 'hidden')
    # assert model.data(model.index(0, 1, None)) == 'hidden'

    # model.setData(model.index(1, 2, None), 'green')
    # assert model.data(model.index(1, 2, None)) == 'green'

    # # Check the structure of the model
    # assert model.index(0, 0, None).isValid()
    # assert model.index(0, 0, model.index(0, 0, None)).isValid()
    # assert model.index(1, 0, model.index(0, 0, None)).isValid()

    # Show the tree view and run the event loop
    tree_view.show()
    sys.exit(app.exec_())

