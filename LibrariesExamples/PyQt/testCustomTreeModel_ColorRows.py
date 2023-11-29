import sys
from PyQt5.QtWidgets import QApplication, QTreeView
from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex

from PyQt5.QtCore import Qt, QVariant
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QStyledItemDelegate, QColorDialog

from PyQt5.QtGui import QBrush, QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QVBoxLayout, QWidget, QStyleFactory

from pyphoplacecellanalysis.External import pyqtgraph

class CustomDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        if index.column() == 2:
            editor = QColorDialog(parent)
            editor.setOption(QColorDialog.ShowAlphaChannel, True)
            return editor
        else:
            return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if index.column() == 2:
            value = index.data(Qt.DisplayRole)
            if value is not None:
                value = QColor(value)
            editor.setCurrentColor(value)
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if index.column() == 2:
            color = editor.currentColor().rgba()
            model.setData(index, QVariant(color), Qt.DisplayRole)
            model.setData(index, QVariant(QColor(color)), Qt.DecorationRole)
        else:
            super().setModelData(editor, model, index)

    def updateEditorGeometry(self, editor, option, index):
        if index.column() == 2:
            editor.setGeometry(option.rect)
        else:
            super().updateEditorGeometry(editor, option, index)

    def paint(self, painter, option, index):
        if index.column() == 2:
            value = index.data(Qt.DecorationRole)
            if value is not None:
                painter.fillRect(option.rect, value)
            else:
                super().paint(painter, option, index)
        else:
            super().paint(painter, option, index)


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
        self.setup_model_data(data, self.rootItem)

    def columnCount(self, parent):
        if parent.isValid():
            return parent.internalPointer().columnCount()
        return self.rootItem.columnCount()

    # def data(self, index, role):
    #     if not index.isValid():
    #         return None

    #     if role != Qt.DisplayRole and role != Qt.EditRole:
    #         return None

    #     item = index.internalPointer()

    #     return item.data(index.column())

    def data(self, index, role):
        if not index.isValid():
            return None

        item = index.internalPointer()

        if role == Qt.DisplayRole or role == Qt.EditRole:
            if index.column() == 0:
                return item.data('name')
            elif index.column() == 1:
                return item.data('visibility')
            elif index.column() == 2:
                return item.data('color')
        elif role == Qt.CheckStateRole and index.column() == 1:
            return Qt.Checked if item.data('visibility') else Qt.Unchecked
        elif role == Qt.DecorationRole and index.column() == 2:
            color = item.data('color')
            if color is not None:
                pixmap = QPixmap(16, 16)
                pixmap.fill(color)
                icon = QIcon(pixmap)
                return icon

        return None



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

    # def setupModelData(self, data, parent):
    #     for name, visibility, color in data:
    #         itemData = [name, visibility, color]
    #         parent.appendChild(TreeItem(itemData, parent))

    def setup_model_data(self, data, parent=None):
        parent_item = self.root_item if parent is None else parent

        # for name, visibility, color in data:
        for name, row_dict in data.items():
            visibility = row_dict['visibility']
            color = row_dict['color']
            item_data = {'name': name, 'visibility': visibility, 'color': color}
            item = TreeItem(item_data, parent_item)
            parent_item.appendChild(item)






# from treeModel import TreeItem, TreeModel, CustomDelegate


if __name__ == '__main__':
    data = {
        'Item 1': {'visibility': True, 'color': QColor('red')},
        'Item 2': {'visibility': False, 'color': QColor('blue')},
        'Item 3': {'visibility': True, 'color': QColor('green')},
        'Item 4': {'visibility': False, 'color': QColor('purple')},
        'Item 5': {'visibility': True, 'color': QColor('yellow')}
    }

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    model = TreeModel(data)

    view = QTreeView()
    view.setModel(model)
    view.setItemDelegate(CustomDelegate())
    view.expandAll()

    main_window = QMainWindow()
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    layout.addWidget(view)
    main_window.setCentralWidget(central_widget)
    main_window.show()

    sys.exit(app.exec_())

