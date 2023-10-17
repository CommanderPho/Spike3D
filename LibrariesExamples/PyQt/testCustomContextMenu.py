import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMenu, QListWidget, QVBoxLayout, QAction
from PyQt5.QtCore import QEvent, Qt
# from PyQt5.QtGui import QActionEvent

from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable import LocalMenus_AddRenderable
from pyphoplacecellanalysis.GUI.Qt.Menus.BaseMenuProviderMixin import BaseMenuProviderMixin


class MyApp(QWidget):
    def __init__(self, custom_menu=None):
        self.custom_menu = custom_menu
        super().__init__()
        self.setWindowTitle('Insert Context Menu to ListWidget')
        self.window_width, self.window_height = 800, 600
        self.setMinimumSize(self.window_width, self.window_height)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.listWidget = QListWidget()
        self.listWidget.addItems(('Facebook', 'Microsoft', 'Google'))
        self.listWidget.installEventFilter(self)
        layout.addWidget(self.listWidget)

    def eventFilter(self, source, event):
        if event.type() == QEvent.ContextMenu and source is self.listWidget:
            if self.custom_menu is not None:
                menu = self.custom_menu
            else:
                # make default new menu:
                menu = QMenu()
                menu.addAction('Action 1')
                menu.addAction('Action 2')
                menu.addAction('Action 3')

            if menu.exec_(event.globalPos()):
                item = source.itemAt(event.pos())
                print(item.text())
            return True
        return super().eventFilter(source, event)


def create_custom_context_menu(owner):
    # # create context menu
    # self.popMenu = QtGui.QMenu(self)
    # self.popMenu.addAction(QtGui.QAction('test0', self))
    # self.popMenu.addAction(QtGui.QAction('test1', self))
    # self.popMenu.addSeparator()
    # self.popMenu.addAction(QtGui.QAction('test2', self))

    popMenuActionsList = [QAction('test0', parent=owner),
        QAction('test1', parent=owner),
        QAction('test2', parent=owner)]
    
    
    ## create context menu
    owner.popMenu = QMenu(parent=owner)
    # ## Inline Actions:
    # owner.popMenu.addAction(QAction('test0', parent=owner))
    # owner.popMenu.addAction(QAction('test1', parent=owner))
    # owner.popMenu.addSeparator()
    # owner.popMenu.addAction(QAction('test2', parent=owner))
        
    # owner.popMenu.addAction(QAction('test0', parent=owner))
    # owner.popMenu.addAction(QAction('test1', parent=owner))
    # owner.popMenu.addSeparator()
    # owner.popMenu.addAction(QAction('test2', parent=owner))
    
    # nested_action_parent = popMenuActionsList[1]
    nested_action_parent = owner
    # nested_action_parent = owner.popMenu
    nestedMenuActionsList = [QAction('testA', parent=nested_action_parent),
        QAction('testB', parent=nested_action_parent),
        QAction('testC', parent=nested_action_parent)]
    
        
    # nested_menu = owner.popMenu
    # # nested_menu = owner.subMenu
    # nested_menu.addAction(QAction('testA', parent=owner.popMenu))
    # nested_menu.addAction(QAction('testB', parent=owner.popMenu))
    # nested_menu.addSeparator()
    # nested_menu.addAction(QAction('testC', parent=owner.popMenu))
    # # Add a submenu
    # # owner.subMenu = QMenu(owner.popMenu)
    # # owner.subMenu = QMenu(owner)
    
    # owner.popMenu.addMenu(
    
    for an_action in popMenuActionsList:
        owner.popMenu.addAction(an_action)
        
    file_submenu = owner.popMenu.addMenu("Submenu")
    # file_submenu.addAction(button_action2) 
    for an_action in nestedMenuActionsList:
        file_submenu.addAction(an_action)
    
    # owner.subMenu = QMenu(owner.popMenu)
    # owner.subMenu = QMenu(owner)
    # owner.subMenu.addAction(QAction('testA', owner.popMenu))
    # owner.subMenu.addAction(QAction('testB', owner.popMenu))
    # owner.subMenu.addSeparator()
    # owner.subMenu.addAction(QAction('testC', owner.popMenu))
    # subMenuRemoveAction = owner.popMenu.addMenu(owner.subMenu)

    return owner.popMenu
 
 
 
# testCustomContextMenu

if __name__ == '__main__':
    # don't auto scale.
    QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    app = QApplication(sys.argv)
    # app.setStyleSheet('''
    #     QWidget {
    #         font-size: 30px;
    #     }
    # ''')
    
    # widget = LocalMenus_AddRenderable()
    # provided_menu = widget.ui.menuAdd_Renderable
    # myApp = MyApp(custom_menu=provided_menu)
    # widget.show()
    
    myApp = MyApp(custom_menu=None)
    provided_menu = create_custom_context_menu(myApp)
    print(f'provided_menu: {provided_menu}')
    myApp.custom_menu = provided_menu
    myApp.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')

