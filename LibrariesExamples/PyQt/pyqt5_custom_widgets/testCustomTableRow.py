import sys
from attrs import define, field, Factory
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton, QWidget

@define(slots=False)
class TableApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PyQt5 Table Row Example')
        
        layout = QVBoxLayout()
        
        # Initialize table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(1)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(['Name', 'Age', 'City'])
        
        # Add initial data
        self.tableWidget.setItem(0, 0, QTableWidgetItem("John"))
        self.tableWidget.setItem(0, 1, QTableWidgetItem("25"))
        self.tableWidget.setItem(0, 2, QTableWidgetItem("New York"))
        
        # Button to add a new row
        btnAddRow = QPushButton("Add Row", self)
        btnAddRow.clicked.connect(self.onAddRow)
        
        layout.addWidget(self.tableWidget)
        layout.addWidget(btnAddRow)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def onAddRow(self):
        rowPosition = self.tableWidget.rowCount()
        self.tableWidget.insertRow(rowPosition)
        
        # Optionally, you can also add new data to this row here
        # self.tableWidget.setItem(rowPosition, 0, QTableWidgetItem("Data1"))
        # ...

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = TableApp()
    mainWin.show()
    sys.exit(app.exec_())
