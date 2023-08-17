from PyQt5.QtWidgets import QLabel, QComboBox
from PyQt5.QtGui import QStandardItem, QColor, QPalette
from cosense3d.dataset.toolkit.cosense import csColors
from interface.view.draggable_dialog import DraggableDialog


class ObjectViewer(DraggableDialog):
    def __init__(self, parent=None):
        super().__init__(parent, 'Object Info')
        self.add_label('Id', 1, 0)
        self.Id = self.add_label('-1', 1, 1)
        self.add_label('Type', 2, 0)
        self.Type = self.add_type_combobox(2)
        self.hide()

    def add_label(self, name, row, col):
        label = QLabel(f"{name}: ")
        label.setStyleSheet(self.label_style)
        self.layout.addWidget(label, row, col)
        return label

    def add_type_combobox(self, row):
        qcombo = QComboBox()
        model = qcombo.model()
        for cls, color in csColors.items():
            item = QStandardItem(cls)
            item.setBackground(QColor(*color))
            model.appendRow(item)
        self.layout.addWidget(qcombo, row, 1)
        return qcombo

    def updateInfo(self, info_dict):
       self.Id.setText(str(info_dict['Id']))
       self.Type.setCurrentIndex(info_dict['Type'])
       pal = self.Type.palette()
       pal.setColor(QPalette.Button, QColor(*csColors[self.Type.currentText()]))
       self.Type.setPalette(pal)