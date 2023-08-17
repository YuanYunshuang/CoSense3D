from PyQt5.QtWidgets import QLabel, QCheckBox
from PyQt5.QtGui import QStandardItem, QColor, QPalette
from cosense3d.dataset.toolkit.cosense import csColors
from interface.view.draggable_dialog import DraggableDialog


class FrameViewer(DraggableDialog):
    def __init__(self, parent=None):
        super().__init__(parent, 'Frame Info')
        self.add_label('Id:', 1, 0)
        self.layout.setRowMinimumHeight(1, 50)
        self.layout.setColumnMinimumWidth(0, 50)
        self.layout.setColumnMinimumWidth(1, 100)
        self.Id = self.add_label('-1', 1, 1)
        self.add_label('Lidars:', 2, 0)
        self.layout.setRowMinimumHeight(2, 50)
        self.lidar_checkboxes = []

    def add_label(self, name, row, col):
        label = QLabel(name)
        label.setStyleSheet(self.label_style)
        self.layout.addWidget(label, row, col)
        return label

    def add_lidar_checkbox(self, row, lidar_id):
        qckbox = QCheckBox(str(lidar_id))
        qckbox.setChecked(True)
        self.layout.addWidget(qckbox, row, 0)
        self.layout.setRowMinimumHeight(row, 50)
        return qckbox

    def updateInfo(self, info_dict):
       self.Id.setText(str(info_dict['frame']))

    def update_checkboxes(self, lidar_ids):
        for i, lidar_id in enumerate(lidar_ids):
            self.lidar_checkboxes.append(
                self.add_lidar_checkbox(3 + i, lidar_id)
            )
