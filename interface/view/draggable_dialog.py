from PyQt5.QtWidgets import QDialog, QGridLayout, QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent


class DraggableDialog(QDialog):
    def __init__(self, parent=None, name='Floating Dialog'):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 100);")
        self.label_style = "color: yellow; background-color: rgba(255, 255, 255, 0)"
        self.draggable = False
        self.offset = QPoint()

        self.layout = QGridLayout(self)
        self.title_label = QLabel(name)
        self.title_label.setStyleSheet("color: yellow;background-color: rgba(255, 255, 255, 0); font-weight: bold;")
        self.layout.addWidget(self.title_label, 0, 0, 1, 2)
        # label = QLabel("Label1")
        # label.setStyleSheet("color: yellow;background-color: rgba(255, 255, 255, 0)")
        # layout.addWidget(label)
        self.setLayout(self.layout)
        self.layout.setRowMinimumHeight(0, 50)
        self.move(0, 0)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.draggable = True
            self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.draggable:
            self.move(self.mapToParent(event.pos() - self.offset))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.draggable = False


