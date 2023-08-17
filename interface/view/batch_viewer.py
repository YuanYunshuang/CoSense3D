import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, \
    QGridLayout, QVBoxLayout, QPushButton, QDialog, QLabel
from PyQt5.QtCore import Qt, QEvent, QTimer
import numpy as np

from interface.view.label_editor import ObjectEditor


class BatchViwer(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Batch View")
        self.controller = None
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setSpacing(5)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.saveLabel)
        self.timer.start()

    def show_plots(self):
        # Clear any previous plots from the jump out window
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            widget.setParent(None)

        batch_data = self.controller.get_batch_plot_data()
        if batch_data is None:
            qlabel = QLabel("No box is selected to process.")
            self.grid_layout.addWidget(qlabel)
        else:
            # Create the subplots and add them to the dialog window
            frames = sorted(batch_data.keys())

            for i in range(2):
                for j in range(10):
                    index = i * 2 + j
                    if index >= len(frames):
                        self.show()
                        return
                    frame = frames[index]
                    data = batch_data[frame]
                    data['frame'] = frame
                    # Create a VLayout for box top, left, front views
                    editor = ObjectEditor(**data)
                    editor.installEventFilter(self)
                    self.grid_layout.addWidget(editor, i, j)
        self.show()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            obj.setFocus(Qt.MouseFocusReason)
        return super().eventFilter(obj, event)

    def saveLabel(self):
        data_dict = {}
        for i in range(self.grid_layout.rowCount()):
            for j in range(self.grid_layout.columnCount()):
                item = self.grid_layout.itemAtPosition(i, j)
                if item is not None:
                    editor = item.widget()
                    data_dict.update(editor.data())
        self.controller.saveBatchViewLabels(data_dict)

if __name__ == "__main__":
    # Create the application and main window
    app = QApplication([])
    main_window = QWidget()

    # Create a button to open the jump out window
    button = QPushButton('Open Jump Out Window')
    dialog_win = BatchViwer(main_window)
    # Connect the button to the show_jump_out_window function
    button.clicked.connect(dialog_win.add_plots)

    # Add the button to the main window
    layout = QGridLayout()
    layout.addWidget(button)
    main_window.setLayout(layout)
    main_window.show()

    # Start the Qt event loop
    app.exec_()