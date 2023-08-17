import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QCheckBox

class CheckBoxExample(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        label = QLabel("Check the box to enable an option:")
        self.checkbox = QCheckBox("Option")

        # Connect the checkbox to a slot (function)
        self.checkbox.stateChanged.connect(self.checkbox_changed)

        layout.addWidget(label)
        layout.addWidget(self.checkbox)

        self.setLayout(layout)
        self.setWindowTitle("PyQt5 Checkbox Example")

    def checkbox_changed(self, state):
        if state == 2:  # Checked state
            print("Checkbox is checked")
        else:
            print("Checkbox is unchecked")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CheckBoxExample()
    ex.show()
    sys.exit(app.exec_())