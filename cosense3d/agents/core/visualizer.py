import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget, QApplication


class Visualizer(object):
    def __init__(self, args):
        self.app = QApplication(args)
        self.main_window = MainWindow()

    def run(self):
        self.app.installEventFilter(self.main_window)

        desktop = QDesktopWidget().availableGeometry()
        width = (desktop.width() - self.main_window.width()) / 2
        height = (desktop.height() - self.main_window.height()) / 2

        self.main_window.move(width, height)
        self.main_window.init_gui()
        self.main_window.show()

        sys.exit(self.app.exec_())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.header_height = 30
        self.setupUI()
        self.setWindowTitle("Cosense3D")

        # Set window size to screen size
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        width, height = screen.width(), screen.height()
        self.setGeometry(0, 0, width, height)



