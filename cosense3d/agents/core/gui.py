import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget, QApplication

from interface.view.viewer import PointCloudWidget

class GUI(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(GUI, self).__init__()
        self.header_height = 30
        self.setupUI()
        self.setWindowTitle("Cosense3D")

        # Set window size to screen size
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        width, height = screen.width(), screen.height()
        self.setGeometry(0, 0, width, height)

    def setupUI(self):
        # OpenGL mainview
        # self.glWidgetcontainer = QtWidgets.QWidget(self)
        # self.glWidgetcontainer.setGeometry(QtCore.QRect(0, 0, 800, 600 - self.header_height))
        self.glWidget0 = PointCloudWidget('MAINVIEW', self)
        self.setCentralWidget(self.glWidget0)
        self.get_toolbar()

    def setRunner(self, runner):
        self.runner = runner

    def initGUI(self):
        # connect all events
        self.connect_events_to_funcs()

    def get_toolbar(self):
        self.toolbar = self.addToolBar("Toolbar")
        self.infos = ['scene', 'frame']
        self.tools = ['step']
        # add label combo pairs
        for name in self.infos:
            qlabel = QtWidgets.QLabel(f' {name[0].upper()}{name[1:]}:')
            w1 = qlabel.sizeHint().width()
            qlabel.setMinimumWidth(w1 + 5)
            qlabel.setMaximumWidth(w1 + 25)
            qcombo = QtWidgets.QComboBox()
            qcombo.addItem('---------')
            w2 = qcombo.sizeHint().width()
            qcombo.setMinimumWidth(w2 + 25)
            qcombo.setMaximumWidth(w2 + 50)
            if not name=='type':
                qcombo.setStyleSheet(open("../../interface/ui/css/combobox.css", "r").read())
            setattr(self, f'label_{name}', qlabel)
            setattr(self, f'combo_{name}', qcombo)
            setattr(self, f'cur_{name}', None)

            self.toolbar.addWidget(getattr(self, f'label_{name}'))
            self.toolbar.addWidget(getattr(self, f'combo_{name}'))

        for name in self.tools:
            bname = f'{name[0].upper()}{name[1:]}'
            qbutton = QtWidgets.QToolButton()
            qbutton.setText(bname)
            # qbutton.setIcon(QtGui.QIcon(f"./interface/ui/icons/{name}.png"))
            w = qbutton.sizeHint().width() + 1
            qbutton.setMaximumWidth(w)
            setattr(self, f'button_{name}', qbutton)
            self.toolbar.addWidget(getattr(self, f'button_{name}'))

    def connect_events_to_funcs(self):
        self.button_step.clicked.connect(self.step)

    def step(self):
        self.runner.step()
        data = self.runner.controller.vis_data()
        self.glWidget0.updateFrameData(*data)





