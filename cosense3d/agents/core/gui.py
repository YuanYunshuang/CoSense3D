import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget, QApplication

from cosense3d.agents.viewer.gl_viewer import GLViewer
from cosense3d.agents.viewer.canvas_viewer import CanvasViewer


class GUI(QtWidgets.QMainWindow):
    def __init__(self, mode, cfg) -> None:
        super(GUI, self).__init__()
        self.mode = mode
        self.header_height = 30
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.css_dir = os.path.join(path, 'viewer', 'css')
        self.setupUI(cfg)
        self.setWindowTitle("Cosense3D")

        # Set window size to screen size
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        width, height = screen.width(), screen.height()
        self.setGeometry(0, 0, width, height)

    def setupUI(self, cfg):
        self.tabs = QtWidgets.QTabWidget()
        self.glViewer0 = GLViewer('MAINVIEW', self)
        self.tabs.addTab(self.glViewer0, 'GLViewer')
        self.canvas = CanvasViewer(**cfg['canvas_viewer'])
        self.tabs.addTab(self.canvas, 'Canvas')
        self.setCentralWidget(self.tabs)
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
            # if not name=='type':
            #     css_file = f"{self.css_dir}/combobox.css"
            #     qcombo.setStyleSheet(open(css_file, "r").read())
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
        data = self.runner.vis_data()
        self.glViewer0.refresh(data)
        self.canvas.refresh(data)





