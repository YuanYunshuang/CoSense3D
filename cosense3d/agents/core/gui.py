import os
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget, QApplication

from cosense3d.agents.viewer.gl_viewer import GLViewer
from cosense3d.agents.viewer.output_viewer import OutputViewer
from cosense3d.agents.viewer.img_viewer import ImgViewer


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

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)

    def setupUI(self, cfg):
        self.tabs = QtWidgets.QTabWidget()

        self.glViewer0 = GLViewer('MAINVIEW', self)
        self.tabs.addTab(self.glViewer0, 'GLViewer')

        self.img_viewer = ImgViewer()
        self.tabs.addTab(self.img_viewer, 'ImgViewer')

        self.output_viewer = OutputViewer(**cfg['output_viewer'])
        self.tabs.addTab(self.output_viewer, 'OutputViewer')

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
        self.tools = ['start', 'stop', 'step']
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
        self.button_start.clicked.connect(self.start)
        self.button_stop.clicked.connect(self.stop)

    def step(self):
        self.runner.step()
        data = self.runner.vis_data()
        self.glViewer0.refresh(data)
        self.img_viewer.refresh(data)
        self.output_viewer.refresh(data)
        if self.runner.iter == self.runner.total_iter:
            self.timer.stop()

    def start(self):
        self.timer.start(100)  # Trigger the animate method every 100ms

    def stop(self):
        self.timer.stop()






