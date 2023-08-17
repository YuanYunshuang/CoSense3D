from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QEvent, QObject, Qt
from OpenGL import GL
from functools import partial

from interface.ui import *
from interface.view.viewer import PointCloudWidget
from interface.view.batch_viewer import BatchViwer
from interface.view.object_viewer import ObjectViewer
from interface.view.frame_viewer import FrameViewer
from cosense3d.dataset.toolkit.cosense import csColors


class Menu:
    def __init__(self, cfg: dict, win: QtWidgets.QMainWindow):
        self.cfg = cfg
        self.win = win
        for mname, actions in cfg.items():
            minst = QtWidgets.QMenu(mname[0].upper() + mname[1:], win)
            setattr(self, mname, minst)
            for action in actions.keys():
                aname = action[0].upper() + action[1:]
                action_inst = QtWidgets.QAction(aname, win)
                setattr(self, mname + aname, action_inst)
                minst.addAction(action_inst)
                win.menuBar().addMenu(minst)

    def connect_events(self, controller):
        for mname, actions in self.cfg.items():
            for action, connet_fn in actions.items():
                attr = mname + action[0].upper() + action[1:]
                getattr(self, attr).triggered.connect(
                    getattr(controller, connet_fn)
                )


class GUI(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(GUI, self).__init__()
        self.controller = None
        self.header_height = 30
        self.setupUI()
        self.setWindowTitle("TAL annotator")

        # Set window size to screen size
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        width, height = screen.width(), screen.height()
        self.setGeometry(0, 0, width, height)

        # Start event cycle
        # self.timer = QtCore.QTimer(self)
        # self.timer.setInterval(60)  # period, in milliseconds
        # self.timer.timeout.connect(self.loop_gui)
        # self.timer.start()

    def init_gui(self):
        self.controller.init_controller()
        # connect all events
        self.connect_events_to_funcs()

    def keyPressEvent(self, evt: QtGui.QKeyEvent) -> None:
        if evt.key() == Qt.Key_3:
            evt.accept()
            self.controller.last_frame()
        if evt.key() == Qt.Key_4:
            evt.accept()
            self.controller.next_frame()

    def setController(self, controller):
        self.controller = controller
        self.glWidget0.controller = controller
        self.batch_viewer.controller = controller

    def setupUI(self):
        # OpenGL mainview
        # self.glWidgetcontainer = QtWidgets.QWidget(self)
        # self.glWidgetcontainer.setGeometry(QtCore.QRect(0, 0, 800, 600 - self.header_height))
        self.glWidget0 = PointCloudWidget('MAINVIEW', self)
        self.setCentralWidget(self.glWidget0)

        self.batch_viewer = BatchViwer(self.centralWidget())
        self.object_viewer = ObjectViewer(self.centralWidget())
        self.frame_viewer = FrameViewer(self.centralWidget())

        self.menu = Menu(MENU, self)
        self.get_toolbar()

    def get_toolbar(self):
        self.toolbar = self.addToolBar("Toolbar")
        self.infos = ['scene', 'frame', 'object', 'type']
        self.classes = ['type']
        self.tools = ['batch_view', 'auto', 'semi_auto', 'interpolate', 'save']
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
                qcombo.setStyleSheet(open("interface/ui/css/combobox.css", "r").read())
            setattr(self, f'label_{name}', qlabel)
            setattr(self, f'combo_{name}', qcombo)
            setattr(self, f'cur_{name}', None)

            self.toolbar.addWidget(getattr(self, f'label_{name}'))
            self.toolbar.addWidget(getattr(self, f'combo_{name}'))

        for name in self.tools:
            bname = f'{name[0].upper()}{name[1:]}:'
            qbutton = QtWidgets.QToolButton()
            qbutton.setText(bname)
            qbutton.setIcon(QtGui.QIcon(f"./interface/ui/icons/{name}.png"))
            w = qbutton.sizeHint().width() + 1
            qbutton.setMaximumWidth(w)
            setattr(self, f'button_{name}', qbutton)
            self.toolbar.addWidget(getattr(self, f'button_{name}'))

        # add items to combox for types of objects
        self.combo_type.clear()
        model = self.combo_type.model()
        for cls, color in csColors.items():
            item = QtGui.QStandardItem(cls)
            item.setBackground(QtGui.QColor(*color))
            # font = item.font()
            # font.setPointSize(10)
            # item.setFont(font)
            model.appendRow(item)
        self.updateType()

    def update_toolbar_info(self, name, items):
        combobox = getattr(self, f'combo_{name}')
        combobox.clear()
        combobox.addItem('------')
        combobox.addItems(items)

        depth_enabled = GL.glGetBooleanv(GL.GL_DEPTH_TEST)
        print('update_toolbar_info after:', depth_enabled)

    def connect_events_to_funcs(self):
        self.menu.connect_events(self.controller)

        for name in self.infos:
            combobox = getattr(self, f'combo_{name}')
            combobox.currentIndexChanged.connect(
                getattr(self.controller, f"change_{name}")
            )

        self.button_batch_view.clicked.connect(self.batch_viewer.show_plots)
        self.button_save.clicked.connect(self.controller.save)
        # self.button_auto.clicked.connect(self.controller.track)
        self.object_viewer.Type.currentIndexChanged.connect(self.controller.change_active_object_type)

    def reconnect_checkboxes(self):
        for ckbox in self.frame_viewer.lidar_checkboxes:
            ckbox.stateChanged.connect(
                partial(self.controller.frame_viewer_checkbox_changed, text=ckbox.text()))

    def updateScenario(self, scenario_info):
        self.frame_viewer.update_checkboxes(scenario_info['lidar_ids'])
        self.reconnect_checkboxes()

    def updateFrame(self, index, frame_info):
        self.combo_frame.setCurrentIndex(index)
        self.combo_frame.setCurrentText(frame_info['frame'])
        self.frame_viewer.updateInfo(frame_info)
        # self.update()

    def updateType(self):
        name = self.combo_type.currentText()
        pal = self.combo_type.palette()
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor(*csColors[name]))
        self.combo_type.setPalette(pal)

    # def resizeEvent(self, event):
    #     h = self.header_height
    #     # Resize the GL widget to match the current window size
    #     self.glWidget0.setGeometry(0, h, self.width(), self.height() - h)
    #     self.glWidget0.update()

