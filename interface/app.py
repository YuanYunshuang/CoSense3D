#!/usr/bin/python
import sys
import logging
logging.getLogger().setLevel(logging.DEBUG)

from PyQt5.QtWidgets import QDesktopWidget, QApplication

from interface.view.gui import GUI
from interface.control.controller import Controller
from cosense3d.tools.qt_tracker import Tracker
from cosense3d.tools.qt_singleton_tracker import SingletonTracker
from cosense3d.utils.misc import load_yaml

import os
from PyQt5.QtCore import QDir


class App(object):
    def __init__(self, args):
        path = os.path.dirname(os.path.abspath(__file__))
        QDir.addSearchPath('images', f"{path}/ui/icons")
        self.appcfg = load_yaml(os.path.join(os.path.dirname(path), 'config', 'app.yaml'))
        self.app = QApplication(args)
        self.gui = GUI()
        self.controller = Controller(self.appcfg.get('sensor_cfg', None))
        # TODO: add tracker back
        # tracker = Tracker(os.path.join(self.appcfg['dl_log_dir']['tracker'], 'config.yaml'))
        # singleton_tracker = SingletonTracker(os.path.join(
        #     self.appcfg['dl_log_dir']['singleton_tracker'], 'config.yaml'))

        # bridge controller and gui and tracker
        self.gui.setController(self.controller)
        self.controller.view = self.gui
        # self.controller.set_trackers(tracker, singleton_tracker)

    def run(self):
        self.app.installEventFilter(self.gui)

        # self.app.setStyle("Fusion")
        desktop = QDesktopWidget().availableGeometry()
        width = (desktop.width() - self.gui.width()) / 2
        height = (desktop.height() - self.gui.height()) / 2

        self.gui.move(width, height)
        self.gui.init_gui()
        # Start GUI
        self.gui.show()

        logging.info("Showing GUI...")
        sys.exit(self.app.exec_())