import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from cosense3d.utils.vislib import draw_points_boxes_plt


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, data_keys, width=5, height=4, dpi=100, title='plot', nrows=1, ncols=1):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle(title, fontsize=16)
        self.axes = fig.subplots(nrows, ncols)
        self.data_keys = data_keys
        super(MplCanvas, self).__init__(fig)

    def update_title(self, scenario, frame, cav_id):
        self.axes.set_title(f"{scenario[cav_id]}.{frame[cav_id]}")


class BaseLinePlot(MplCanvas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def refresh(self, data, **kwargs):
        self.draw()


class TrainViewer(QtWidgets.QWidget):
    def __init__(self, plots, cols=4, parent=None):
        super(TrainViewer, self).__init__(parent)
        self.cols = cols
        layout = QtWidgets.QGridLayout(self)
        self.plots = []
        for p in plots:
            plot = globals()[p['title']](**p)
            layout.addWidget(plot)
            self.plots.append(plot)

    def refresh(self, data, **kwargs):
        for plot in self.plots:
            plot.refresh(data)


