import matplotlib

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, title='plot'):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle('Plot', fontsize=16)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class CanvasViewer(QtWidgets.QWidget):
    def __init__(self, plots=['plot1', 'plot2'], parent=None):
        super(CanvasViewer, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        for p in plots:
            plot = MplCanvas(title=p)
            layout.addWidget(plot)
