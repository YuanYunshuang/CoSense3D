import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from cosense3d.utils import vislib


class ImgViewer(FigureCanvasQTAgg):

    def __init__(self, dpi=100):
        self.fig = Figure(dpi=dpi)
        super(ImgViewer, self).__init__(self.fig)

    def refresh(self, data):
        self.fig.clear()
        n_cavs = len(data['input']['imgs'])
        n_imgs = len(list(data['input']['imgs'].values())[0])
        for i, (k, imgs) in enumerate(data['input']['imgs'].items()):
            for j in range(n_imgs):
                ax = self.fig.add_subplot(n_cavs, n_imgs, i * n_imgs + j + 1)
                bboxes2d = data['input']['bboxes2d'][k][j]
                img = imgs[j]
                ax = vislib.draw_2d_bboxes_on_img(img, bboxes2d, ax)
