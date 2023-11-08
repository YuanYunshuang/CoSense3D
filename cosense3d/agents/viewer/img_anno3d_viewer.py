import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from cosense3d.utils import vislib


class ImgAnno3DViewer(FigureCanvasQTAgg):

    def __init__(self, dpi=50):
        self.fig = Figure(dpi=dpi)
        super(ImgAnno3DViewer, self).__init__(self.fig)

    def refresh(self, data):
        if len(data['input']['imgs']) == 0:
            return
        self.fig.clear()
        n_cavs = len(data['input']['imgs'])
        n_imgs = len(list(data['input']['imgs'].values())[0])
        cav_ids = sorted(list(data['input']['imgs'].keys()))
        for i, cav_id in enumerate(cav_ids):
            bboxes3d = np.array(list(data['input']['local_labels'][cav_id].values()))[:, [1, 2, 3, 4, 5, 6, 9]]
            for j in range(n_imgs):
                ax = self.fig.add_subplot(n_cavs, n_imgs, i * n_imgs + j + 1)
                img = data['input']['imgs'][cav_id][j].astype(np.uint8)
                lidar2img = data['input']['lidar2img'][cav_id][j]
                vislib.draw_3d_points_boxes_on_img(ax, img, lidar2img, boxes=bboxes3d)
        self.draw()
