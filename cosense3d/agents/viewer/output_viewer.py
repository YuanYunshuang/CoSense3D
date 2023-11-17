import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from cosense3d.utils.vislib import draw_points_boxes_plt


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, width=5, height=4, dpi=100, title='plot', nrows=1, ncols=1):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle(title, fontsize=16)
        self.axes = fig.subplots(nrows, ncols)
        self.data = {}
        super(MplCanvas, self).__init__(fig)

    def update_title(self, meta, cav_id):
        self.axes.set_title(f"{meta['scenario'][cav_id]}.{meta['frame'][cav_id]}")


class BEVSparseCanvas(MplCanvas):
    def __init__(self, lidar_range=None, **kwargs):
        super().__init__(**kwargs)
        self.lidar_range = lidar_range

    def refresh(self, data):
        if 'bev' not in data:
            return
        for cav_id, data_dict in data['bev'].items():
            centers = data_dict['center'].cpu().numpy()
            conf = data_dict['conf'][:, 1:].detach().max(dim=-1).values.cpu().numpy()
            self.axes.clear()
            self.update_title(data['meta'], cav_id)
            self.scatter = self.axes.scatter(centers[:, 0], centers[:, 1],
                                             cmap='jet', c=conf, s=2, vmin=0, vmax=1)
            # self.scatter.set_array(conf)
            # self.scatter.set_offsets(centers)
            self.draw()
            break


class DetectionCanvas(MplCanvas):
    def __init__(self, lidar_range=None, **kwargs):
        super().__init__(**kwargs)
        self.lidar_range = lidar_range

    def refresh(self, data):
        if 'detection' not in data:
            return
        for cav_id, det_dict in data['detection'].items():
            self.axes.clear()
            self.update_title(data['meta'], cav_id)
            # plot points
            for points in data['input']['pcds'].values():
                draw_points_boxes_plt(
                    pc_range=self.lidar_range,
                    points=points,
                    ax=self.axes,
                    # return_ax=True
                )
            # plot centers
            if 'ctr' in det_dict:
                centers = det_dict['ctr'].detach().cpu().numpy()
                conf = det_dict['conf'][:, 0, 1].detach().cpu().numpy()
                mask = conf > 0.5
                centers = centers[mask]
                conf = conf[mask]
                self.axes.scatter(centers[:, 0], centers[:, 1],
                                  cmap='jet', c=conf, s=.1, vmin=0, vmax=1)
            # plot pcds and boxes
            gt_boxes = list(data['input']['global_labels'][cav_id].values())
            gt_boxes = np.array(gt_boxes)[:, [1, 2, 3, 4, 5, 6, 9]]
            pred_boxes = det_dict['box'].detach().cpu().numpy()
            draw_points_boxes_plt(
                pc_range=self.lidar_range,
                boxes_pred=pred_boxes,
                boxes_gt=gt_boxes,
                ax=self.axes,
                # return_ax=True
            )
            self.draw()
            break


class OutputViewer(QtWidgets.QWidget):
    def __init__(self, plots, parent=None):
        super(OutputViewer, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.plots = []
        for p in plots:
            plot = globals()[p['title']](**p)
            layout.addWidget(plot)
            self.plots.append(plot)

    def refresh(self, data):
        for plot in self.plots:
            plot.refresh(data)


