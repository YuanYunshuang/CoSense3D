import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from interface.view.graph_items import LineItem, RectangleItem


class ObjectViewWidget(pg.PlotWidget):

    def __init__(self, points, box, plane, parent=None):
        super().__init__(parent)
        assert plane in ['xy', 'zx', 'yz']
        self.points = points
        self.box = box
        self.plane = plane

        self.scatter_item = pg.ScatterPlotItem(
            pos=self.get2DPoints(), size=2, brush='b'
        )
        self.addItem(self.scatter_item)

        # Add the bounding box
        rect = self.getRectangle()
        self.rect_item = RectangleItem(rect)
        self.addItem(self.rect_item)

        self.setBoundries()
        self.setRotationAx()
        self.setAspectRatio()
        self.setAxisFixedRange()
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setStyleSheet("border: 1px solid gray;")
        self.resetChanges()

    def resetChanges(self):
        self.dx = 0
        self.dy = 0
        self.dr = 0
        self.dw = 0
        self.dh = 0

    @property
    def active(self):
        return self.rect_item.active or self.rotation_axis.active\
            or self.boundry_is_active

    @property
    def boundry_is_active(self):
        return self.leftbd.active or self.rightbd.active \
            or self.topbd.active or self.bottombd.active

    def handle_key_press(self, event):
        if event.key() == QtCore.Qt.Key_A:
            if self.rect_item.active:
                self.dx -= 0.1
            elif self.leftbd.active:
                self.dx -= 0.05
                self.dw += 0.1
            elif self.rightbd.active:
                self.dx -= 0.05
                self.dw -= 0.1
        elif event.key() == QtCore.Qt.Key_D:
            if self.rect_item.active:
                self.dx += 0.1
            elif self.leftbd.active:
                self.dx += 0.05
                self.dw -= 0.1
            elif self.rightbd.active:
                self.dx += 0.05
                self.dw += 0.1
        elif event.key() == QtCore.Qt.Key_W:
            if self.rect_item.active:
                self.dy += 0.1
            elif self.topbd.active:
                self.dy += 0.05
                self.dh += 0.1
            elif self.bottombd.active:
                self.dy -= 0.05
                self.dh -= 0.1
        elif event.key() == QtCore.Qt.Key_S:
            if self.rect_item.active:
                self.dy -= 0.1
            elif self.topbd.active:
                self.dy -= 0.05
                self.dh -= 0.1
            elif self.bottombd.active:
                self.dy -= 0.05
                self.dh += 0.1
        elif event.key() == QtCore.Qt.Key_Q:
            self.dr += 1
        elif event.key() == QtCore.Qt.Key_E:
            self.dr -= 1
        super().keyPressEvent(event)

    def handle_mouse_event(self, evt):
        if evt.type() == QtCore.QEvent.MouseButtonPress:
            if evt.button() == QtCore.Qt.LeftButton:
                if self.line_active:
                    self.edit_direction_start = evt.pos()
                else:
                    self.edit_size_start = evt.pos()
                self.timer.start()
        # elif evt.type() == QtCore.QEvent.MouseButtonRelease:

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPos = self.scatter_item.mapFromDevice(event.pos())
            if self.line_active:
                self.edit_direction_start = self.scatter_item.mapFromDevice(event.pos())
            else:
                self.edit_pos_start = self.scatter_item.mapFromDevice(event.pos())
        # super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            transform = QtGui.QTransform()
            # currPos = self.scatter_item.mapFromDevice(event.pos())
            # diff = currPos - self.lastPos
            # alpha = 0.2
            # smoothedDiff = alpha * diff + (1 - alpha) * self.lastDiff
            # self.move(self.pos() + smoothedDiff)
            # self.lastPos = currPos
            # self.lastDiff = smoothedDiff
            if getattr(self, 'edit_direction_start', False):
                currPos = self.scatter_item.mapFromDevice(event.pos())
                dx = currPos.x() - self.edit_direction_start.x()
                dy = currPos.y() - self.edit_direction_start.x()
                transform.rotate(np.arctan2(dy, dx) - np.pi / 2)
                self.edit_direction_start = False
            elif getattr(self, 'edit_pos_start', False):
                currPos = self.scatter_item.mapFromDevice(event.pos())
                dx = currPos.x() - self.edit_pos_start.x()
                dy = currPos.y() - self.edit_pos_start.y()
                transform.translate(dx, dy)
                self.edit_size_start = False
            self.scatter_item.setTransform(transform)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            if getattr(self, 'edit_direction_start', False):
                self.edit_direction_start = False
            elif getattr(self, 'edit_size_start', False):
                self.edit_pos_start = False
        else:
            super().mouseReleaseEvent(event)

    def get2DPoints(self):
        if self.plane == 'xy':
            x = self.points[:, 1]
            y = self.points[:, 0]
        elif self.plane == 'zx':
            x = self.points[:, 0]
            y = self.points[:, 2]
        elif self.plane == 'yz':
            x = self.points[:, 1]
            y = self.points[:, 2]
        else:
            raise NotImplementedError

        return np.stack([x, y], axis=1)

    def resetData(self, points, box):
        self.points = points
        self.box = box
        points = self.get2DPoints()
        self.scatter_item.setData(pos=points)
        self.rect_item.setRect(*self.getRectangle())
        self.updateRotationAx()
        self.updateBoundries()
        self.resetChanges()
        self.setAspectRatio()
        self.setAxisFixedRange()
        self.update()

    def setBoundries(self):
        # lines
        tl, tr, bl, br = self.getCorners()
        self.leftbd = LineItem(tl + bl)
        self.addItem(self.leftbd)
        self.rightbd = LineItem(tr + br)
        self.addItem(self.rightbd)
        self.topbd = LineItem(tl + tr)
        self.addItem(self.topbd)
        self.bottombd = LineItem(bl + br)
        self.addItem(self.bottombd)

    def updateBoundries(self):
        tl, tr, bl, br = self.getCorners()
        self.leftbd.setLine(*(tl + bl))
        self.rightbd.setLine(*(tr + br))
        self.topbd.setLine(*(tl + tr))
        self.bottombd.setLine(*(bl + br))

    def setRotationAx(self):
        ax = self.getRotationAxis()
        self.rotation_axis = LineItem(ax)
        self.addItem(self.rotation_axis)

    def updateRotationAx(self):
        self.rotation_axis.setLine(*self.getRotationAxis())

    def getRectangle(self):
        if self.plane == 'xy':
            x = self.box[1] - self.box[4] / 2  # real y
            y = self.box[0] - self.box[3] / 2  # real x
            w = self.box[4]  # real w
            h = self.box[3]  # real l
        elif self.plane == 'zx':
            x = self.box[0] - self.box[3] / 2  # real x
            y = self.box[2] - self.box[5] / 2  # real z
            w = self.box[3]  # real l
            h = self.box[5]  # real h
        elif self.plane == 'yz':
            x = self.box[1] - self.box[4] / 2  # real y
            y = self.box[2] - self.box[5] / 2  # real z
            w = self.box[4]  # real w
            h = self.box[5]  # real h
        else:
            raise NotImplementedError
        return x, y, w, h

    def getRotationAxis(self):
        if self.plane == 'xy':
            # front direction, point to the view top
            y1 = self.box[0] + self.box[3] / 2
            x1 = self.box[1]
            y2 = self.box[0] + self.box[3]
            x2 = self.box[1]
        elif self.plane == 'zx':
            # front direction, point to the view right
            x1 = self.box[0] + self.box[3] / 2
            y1 = self.box[2] + self.box[5] / 2
            x2 = self.box[0] + self.box[3]
            y2 = self.box[2] + self.box[5] / 2
        elif self.plane == 'yz':
            # front direction, point to the view top
            x1 = self.box[1]
            y1 = self.box[2] + self.box[5] / 2
            x2 = self.box[1]
            y2 = self.box[2] + self.box[5]
        else:
            raise NotImplementedError
        return x1, y1, x2, y2

    def getCorners(self):
        if self.plane == 'xy':
            tl = [self.box[1] - self.box[4] / 2, self.box[0] + self.box[3] / 2]
            tr = [self.box[1] + self.box[4] / 2, self.box[0] + self.box[3] / 2]
            br = [self.box[1] + self.box[4] / 2, self.box[0] - self.box[3] / 2]
            bl = [self.box[1] - self.box[4] / 2, self.box[0] - self.box[3] / 2]
        elif self.plane == 'zx':
            tl = [self.box[0] - self.box[3] / 2, self.box[2] + self.box[5] / 2]
            tr = [self.box[0] + self.box[3] / 2, self.box[2] + self.box[5] / 2]
            br = [self.box[0] + self.box[3] / 2, self.box[2] - self.box[5] / 2]
            bl = [self.box[0] - self.box[3] / 2, self.box[2] - self.box[5] / 2]
        elif self.plane == 'yz':
            tl = [self.box[1] - self.box[4] / 2, self.box[2] + self.box[5] / 2]
            tr = [self.box[1] + self.box[4] / 2, self.box[2] + self.box[5] / 2]
            bl = [self.box[1] - self.box[4] / 2, self.box[2] - self.box[5] / 2]
            br = [self.box[1] + self.box[4] / 2, self.box[2] - self.box[5] / 2]
        else:
            raise NotImplementedError
        return tl, tr, bl, br

    def setAspectRatio(self):
        if self.plane == 'xy':
            self.setAspectLocked(self.box[4] / self.box[3])
        elif self.plane == 'zx':
            self.setAspectLocked(self.box[3] / self.box[5])
        elif self.plane == 'yz':
            self.setAspectLocked(self.box[4] / self.box[5])

    def setAxisFixedRange(self):
        if self.plane == 'xy':
            rect = QtCore.QRectF(
                - self.box[1] - self.box[4],
                self.box[0] - self.box[3],
                self.box[4] * 2,
                self.box[3] * 2
            )
            self.setRange(rect)
        elif self.plane == 'zx':
            rect = QtCore.QRectF(
                self.box[0] - self.box[3],
                self.box[2] - self.box[5],
                self.box[3] * 2,
                self.box[5] * 2
            )
            self.setRange(rect)
        elif self.plane == 'yz':
            rect = QtCore.QRectF(
                self.box[1] - self.box[4],
                self.box[2] - self.box[5],
                self.box[4] * 2,
                self.box[5] * 2
            )
            self.setRange(rect)


class ObjectEditor(QtWidgets.QWidget):
    def __init__(self, frame, pcd, label, status=''):
        super().__init__()
        self.frame = frame
        self.pcd = pcd[:, :3]
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        self.label = label
        self.label_transformed = copy.deepcopy(label)
        self.status = status

        # set frame label
        vbox = QtWidgets.QVBoxLayout()
        qlabel = QtWidgets.QLabel(f"{frame}.{status}")
        qlabel.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(qlabel)

        # set 2d views
        points, box = self.toLabelCoor(self.pcd, self.label)
        self.plt_xy = ObjectViewWidget(points, box, 'xy')
        self.plt_zx = ObjectViewWidget(points, box, 'zx')
        self.plt_yz = ObjectViewWidget(points, box, 'yz')
        vbox.addWidget(self.plt_xy)
        vbox.addWidget(self.plt_zx)
        vbox.addWidget(self.plt_yz)

        vbox.setSpacing(0)
        self.setLayout(vbox)

    def toLabelCoor(self, pcd, label):
        self.box_center = label[:3]
        points = pcd - self.box_center.reshape(1, 3)
        R = Rotation.from_euler('xyz', label[-3:]).as_matrix()
        points = (R.T @ points.T).T
        box = copy.deepcopy(label)
        box[:3] = 0
        return points, box

    def child_transform(self):
        dt = np.zeros(3)
        ds = np.zeros(3)
        drot = np.zeros(3)
        if self.plt_xy.active:
            dt[0] += self.plt_xy.dy  # 3d pcd x = plot y
            dt[1] += self.plt_xy.dx    # 3d pcd y = plot x
            drot[2] -= self.plt_xy.dr  # yaw
            ds[1] += self.plt_xy.dw
            ds[0] += self.plt_xy.dh
        elif self.plt_zx.active:
            dt[2] += self.plt_zx.dy  # 3d pcd z = plot y
            dt[0] += self.plt_zx.dx    # 3d pcd x = plot x
            drot[1] += self.plt_zx.dr  # pitch
            ds[0] += self.plt_zx.dw  # pitch
            ds[2] += self.plt_zx.dh  # pitch
        elif self.plt_yz.active:
            dt[1] += self.plt_yz.dx  # 3d pcd y = plot x
            dt[2] += self.plt_yz.dy    # 3d pcd z = plot y
            drot[0] += self.plt_yz.dr  # roll
            ds[1] += self.plt_yz.dw  # roll
            ds[2] += self.plt_yz.dh  # roll

        return drot, ds, dt

    def keyPressEvent(self, evt: QtGui.QKeyEvent) -> None:
        if self.plt_xy.active:
            self.plt_xy.handle_key_press(evt)
        elif self.plt_zx.active:
            self.plt_zx.handle_key_press(evt)
        elif self.plt_yz.active:
            self.plt_yz.handle_key_press(evt)
        self.update_plots()

    def applyTransform(self, drot, ds, dt):
        self.label_transformed[-3:] += np.deg2rad(drot)
        self.label_transformed[3:6] += ds
        self.label_transformed[:3] += dt

    def update_plots(self):
        drot , ds, dt = self.child_transform()
        if sum([abs(x) for x in dt + drot]) > 0:
            self.applyTransform(drot , ds, dt)
            new_points, new_box = self.toLabelCoor(self.pcd, self.label_transformed)
            self.plt_xy.resetData(new_points, new_box)
            self.plt_zx.resetData(new_points, new_box)
            self.plt_yz.resetData(new_points, new_box)
            self.update()

    def data(self):
        """Return current data"""
        return {
            self.frame: {
                'label': self.label_transformed,
                'status': self.status
            }
        }


