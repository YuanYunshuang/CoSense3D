from typing import TYPE_CHECKING, List, Tuple, Union
Color4f = Tuple[float, float, float, float]  # type alias for type hinting


import logging
import ctypes

import numpy as np
from PyQt5.QtCore import Qt, QEvent
from PyQt5 import QtWidgets, QtGui, QtCore
from OpenGL import GL
from interface.view.glGraphicsItem import ScatterPlotItem
from interface.view.gl_viewer import GLViewWidget

SIZE_OF_FLOAT = ctypes.sizeof(ctypes.c_float)
TRANSLATION_FACTOR = 0.03


# Main widget for presenting the point cloud
class PointCloudWidget(GLViewWidget):

    def __init__(self, name: str, parent=None) -> None:
        super(PointCloudWidget, self).__init__(parent)
        self.setObjectName(name)

        self.setCameraPosition(distance=10, elevation=30, azimuth=45)
        self.pan(0, 0, 0)
        self.points = np.array([[0, 0, 0]])
        self.pcd_item = ScatterPlotItem(pos=self.points, size=3, glOptions='opaque')
        self.addItem(self.pcd_item)

        # point cloud data
        self.pcd = None

    def updatePCD(self, points, *args):
        depth_enabled = GL.glGetBooleanv(GL.GL_DEPTH_TEST)
        print('viwer updatePoints before:', depth_enabled)
        self.points = points
        self.pcd_item.setData(pos=points)
        depth_enabled = GL.glGetBooleanv(GL.GL_DEPTH_TEST)
        print('viwer updatePoints after:', depth_enabled)

    def paintGL(self, region=None, viewport=None, useItemNames=False):
        depthRange = np.zeros(2, dtype=np.float32)
        GL.glGetFloatv(GL.GL_DEPTH_RANGE, depthRange)

        super().paintGL()
        self.draw_depth_buffer()

    def get_world_coords(self, evt: QEvent):
        # Retrieve the position of the event in widget coordinates
        x = evt.pos().x() * self.DEVICE_PIXEL_RATIO
        y = self.height() - evt.pos().y() * self.DEVICE_PIXEL_RATIO
        z = self.get_depth(x, y)

        # Convert the widget coordinates to NDC
        x = x / self.width()
        y = y / self.height()

        # Convert the NDC coordinates to clip coordinates
        clip_x = 2.0 * x - 1.0
        clip_y = 2.0 * y - 1.0

        # pcd_pose = self.pcd_item.mapFromView(Vector(x, y, z))
        view_matrix = self.viewMatrix()
        projection_matrix = self.projectionMatrix()
        screen_pos = QtGui.QVector3D(clip_x, clip_y, z)
        # screen --> camera --> world
        world_pos = (
            view_matrix.inverted()[0] * projection_matrix.inverted()[0] * screen_pos
        )
        # world --> object (pcd object space)
        pcd_pos = self.pcd_item.mapFromView(world_pos)
        return pcd_pos

    def draw_depth_buffer(self):
        # self.makeCurrent()
        # Get the OpenGL extensions string
        depth_enabled = GL.glGetBooleanv(GL.GL_DEPTH_TEST)
        print(depth_enabled)
        # Retrieve the dimensions of the framebuffer
        viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        width, height = viewport[2], viewport[3]

        # Create a buffer to hold the depth values
        # depth_buffer = np.zeros((height, width), dtype=np.float32)

        # Read the depth buffer into the buffer
        depth_buffer = GL.glReadPixels(0, 0, width, height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)

        # Convert the depth buffer to an image
        depth_image = (1 - (depth_buffer + 1) / 2) * 255
        depth_image = np.repeat(depth_image[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

        # Save the image to a file
        import imageio
        imageio.imwrite('/media/hdd/tmp/depth_image.png', depth_image)

