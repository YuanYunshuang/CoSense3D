import OpenGL.GL as GL
from OpenGL import GLU
from PyQt5 import QtGui, QtOpenGL, QtCore, QtWidgets
import numpy as np
import logging
import ctypes
import queue
from interface.view.utils import gl_depth_deactive

SIZE_OF_FLOAT = ctypes.sizeof(ctypes.c_float)
TRANSLATION_FACTOR = 0.03
POINT_SIZE = 3
BG_COLOR = (0.0, 0.0, 0.0, 1)


class PointCloudWidget(QtWidgets.QOpenGLWidget):
    NEAR_PLANE = 0.1
    FAR_PLANE = 1000
    FOV = 45

    def __init__(self, name: str, parent=None) -> None:
        self.parent = parent
        QtWidgets.QOpenGLWidget.__init__(self, parent)
        self.setObjectName(name)
        # self.setMouseTracking(
        #     True
        # )  # mouseMoveEvent is called also without button pressed
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self.modelview = None
        self.projection = None
        self.world2pcd = None
        self.DEVICE_PIXEL_RATIO = (
            self.devicePixelRatioF()
        )  # 1 = normal; 2 = retina display
        self.tasks = queue.Queue()

        # point cloud data and buffer
        self.pcd = None
        self.vbo_idx = None
        self.vbo = None

        # label data
        self.points = None
        self.lines = None

    def get_vertex_buffer(self, pcd=None):
        if pcd is not None:
            self.pcd = pcd
        attributes = self.pcd.flatten()
        buffer_data = (ctypes.c_float * len(attributes))(*attributes)
        buffer_size = SIZE_OF_FLOAT * len(attributes)

        # GL.glDeleteBuffers(1, self.vbo)
        self.vbo = GL.glGenBuffers(1, self.vbo_idx)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer_size, buffer_data, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def updatePCD(self, pcd):
        self.pcd = pcd
        self.update()

    def initializeGL(self) -> None:
        self.vbo_idx = 1
        GL.glClearColor(*BG_COLOR)  # screen background color
        GL.glEnable(GL.GL_DEPTH_TEST)  # for visualization of depth
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_BLEND)  # enable transparency
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        logging.info(f"Intialized {self.objectName()}.")

    def resizeGL(self, width, height) -> None:
        logging.info(f"Resized {self.objectName()}.")
        GL.glViewport(0, 0, width, height)
        self.viewport = [0, 0, width, height]
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect = width / float(height)

        GLU.gluPerspective(self.FOV, aspect, self.NEAR_PLANE, self.FAR_PLANE)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        self.paintGL()

    def paintGL(self) -> None:
        if self.pcd is not None:
            GL.glClearColor(*BG_COLOR)  # screen background color
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self.get_vertex_buffer()
            GL.glLoadIdentity()
            pcd_min = np.min(self.pcd[:, :3], axis=0)
            pcd_max = np.max(self.pcd[:, :3], axis=0)
            pcd_center = - (pcd_max + pcd_min) / 2
            max_xy = np.abs(self.pcd[:, :2].max())
            z = 80 / np.tan(np.deg2rad(self.FOV) / 2)
            # pcd[:, 2] -= (z + pcd_center[2])
            pcd_center[2] -= z
            GL.glTranslate(*pcd_center)
            self.world2pcd = pcd_center
            self.modelview = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
            self.projection = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)

            GL.glPointSize(POINT_SIZE)
            # (12 bytes) : [xyz] * sizeof(float) or
            # (24 bytes) : [xyzrgb] * sizeof(float)
            num_attr = 3
            stride = num_attr * SIZE_OF_FLOAT
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointer(3, GL.GL_FLOAT, stride, None)
            GL.glColor4f(1.0, 1.0, 1.0, 1.0)

            GL.glDrawArrays(GL.GL_POINTS, 0, len(self.pcd))  # Draw the points
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            self.draw_depth_buffer()

    def eventFilter(self, evt_obj: 'QObject', evt: 'QEvent') -> bool:
        if evt.type() == QtCore.QEvent.MouseButtonPress:
            self.mousePressEvent(evt)
        else:
            evt.ignore()
        return False

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePos = lpos
        self.draw_depth_buffer()

    def draw_depth_buffer(self):
        # Get the OpenGL extensions string
        depth_enabled = GL.glGetBooleanv(GL.GL_DEPTH_TEST)
        print('draw_depth_buffer', depth_enabled)
        # Retrieve the dimensions of the framebuffer
        viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        width, height = viewport[2], viewport[3]

        # Create a buffer to hold the depth values
        depth_buffer = np.zeros((height, width), dtype=np.float32)

        # Read the depth buffer into the buffer
        GL.glReadPixels(0, 0, width, height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, depth_buffer)

        # Convert the depth buffer to an image
        depth_image = ((1 - depth_buffer) * 1000) * 255
        depth_image = np.repeat(depth_image[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

        # Save the image to a file
        import imageio
        imageio.imwrite('/media/hdd/tmp/depth_image.png', depth_image)