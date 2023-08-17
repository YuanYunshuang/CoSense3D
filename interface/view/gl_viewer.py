import logging
import ctypes
import queue
from math import cos, radians, sin, tan

import numpy as np
from PyQt5 import QtGui, QtOpenGL, QtCore, QtWidgets
from pyqtgraph.Vector import Vector
from OpenGL.GL import *
import OpenGL.GL.framebufferobjects as glfbo  # noqa

SIZE_OF_FLOAT = ctypes.sizeof(ctypes.c_float)
TRANSLATION_FACTOR = 0.03
POINT_SIZE = 3
BG_COLOR = (0.0, 0.0, 0.0, 1)


class GLViewWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, name: str, parent=None, rotationMethod='euler') -> None:
        self.parent = parent
        QtWidgets.QOpenGLWidget.__init__(self, parent)
        self.setMouseTracking(
            True
        )  # mouseMoveEvent is called also without button pressed
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        if rotationMethod not in ["euler", "quaternion"]:
            raise ValueError("Rotation method should be either 'euler' or 'quaternion'")

        self.opts = {
            'center': Vector(0, 0, 0),  ## will always appear at the center of the widget
            'rotation': QtGui.QQuaternion(1, 0, 0, 0),  ## camera rotation (quaternion:wxyz)
            'distance': 10.0,  ## distance of camera from center
            'fov': 60,  ## horizontal field of view in degrees
            'elevation': 30,  ## camera's angle of elevation in degrees
            'azimuth': 45,  ## camera's azimuthal angle in degrees
            ## (rotation around z-axis 0 points along x-axis)
            'viewport': None,  ## glViewport params; None == whole widget
            ## note that 'viewport' is in device pixels
            'rotationMethod': rotationMethod
        }

        self.items = []

    def _updateScreen(self, screen):
        self._updatePixelRatio()
        if screen is not None:
            screen.physicalDotsPerInchChanged.connect(self._updatePixelRatio)
            screen.logicalDotsPerInchChanged.connect(self._updatePixelRatio)

    def _updatePixelRatio(self):
        event = QtGui.QResizeEvent(self.size(), self.size())
        self.resizeEvent(event)

    def showEvent(self, event):
        window = self.window().windowHandle()
        window.screenChanged.connect(self._updateScreen)
        self._updateScreen(window.screen())

    def deviceWidth(self):
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr)

    def deviceHeight(self):
        dpr = self.devicePixelRatioF()
        return int(self.height() * dpr)

    def getViewport(self):
        vp = self.opts['viewport']
        if vp is None:
            return (0, 0, self.deviceWidth(), self.deviceHeight())
        else:
            return vp

    def setProjection(self, region=None):
        m = self.projectionMatrix(region)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(np.array(m.data(), dtype=np.float32))

    def projectionMatrix(self, region=None):
        if region is None:
            region = (0, 0, self.deviceWidth(), self.deviceHeight())

        x0, y0, w, h = self.getViewport()
        dist = self.opts['distance']
        fov = self.opts['fov']
        nearClip = dist * 0.001
        farClip = dist * 1000.

        r = nearClip * tan(0.5 * radians(fov))
        t = r * h / w

        ## Note that X0 and width in these equations must be the values used in viewport
        left = r * ((region[0] - x0) * (2.0 / w) - 1)
        right = r * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
        bottom = t * ((region[1] - y0) * (2.0 / h) - 1)
        top = t * ((region[1] + region[3] - y0) * (2.0 / h) - 1)

        tr = QtGui.QMatrix4x4()
        tr.frustum(left, right, bottom, top, nearClip, farClip)
        return tr

    def setModelview(self):
        m = self.viewMatrix()
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(np.array(m.data(), dtype=np.float32))

    def viewMatrix(self):
        tr = QtGui.QMatrix4x4()
        tr.translate(0.0, 0.0, -self.opts['distance'])
        if self.opts['rotationMethod'] == 'quaternion':
            tr.rotate(self.opts['rotation'])
        else:
            # default rotation method
            tr.rotate(self.opts['elevation'] - 90, 1, 0, 0)
            tr.rotate(self.opts['azimuth'] + 90, 0, 0, -1)
        center = self.opts['center']
        tr.translate(-center.x(), -center.y(), -center.z())
        return tr

    def setCameraPosition(self, pos=None, distance=None, elevation=None, azimuth=None, rotation=None):
        if rotation is not None:
            # Alternatively, we could define that rotation overrides elevation and azimuth
            if elevation is not None:
                raise ValueError("cannot set both rotation and elevation")
            if azimuth is not None:
                raise ValueError("cannot set both rotation and azimuth")

        if pos is not None:
            self.opts['center'] = pos
        if distance is not None:
            self.opts['distance'] = distance

        if self.opts['rotationMethod'] == "quaternion":
            # note that "quaternion" mode modifies only opts['rotation']
            if elevation is not None or azimuth is not None:
                eu = self.opts['rotation'].toEulerAngles()
                if azimuth is not None:
                    eu.setZ(-azimuth-90)
                if elevation is not None:
                    eu.setX(elevation-90)
                self.opts['rotation'] = QtGui.QQuaternion.fromEulerAngles(eu)
            if rotation is not None:
                self.opts['rotation'] = rotation
        else:
            # note that "euler" mode modifies only opts['elevation'] and opts['azimuth']
            if elevation is not None:
                self.opts['elevation'] = elevation
            if azimuth is not None:
                self.opts['azimuth'] = azimuth
            if rotation is not None:
                eu = rotation.toEulerAngles()
                self.opts['elevation'] = eu.x() + 90
                self.opts['azimuth'] = -eu.z() - 90

        self.update()

    def cameraPosition(self):
        """Return current position of camera based on center, dist, elevation, and azimuth"""
        center = self.opts['center']
        dist = self.opts['distance']

        if self.opts['rotationMethod'] == "quaternion":
            pos = Vector(center - self.opts['rotation'].rotatedVector(Vector(0,0,dist) ))
        else:
            # using 'euler' rotation method
            elev = radians(self.opts['elevation'])
            azim = radians(self.opts['azimuth'])
            pos = Vector(
                center.x() + dist * cos(elev) * cos(azim),
                center.y() + dist * cos(elev) * sin(azim),
                center.z() + dist * sin(elev)
            )
        return pos

    def pan(self, dx, dy, dz, relative='global'):
        if relative == 'global':
            self.opts['center'] += QtGui.QVector3D(dx, dy, dz)
        else:
            raise ValueError("relative argument currently only support global")

        self.update()

    def pixelSize(self, pos):
        """
        Return the approximate size of a screen pixel at the location pos
        Pos may be a Vector or an (N,3) array of locations
        """
        cam = self.cameraPosition()
        if isinstance(pos, np.ndarray):
            cam = np.array(cam).reshape((1,)*(pos.ndim-1)+(3,))
            dist = ((pos-cam)**2).sum(axis=-1)**0.5
        else:
            dist = (pos-cam).length()
        # ? dist here is an approximate value, because near plane is far from object
        # visible x extent in view coor.
        xDist = dist * 2. * tan(0.5 * radians(self.opts['fov']))
        return xDist / self.width()

    def initializeGL(self):
        """
        Initialize items that were not initialized during addItem().
        """
        ctx = self.context()
        fmt = ctx.format()
        if ctx.isOpenGLES() or fmt.version() < (2, 0):
            verString = glGetString(GL_VERSION)
            raise RuntimeError(
                "pyqtgraph.opengl: Requires >= OpenGL 2.0 (not ES); Found %s" % verString
            )

        for item in self.items:
            if not item.isInitialized():
                item.initialize()

    def paintGL(self, region=None, viewport=None, useItemNames=False):
        """
        viewport specifies the arguments to glViewport. If None, then we use self.opts['viewport']
        region specifies the sub-region of self.opts['viewport'] that should be rendered.
        Note that we may use viewport != self.opts['viewport'] when exporting.
        """
        if viewport is None:
            glViewport(*self.getViewport())
        else:
            glViewport(*viewport)
        self.setProjection(region=region)
        self.setModelview()
        glClearColor(*BG_COLOR)
        glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )
        self.drawItemTree(useItemNames=useItemNames)

    def addItem(self, item):
        self.items.append(item)

        if self.isValid():
            item.initialize()

        item._setView(self)
        self.update()

    def drawItemTree(self, item=None, useItemNames=False):
        if item is None:
            items = [x for x in self.items if x.parentItem() is None]
        else:
            items = item.childItems()
            items.append(item)
        for i in items:
            if i is item:
                try:
                    glPushAttrib(GL_ALL_ATTRIB_BITS)
                    if useItemNames:
                        glLoadName(i._id)
                        self._itemNames[i._id] = i
                    i.paint()
                except:
                    from .. import debug
                    debug.printExc()
                    print("Error while drawing item %s." % str(item))

                finally:
                    glPopAttrib()
            else:
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                try:
                    tr = i.transform()
                    glMultMatrixf(np.array(tr.data(), dtype=np.float32))
                    self.drawItemTree(i, useItemNames=useItemNames)
                finally:
                    glMatrixMode(GL_MODELVIEW)
                    glPopMatrix()

    def renderToArray(self, size, format=GL_BGRA, type=GL_UNSIGNED_BYTE, textureSize=1024,
                      padding=256):
        w, h = map(int, size)

        self.makeCurrent()
        tex = None
        fb = None
        depth_buf = None
        try:
            output = np.empty((h, w, 4), dtype=np.ubyte)
            fb = glfbo.glGenFramebuffers(1)
            glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, fb)

            glEnable(GL_TEXTURE_2D)
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            texwidth = textureSize
            data = np.zeros((texwidth, texwidth, 4), dtype=np.ubyte)

            ## Test texture dimensions first
            glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA, texwidth, texwidth, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, None)
            if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
                raise RuntimeError(
                    "OpenGL failed to create 2D texture (%dx%d); too large for this hardware." % data.shape[
                                                                                                 :2])
            ## create texture
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texwidth, texwidth, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, data)

            # Create depth buffer
            depth_buf = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, depth_buf)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, texwidth, texwidth)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
                                      depth_buf)

            self.opts['viewport'] = (
            0, 0, w, h)  # viewport is the complete image; this ensures that paintGL(region=...)
            # is interpreted correctly.
            p2 = 2 * padding
            for x in range(-padding, w - padding, texwidth - p2):
                for y in range(-padding, h - padding, texwidth - p2):
                    x2 = min(x + texwidth, w + padding)
                    y2 = min(y + texwidth, h + padding)
                    w2 = x2 - x
                    h2 = y2 - y

                    ## render to texture
                    glfbo.glFramebufferTexture2D(glfbo.GL_FRAMEBUFFER, glfbo.GL_COLOR_ATTACHMENT0,
                                                 GL_TEXTURE_2D, tex, 0)

                    self.paintGL(region=(x, h - y - h2, w2, h2),
                                 viewport=(0, 0, w2, h2))  # only render sub-region
                    glBindTexture(GL_TEXTURE_2D, tex)  # fixes issue #366

                    ## read texture back to array
                    data = glGetTexImage(GL_TEXTURE_2D, 0, format, type)
                    data = np.frombuffer(data, dtype=np.ubyte).reshape(texwidth, texwidth, 4)[::-1,
                           ...]
                    output[y + padding:y2 - padding, x + padding:x2 - padding] = data[-(
                                h2 - padding):-padding, padding:w2 - padding]

        finally:
            self.opts['viewport'] = None
            glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
            if tex is not None:
                glDeleteTextures([tex])
            if fb is not None:
                glfbo.glDeleteFramebuffers([fb])
            if depth_buf is not None:
                glDeleteRenderbuffers(1, [depth_buf])

        return output