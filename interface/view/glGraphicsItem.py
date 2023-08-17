from OpenGL.GL import *  # noqa
from OpenGL import GL
from PyQt5 import QtCore
import numpy as np
from pyqtgraph.Transform3D import Transform3D

from .utils import clip_array
from . import shaders


GLOptions = {
    'opaque': {
        GL_DEPTH_TEST: True,
        GL_BLEND: False,
        GL_ALPHA_TEST: False,
        GL_CULL_FACE: False,
    },
    'translucent': {
        GL_DEPTH_TEST: True,
        GL_BLEND: True,
        GL_ALPHA_TEST: False,
        GL_CULL_FACE: False,
        'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
    },
    'additive': {
        GL_DEPTH_TEST: False,
        GL_BLEND: True,
        GL_ALPHA_TEST: False,
        GL_CULL_FACE: False,
        'glBlendFunc': (GL_SRC_ALPHA, GL_ONE),
    },
}


class GLGraphicsItem(QtCore.QObject):
    _nextId = 0

    def __init__(self, parentItem=None):
        super().__init__()
        self._id = GLGraphicsItem._nextId
        GLGraphicsItem._nextId += 1

        self.__parent = None
        self.__view = None
        self.__children = set()
        self.__transform = Transform3D()
        self.__visible = True
        self.__initialized = False
        self.setParentItem(parentItem)
        self.__glOpts = {}

    def view(self):
        return self.__view

    def _setView(self, v):
        self.__view = v

    def setParentItem(self, item):
        """Set this item's parent in the scenegraph hierarchy."""
        # # remove self from the old parent, only possible to be called after init
        # if self.__parent is not None:
        #     self.__parent.__children.remove(self)

        # if the given parent item exist, add self as its child and set self's parent attr
        if item is not None:
            item.__children.add(self)
        self.__parent = item

        # ensure self view is consistant with the view of the new parent if given
        if self.__parent is not None and self.view() is not self.__parent.view():
            if self.view() is not None:
                self.view().removeItem(self)
            self.__parent.view().addItem(self)

    def setGLOptions(self, opts):
        if isinstance(opts, str):
            opts = GLOptions[opts]
        self.__glOpts = opts.copy()
        self.update()

    def updateGLOptions(self, opts):
        self.__glOpts.update(opts)

    def initialize(self):
        self.initializeGL()
        self.__initialized = True

    def isInitialized(self):
        return self.__initialized

    def initializeGL(self):
        """
        Called after an item is added to a GLViewWidget.
        The widget's GL context is made current before this method is called.
        (So this would be an appropriate time to generate lists, upload textures, etc.)

        This function is defined in the child class.
        """
        pass

    def setupGLState(self):
        """
        This method is responsible for preparing the GL state options needed to render
        this item (blending, depth testing, etc). The method is called immediately before painting the item.
        """
        for k,v in self.__glOpts.items():
            if v is None:
                continue
            if isinstance(k, str):
                func = getattr(GL, k)
                func(*v)
            else:
                if v is True:
                    glEnable(k)
                else:
                    glDisable(k)

    def parentItem(self):
        """Return a this item's parent in the scenegraph hierarchy."""
        return self.__parent

    def childItems(self):
        """Return a list of this item's children in the scenegraph hierarchy."""
        return list(self.__children)

    def paint(self):
        """
        Called by the GLViewWidget to draw this item.
        It is the responsibility of the item to set up its own modelview matrix,
        but the caller will take care of pushing/popping.

        This function should be overridden by child class if something needs to be drawn
        """
        self.setupGLState()

    def update(self):
        """
        Indicates that this item needs to be redrawn, and schedules an update
        with the view it is displayed in.
        """
        v = self.view()
        if v is None:
            return
        v.update()

    def viewTransform(self):
        """Return the transform mapping this item's local coordinate system to the
        view coordinate system."""
        tr = self.__transform
        p = self
        while True:
            p = p.parentItem()
            if p is None:
                break
            tr = p.transform() * tr
        return Transform3D(tr)

    def transform(self):
        """Return this item's transform object."""
        return self.__transform

    def mapToView(self, point):
        tr = self.viewTransform()
        if tr is None:
            return point
        return tr.map(point)


class ScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""
    def __init__(self, **kwds):
        super().__init__()
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.pos = None
        self.size = 10
        self.color = [1.0, 1.0, 1.0, 0.5]
        self.pxMode = True
        self.setData(**kwds)
        self.shader = None

    def setData(self, **kwds):
        args = ['pos', 'color', 'size', 'pxMode']
        for k in kwds.keys():
            if k not in args:
                raise Exception(
                    'Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))

        # save data in C-style continuous memory block
        for k in args:
            if k in kwds:
                v = kwds.pop(k)
                if isinstance(v, np.ndarray):
                    setattr(self, k, np.ascontiguousarray(v, dtype=np.float32))
                elif isinstance(v, int) or isinstance(v, float):
                    setattr(self, k, v)
                else:
                    raise NotImplementedError

        self.pxMode = kwds.get('pxMode', self.pxMode)
        self.update()

    def initializeGL(self):
        if self.shader is not None:
            return

        ## Generate texture for rendering points
        w = 64
        def genTexture(x,y):
            """
            Generates a grayscale circular texture with a "hill" in the center,
            x, y, is the possible index combination of an array
            """
            # get the distances to the texture center
            r = np.hypot((x-(w-1)/2.), (y-(w-1)/2.))
            # generate gray scale texture for each point
            texture = 255 * (w / 2 - clip_array(r, w / 2 - 1, w / 2))
            return texture
        pData = np.empty((w, w, 4))
        pData[:] = 255
        # set alpha
        pData[:, :, 3] = np.fromfunction(genTexture, pData.shape[:2])
        pData = pData.astype(np.ubyte)

        if getattr(self, "pointTexture", None) is None:
            self.pointTexture = glGenTextures(1)  # texture id
        # activate texture unit 0
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_TEXTURE_2D)
        # bind a texture object to the current texture unit 0
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     pData.shape[0], pData.shape[1], 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, pData)

        # pointSprite shader for performing the transformation from a point to a quad
        self.shader = shaders.getShaderProgram('pointSprite')

    def paint(self):
        if self.pos is None:
            return

        self.setupGLState()

        glEnable(GL_POINT_SPRITE)

        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)

        # use vertex coor to sample point sprite texture
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        # set minification filter:
        # linear interpolation between texels when texture is rendered in smaller size
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # set magnification filter:
        # linear interpolation between texels when texture is rendered in larger size
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # set warp mode to EDGE mode in S/horizontal direction
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        # set warp mode to EDGE mode in T/vertical direction
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glEnable(GL_PROGRAM_POINT_SIZE)

        with self.shader:
            glEnableClientState(GL_VERTEX_ARRAY)
            try:
                pos = self.pos
                glVertexPointerf(pos)

                # ------define point colors------
                if isinstance(self.color, np.ndarray):
                    # use custom colors for points
                    glEnableClientState(GL_COLOR_ARRAY)
                    glColorPointerf(self.color)
                else:
                    # use uniform color for all points
                    color = self.color
                    assert isinstance(color, tuple) or isinstance(color, list)
                    glColor4f(*color)

                # ------define point sizes------
                if not self.pxMode or isinstance(self.size, np.ndarray):
                    # use custom sizes
                    glEnableClientState(GL_NORMAL_ARRAY)
                    norm = np.zeros(pos.shape, dtype=np.float32)
                    if self.pxMode:
                        norm[..., 0] = self.size
                    else:  # size defined according to pos in global space
                        # get global pos in the viewer
                        gpos = self.mapToView(pos.transpose()).transpose()
                        if self.view():
                            pxSize = self.view().pixelSize(gpos)
                        else:
                            pxSize = self.parentItem().view().pixelSize(gpos)
                        norm[..., 0] = self.size / pxSize

                    glNormalPointerf(norm)
                else:
                    # use uniform sizes
                    # vertex shader uses norm.x to determine point size
                    glNormal3f(self.size, 0, 0)
                glDrawArrays(GL_POINTS, 0, pos.shape[0])
            finally:
                glDisableClientState(GL_NORMAL_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_COLOR_ARRAY)
                glDisable(GL_TEXTURE_2D)