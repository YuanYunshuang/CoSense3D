from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram, \
    QColor, QOpenGLBuffer, QMatrix4x4
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


class RectangleItem(GLGraphicsItem):
    def __init__(self, color=(1.0, 0.0, 0.0, 1.0)):
        super().__init__()
        self.pos = [0, 0]
        self.size = [5, 10]
        self.color = color

    def paint(self):
        if self.pos is not None and self.size is not None:
            glColor4f(*self.color)
            glBegin(GL_QUADS)
            glVertex3f(self.pos[0], self.pos[1], 0)
            glVertex3f(self.pos[0] + self.size[0], self.pos[1], 0)
            glVertex3f(self.pos[0] + self.size[0], self.pos[1] + self.size[1], 0)
            glVertex3f(self.pos[0], self.pos[1] + self.size[1], 0)
            glEnd()

    def setRectangle(self, p1, p2):
        self.pos = [p1.x(), p1.y()]
        self.pos = [1, 1]
        self.size = [3, 3]

    def removeRectangle(self):
        self.pos = None
        self.size = None



class Rectangle:
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

    def initialize(self):
        self.program = QOpenGLShaderProgram()
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, """
            attribute highp vec4 posAttr;
            uniform highp mat4 matrix;
            void main() {
                gl_Position = matrix * posAttr;
            }
        """)
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, """
            uniform highp vec4 color;
            void main() {
                gl_FragColor = color;
            }
        """)

        if not self.program.link():
            print("Shader program failed to link:", self.program.log())
            return

    def get_rect_vertices(self, rectangle):
        return [
                rectangle.topLeft().x(), rectangle.topLeft().y(),
                rectangle.topRight().x(), rectangle.topRight().y(),
                rectangle.bottomRight().x(), rectangle.bottomRight().y(),
                rectangle.bottomLeft().x(), rectangle.bottomLeft().y()]

    def paint(self, rectangle):
        try:
            vertices = self.get_rect_vertices(rectangle)
            self.program.bind()

            vertex_buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            vertex_buffer.create()
            vertex_buffer.bind()
            vertex_buffer.setUsagePattern(QOpenGLBuffer.StaticDraw)
            vertex_buffer.allocate(len(vertices) * 4)
            data = np.array(vertices).astype(np.float32)
            vertex_buffer.write(0, data, len(vertices) * 4)

            self.program.enableAttributeArray("posAttr")
            self.program.setAttributeBuffer("posAttr", GL_FLOAT, 0, 2)

            matrix = QMatrix4x4()
            matrix.ortho(0, self.parent.width(), self.parent.height(), 0, -1, 1)

            self.program.setUniformValue("matrix", matrix)
            self.program.setUniformValue("color", QColor(255, 255, 0, 60))

            glDrawArrays(GL_QUADS, 0, 4)
        finally:
            vertex_buffer.destroy()
            self.program.release()