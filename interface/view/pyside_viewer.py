from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QVector3D, QMouseEvent
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
import numpy as np


class PointCloudViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._points = np.random.rand(1000, 3)  # Generate some random points for testing
        self._lines = []  # Store the lines drawn by the user
        self._last_click = None  # Store the last click position

    def initializeGL(self):
        self.gl = self.context().versionFunctions()  # Get the OpenGL version functions

        # Create the vertex shader
        vertex_shader = QOpenGLShader(QOpenGLShader.Vertex)
        vertex_shader.compileSourceCode("""
            attribute highp vec4 position;
            uniform highp mat4 modelview_projection;
            void main() {
                gl_Position = modelview_projection * position;
            }
        """)

        # Create the fragment shader
        fragment_shader = QOpenGLShader(QOpenGLShader.Fragment)
        fragment_shader.compileSourceCode("""
            uniform lowp vec4 color;
            void main() {
                gl_FragColor = color;
            }
        """)

        # Create the shader program
        self.shader_program = QOpenGLShaderProgram()
        self.shader_program.addShader(vertex_shader)
        self.shader_program.addShader(fragment_shader)
        self.shader_program.link()
        self.shader_program.bind()

        # Set up the vertex buffer
        self.vertex_buffer = self.gl.glGenBuffers(1)
        self.gl.glBindBuffer(self.gl.GL_ARRAY_BUFFER, self.vertex_buffer)
        self.gl.glBufferData(self.gl.GL_ARRAY_BUFFER, self._points.nbytes, self._points, self.gl.GL_STATIC_DRAW)

        # Enable the vertex attribute array
        position_loc = self.shader_program.attributeLocation('position')
        self.gl.glEnableVertexAttribArray(position_loc)
        self.gl.glVertexAttribPointer(position_loc, 3, self.gl.GL_FLOAT, False, 0, None)

        # Set the clear color to black
        self.gl.glClearColor(0, 0, 0, 1)

        # Enable depth testing
        self.gl.glEnable(self.gl.GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        self.gl.glViewport(0, 0, width, height)

        # Set up the projection matrix
        self.projection_matrix = self.computeProjectionMatrix(width, height)

    def paintGL(self):
        # Clear the color and depth buffers
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)

        # Set up the modelview matrix
        self.modelview_matrix = self.computeModelViewMatrix()

        # Set the uniform values in the shader program
        modelview_projection_loc = self.shader_program.uniformLocation('modelview_projection')
        self.gl.glUniformMatrix4fv(modelview_projection_loc, 1, False, np.matmul(self.projection_matrix, self.modelview_matrix).tolist())

        # Draw the points
        self.shader_program.setUniformValue('color', Qt.white)
        self.gl.glDrawArrays(self.gl.GL_POINTS, 0, len(self._points))

        # Draw the lines
        self.shader_program.setUniformValue('color', Qt.red)
        self.gl.glLineWidth(2.0)
        self.gl.glBegin(self.gl.GL_LINES)
        for p1, p2 in self._lines:
            self.gl.glVertex3f(*p1)
            self.gl.glVertex3f(*p2)
        self.gl.glEnd()


if __name__ == '__main__':
    app = QApplication([])

    viewer = PointCloudViewer()
    viewer.show()

    app.exec_()
