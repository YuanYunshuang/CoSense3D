import copy
from typing import TYPE_CHECKING, List, Tuple, Union
import numpy as np
import OpenGL.GL as GL

Color4f = Tuple[float, float, float, float]  # type alias for type hinting
PointList = List[List[float]]

class Transform3D:
    def __init__(self):
        self._matrix4x4 = np.eye(4)


def clip_array(arr, vmin, vmax, out=None):
    # replacement for np.clip due to regression in
    # performance since numpy 1.17
    # https://github.com/numpy/numpy/issues/14281

    if vmin is None and vmax is None:
        # let np.clip handle the error
        return np.clip(arr, vmin, vmax, out=out)
    result = copy.deepcopy(arr)
    if vmin is not None:
        result = np.core.umath.maximum(result, vmin, out=out)
    if vmax is not None:
        result = np.core.umath.minimum(result, vmax, out=out)
    return result


# Creates a circular mask with radius around center
def circular_mask(arr_length, center, radius):
    dx = np.arange(arr_length)
    dx2 = (dx[np.newaxis, :] - center) ** 2 + \
          (dx[:, np.newaxis] - center) ** 2
    return dx2 < radius ** 2


# Returns the minimum (closest) depth for a specified radius around the center
def depth_min(depths, center, r=10) -> float:
    selected_depths = depths[circular_mask(len(depths), center, r)]
    filtered_depths = selected_depths[(0 < selected_depths) & (selected_depths < 1)]
    if 0 in depths:  # Check if cursor is at widget border
        return 1
    if len(filtered_depths) > 0:
        return np.min(filtered_depths)
    else:
        return 1


# Returns the mean depth for a specified radius around the center
def depth_smoothing(depths, center, r=15) -> float:
    selected_depths = depths[circular_mask(len(depths), center, r)]
    if 0 in depths:  # Check if cursor is at widget border
        return 1
    elif np.isnan(
        selected_depths[selected_depths < 1]
    ).all():  # prevent mean of empty slice
        return 1
    return float(np.nanmedian(selected_depths[selected_depths < 1]))

def draw_points(
    points: PointList, color: Color4f = (0, 1, 1, 1), point_size: int = 30
) -> None:
    GL.glColor4d(*color)
    GL.glPointSize(point_size)
    GL.glBegin(GL.GL_POINTS)
    for point in points:
        GL.glVertex3d(*point)
        print("draw point:", point)
    GL.glEnd()


def draw_lines(
    points: PointList, color: Color4f = (0, 1, 1, 1), line_width: int = 2
) -> None:
    GL.glColor4d(*color)
    GL.glLineWidth(line_width)
    GL.glBegin(GL.GL_LINES)
    for point in points:
        GL.glVertex3d(*point)
    GL.glEnd()


def gl_depth_deactive(func):
    def wrapper(*args):
        # GL.glDepthFunc(GL.GL_ALWAYS)
        GL.glDepthMask(GL.GL_FALSE)
        func(*args)
        GL.glDepthMask(GL.GL_TRUE)
        # GL.glDepthFunc(GL.GL_LESS)
    return wrapper


