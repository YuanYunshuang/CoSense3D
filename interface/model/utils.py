import numpy as np
from cosense3d.utils import pclib


def load_pcd(filename: str):
    lidar_dict = pclib.load_pcd(filename)
    return lidar_dict['xyz']


def line_parames_from_points2d(point1, point2):
    homo_params = np.array([[*point1[:2], 1.0], [*point2[:2], 1.0]])
    homo_params = np.cross(homo_params[0], homo_params[1])
    homo_params_norm = homo_params / (np.linalg.norm(homo_params[:-1]) + 1e-6)
    return homo_params_norm


def line_parames_from_points3d(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    e = point2 - point1
    r = point1 / e.sum()
    e = e / e.sum()
    return e, r


def foot_point2line(point, line):
    a = line[0]
    b = line[1]
    c = line[2]

    px = point[0]
    py = point[1]
    res = a * px + b * py + c
    s_ab = a ** 2 + b ** 2
    s_ab = 1
    fx = -a * res / s_ab + px
    fy = -b * res / s_ab + py

    return np.array([fx, fy])


def line_intersection(line1, line2):
    x = (line2[2] * line1[1] - line1[2] * line2[1]) / \
        (line1[0] * line2[1] - line2[0] * line1[1])
    y = -line1[0] / line1[1] * x - line1[2] / line1[1]
    return x, y


def line_distance(line1, line2):
    e1, e2 = line1[:-1], line2[:-1]
    r1 = np.array([0, 0, -line1[-1] / line1[-2]])
    r2 = np.array([0, 0, -line2[-1] / line2[-2]])
    n = np.cross(e1, e2)
    d = n * (r1 - r2) / np.linalg.norm(n)
    return d


def dist_points_to_line3d(points, line):
    points = np.array(points)
    line = np.array(line).squeeze()
    assert len(line.shape) == 2
    A, B, C = points, line[0], line[1]
    BA = A - B
    BC = C - B
    dists = np.linalg.norm(np.cross(BA, BC), axis=1) / np.linalg.norm(BC)
    return dists


def dist_points_to_line2d(points, line):
    points = np.array(points)
    line = np.array(line).squeeze()
    assert points.shape[1] == (len(line) - 1)
    dists = points @ line[:-1] + line[-1]
    return dists
