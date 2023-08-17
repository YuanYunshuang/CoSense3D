import numpy as np
import json, codecs
from typing import List, Tuple, Any

from numpy import ndarray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


Point = Tuple[float, float, float]

label_ego = json.loads(codecs.open("./cloud_ego/1070_000317.json", 'r', encoding='utf-8').read())
label_coop = json.loads(codecs.open("./cloud_coop/1070_000317/000387.json", 'r', encoding='utf-8').read())


def label_to_lines(label: List[List[Point]]) -> np.ndarray:
    """
    Break chained lines into two-point defined lines
    :param label: List[List[(float, float, float)]]
    :return:
    """
    lines = []
    for l in label:
        last_point = l[0]
        for p in l[1:]:
            lines.append([last_point, p])
            last_point = p
    return np.array(lines)


def distance_2dlines_to_origin(lines):
    A = np.array([0.0, 0.0]).reshape(1, 2)
    B = lines[:, 0]
    C = lines[:, 1]
    BA = A - B
    BC = C - B
    dists = np.cross(BA, BC) / np.linalg.norm(BC, axis=1)
    return dists


def line_homo_params(lines):
    if len(lines.shape)==3:
        homo_params = np.concatenate([lines, np.ones_like(lines[:, :, :1])], axis=2)
        homo_params = np.cross(homo_params[:, 0], homo_params[:, 1])
        homo_params_norm = homo_params / (np.linalg.norm(homo_params[:, :2], axis=1,
                                                         keepdims=True) + 1e-6)
    else:
        homo_params = np.array([[*lines[0], 1.0], [*lines[1], 1.0]])
        homo_params = np.cross(homo_params[0], homo_params[1])
        homo_params_norm = homo_params / (np.linalg.norm(homo_params[:-1]) + 1e-6)
    return homo_params_norm


def lines_perpendiculars(lines):
    homo_params_norm = line_homo_params(lines)
    a = homo_params_norm[:, 0]
    b = homo_params_norm[:, 1]
    c = homo_params_norm[:, 2]
    fx = -a * c
    fy = -b * c

    return np.stack([fx, fy], axis=1)


def plot_lines_norms(norms, foots, lines):
    for n, f, l in zip(norms, foots, lines):
        plt.plot(l[:, 0], l[:, 1], 'k')
        # plt.plot(f[0], f[1], '*r')
        plt.arrow(f[0], f[1], n[0] * 5, n[1] * 5, color='r',
                  head_starts_at_zero=True, head_width=3)
    plt.show()
    plt.close()


def line_similarities(label: List[List[Point]], similarty_thr_deg: float = 45.0,)\
        -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    lines = label_to_lines(label)[:, :, :2]
    norms = lines[:, 1] - lines[:, 0]
    norms[:, 0] *= -1
    norms = norms[:, [1, 0]]
    tmp = np.linalg.norm(norms, axis=1, keepdims=True)
    norms = norms / tmp
    foots = lines_perpendiculars(lines)

    cos_similarity = norms @ norms.T / np.linalg.norm(norms, axis=1) ** 2
    # distances of lines and origin
    # dists = distance_2dlines_to_origin(lines)
    # dist_similarity = dists[:, None] - dists[None, :]
    # rs = lines[:, 0] / tmp # (14x2)
    # t = (rs[:, None, :] - rs[None, :, :]) / \
    #     (np.repeat(np.eye(len(norms))[:, :, None], 2, axis=2)
    #      + norms[None, :, :] - norms[:, None, :])
    # intersections = norms[:, None, :] * t + rs[:, None, :]

    # select line pairs that has intersection angle > 30
    # cos_similarity[np.triu_indices(len(cos_similarity))] = 1
    indices = np.where(
            np.abs(cos_similarity) < np.cos(np.deg2rad(similarty_thr_deg))
    )
    similarties = cos_similarity[indices[0], indices[1]]

    return lines, indices, norms, similarties


def cal_rotation(norm1, norm2):
    x1, y1 = norm1[0], norm1[1]
    x2, y2 = norm2[0], norm2[1]
    s_t = (x1 ** 2 - x2 ** 2) / (x1 * y2 - x2 * y1)
    c_t = (x1 - s_t * y2) / x2
    theta = (np.arctan2(s_t, c_t) + np.pi * 2) % np.pi
    angle = np.rad2deg(theta)
    return angle


def rotate(line, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    M = np.array([[ct, -st], [st, ct]])
    line_r = line @ M.T
    return line_r


def line_intersection(line1, line2):
    x = (line2[2] * line1[1] - line1[2] * line2[1]) / \
        (line1[0] * line2[1] - line2[0] * line1[1])
    y = -line1[0] / line1[1] * x - line1[2] / line1[1]
    return np.array([x, y])


def main():
    lines_ego, indices_ego, norms_ego, similarities_ego = line_similarities(label_ego)
    lines_coop, indices_coop, norms_coop, similarities_coop = line_similarities(label_coop)

    spatial_consistancy = np.abs(similarities_ego.reshape(-1, 1)
                                 - similarities_coop.reshape(1, -1))
    # consistancy_min_indices = np.argmin(spatial_consistancy, axis=1)
    matches = np.where(spatial_consistancy < 0.01)
    i = 0
    for e, c in zip(*matches):
        line_e1 = lines_ego[indices_ego[0][e]]
        line_e2 = lines_ego[indices_ego[1][e]]
        line_c1 = lines_coop[indices_coop[0][c]]
        line_c2 = lines_coop[indices_coop[1][c]]
        for l in lines_ego:
            plt.plot(l[:, 0], l[:, 1], 'g')
        for l in lines_coop:
            plt.plot(l[:, 0], l[:, 1], '--r')
        plt.plot(line_e1[:, 0], line_e1[:, 1], 'b')
        plt.plot(line_e2[:, 0], line_e2[:, 1], 'b')
        plt.plot(line_c1[:, 0], line_c1[:, 1], '--y')
        plt.plot(line_c2[:, 0], line_c2[:, 1], '--y')
        plt.savefig(f"/media/hdd/yuan/TMP/tmp{i}1.png")
        plt.close()
        theta1 = cal_rotation(norms_ego[indices_ego[0][e]],
                          norms_coop[indices_coop[0][c]])
        theta2 = cal_rotation(norms_ego[indices_ego[1][e]],
                          norms_coop[indices_coop[1][c]])
        theta = (theta1 + theta2) / 2
        # rotate points of lines
        line_c1_r = rotate(line_c1, theta)
        line_c2_r = rotate(line_c2, theta)
        # cal intersection using new lines
        intsec_e = line_intersection(line_homo_params(line_e1),
                                     line_homo_params(line_e2))
        intsec_c = line_intersection(line_homo_params(line_c1_r),
                                     line_homo_params(line_c2_r))
        # translation
        t = intsec_e - intsec_c

        # transform all coop lines
        lines_coop_r = rotate(lines_coop.reshape(-1, 2), theta)
        lines_coop_r = lines_coop_r + t.reshape(1, 2)
        lines_coop_r = lines_coop_r.reshape(-1, 2, 2)
        # plot
        for l in lines_ego:
            plt.plot(l[:, 0], l[:, 1], 'g')
        for l in lines_coop_r:
            plt.plot(l[:, 0], l[:, 1], '--r')
        plt.plot(intsec_e[0], intsec_e[1], '*k')
        plt.plot(intsec_c[0], intsec_c[1], '*k')
        # plt.show()
        plt.savefig(f"/media/hdd/yuan/TMP/tmp{i}2.png")
        i += 1
        plt.close()
        print(i)


if __name__=="__main__":
    lines_ego = label_to_lines(label_ego)[:, :, :2]
    lines_coop = label_to_lines(label_coop)[:, :, :2]
    homo_params_ego = line_homo_params(lines_ego)
    homo_params_coop = line_homo_params(lines_coop)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(homo_params_ego[:, 0], homo_params_ego[:, 1], '.', markersize=5)
    ax.plot(homo_params_coop[:, 0], homo_params_coop[:, 1], '.', markersize=5)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()
    plt.close()
