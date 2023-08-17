import copy
import math
import os
from glob import glob
import numpy as np
import tqdm
import open3d as o3d
import os.path as osp

import cv2
from collections import OrderedDict
from torch.utils.data import Dataset


from scipy.spatial.transform import Rotation as R
from cosense3d.utils.misc import load_yaml, save_json
from cosense3d.dataset.toolkit import register_pcds
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system
    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]
    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list or np.ndarray
        The pose of x1 under world coordinates or
        transformation matrix x1->world
    x2 : list or np.ndarray
        The pose of x2 under world coordinates or
         transformation matrix x2->world

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    if isinstance(x1, list) and isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        x2_to_world = x_to_world(x2)
        world_to_x2 = np.linalg.inv(x2_to_world)
        transformation_matrix = np.dot(world_to_x2, x1_to_world)

    # object pose is list while lidar pose is transformation matrix
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        world_to_x2 = x2
        transformation_matrix = np.dot(world_to_x2, x1_to_world)
    # both are numpy matrix
    else:
        world_to_x2 = np.linalg.inv(x2)
        transformation_matrix = np.dot(world_to_x2, x1)

    return transformation_matrix


def create_bbx(extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        Width, height, length of the bbx.

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """

    bbx = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])

    return bbx


def corner_to_center(corner3d, order='lwh'):
    """
    Convert 8 corners to x, y, z, dx, dy, dz, yaw.

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)

    order : str
        'lwh' or 'hwl'

    Returns
    -------
    box3d : np.ndarray
        (N, 7)
    """
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)
    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1,
                    keepdims=True))
    l = (np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    w = (np.sqrt(
        np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1,
               keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    theta = (np.arctan2(corner3d[:, 1, 1] - corner3d[:, 2, 1],
                        corner3d[:, 1, 0] - corner3d[:, 2, 0]) +
             np.arctan2(corner3d[:, 0, 1] - corner3d[:, 3, 1],
                        corner3d[:, 0, 0] - corner3d[:, 3, 0]) +
             np.arctan2(corner3d[:, 5, 1] - corner3d[:, 6, 1],
                        corner3d[:, 5, 0] - corner3d[:, 6, 0]) +
             np.arctan2(corner3d[:, 4, 1] - corner3d[:, 7, 1],
                        corner3d[:, 4, 0] - corner3d[:, 7, 0]))[:,
            np.newaxis] / 4

    if order == 'lwh':
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(
            batch_size, 7)
    elif order == 'hwl':
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(
            batch_size, 7)
    else:
        raise NotImplementedError


def project_world_objects(object_dict,
                          output_dict,
                          lidar_pose,
                          order):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    order : str
        'lwh' or 'hwl'
    """
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        if 'ass_id' not in object_content:
            ass_id = object_id
        else:
            ass_id = object_content['ass_id']
        if 'obj_type' not in object_content:
            obj_type = 'Car'
        else:
            obj_type = object_content['obj_type']

        # todo: pedestrain is not consdered yet
        # todo: only single class now
        if obj_type == 'Pedestrian':
            continue

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to lidar coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: {'coord': bbx_lidar,
                                            'ass_id': ass_id}})


def opv2v_to_cosense(path_in, path_out, isSim=True, correct_transf=False):
    if isSim:
        order = 'lwh'
    else:
        order = 'hwl'
    for split in ['train', 'test']:
        scenarios = sorted(os.listdir(os.path.join(path_in, split)))
        with open(os.path.join(path_out, f'{split}.txt'), 'w') as fh:
            fh.write('\n'.join(scenarios))
        for s in scenarios:
            print(s)
            visualize = False
            sdict = {}
            spath = os.path.join(path_in, split, s)
            cavs = sorted([x for x in os.listdir(spath) if os.path.isdir(os.path.join(spath, x))])
            ego_lidar_pose = None
            ego_id = cavs[0]
            frames = sorted([x[:-5]
                            for x in os.listdir(os.path.join(spath, ego_id)) if
                            x.endswith('.yaml') and 'sparse_gt' not in x])
            for f in tqdm.tqdm(frames):
                fdict = cs.fdict_template()
                ego_lidar_pose = None
                object_id_stack = []
                object_stack = []
                for i, cav_id in enumerate(cavs):
                    yaml_file = os.path.join(spath, cav_id, f'{f}.yaml')
                    params = load_yaml(yaml_file)
                    if cav_id == ego_id:
                        ego_lidar_pose = params['lidar_pose']

                    transformation = x1_to_x2(params['lidar_pose'], ego_lidar_pose)
                    # correct transformation
                    if not isSim and correct_transf and cav_id != ego_id:
                        ego_lidar_file = os.path.join(path_in, split, s, ego_id, f'{f}.pcd')
                        cav_lidar_file = os.path.join(path_in, split, s, cav_id, f'{f}.pcd')
                        transformation = register_pcds(cav_lidar_file, ego_lidar_file, transformation, visualize)
                        visualize = False

                    rot = R.from_matrix(transformation[:3, :3]).as_euler('xyz', degrees=False)
                    tl = transformation[:3, 3]
                    cav_lidar_pose2ego = tl.tolist() + rot.tolist()
                    cs.update_agent(fdict, cav_id, 'cav')
                    cs.update_agent_lidar(fdict, cav_id, 0,
                                          lidar_pose=cav_lidar_pose2ego,
                                          lidar_file=os.path.join(s, cav_id, f'{f}.pcd'))

                    objects_dict = params['vehicles']
                    output_dict = {}
                    if isSim:
                        ref_pose = ego_lidar_pose
                    else:
                        ref_pose = transformation

                    if not isSim:
                        # add ground truth boxes at cav local coordinate
                        # only for v2vreal
                        project_world_objects(objects_dict,
                                              output_dict,
                                              [0] * 6,
                                              order)
                        boxes_local = []
                        for object_id, object_content in output_dict.items():
                            if object_content['ass_id'] != -1:
                                object_id = object_content['ass_id']
                            else:
                                object_id = object_id
                            if order == 'hwl':
                                object_bbx = object_content['coord'][:, [0, 1, 2, 5, 4, 3, 6]]
                            boxes_local.append(
                                [object_id, 0,] +
                                object_bbx[0, :6].tolist() +
                                [0, 0, object_bbx[0, 6]]
                            )
                        cs.update_agent_gt_boxes(fdict, cav_id, boxes_local)

                    project_world_objects(objects_dict,
                                          output_dict,
                                          ref_pose,
                                          order)

                    for object_id, object_content in output_dict.items():
                        if object_content['ass_id'] != -1:
                            object_id_stack.append(object_content['ass_id'])
                        else:
                            object_id_stack.append(object_id + 100 * int(cav_id))
                        object_stack.append(object_content['coord'])

                # exclude all repetitive objects
                unique_indices = \
                    [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]
                if order == 'hwl':
                    object_stack = object_stack[:, [0, 1, 2, 5, 4, 3, 6]]

                cosense_bbx_center = np.zeros((len(object_stack), 11))
                cosense_bbx_center[:, 0] = np.array(unique_indices)
                cosense_bbx_center[:, 2:8] = object_stack[:, :6]
                cosense_bbx_center[:, 10] = object_stack[:, 6]
                cs.update_frame_bbx(fdict, cosense_bbx_center.tolist())
                fdict['agents'].pop(0)  # remove template agent
                sdict[f] = fdict

                if isinstance(ego_lidar_pose, np.ndarray):
                    rot = R.from_matrix(ego_lidar_pose[:3, :3]).as_euler('xyz', degrees=False)
                    tl = ego_lidar_pose[:3, 3]
                    ego_lidar_pose = tl.tolist() + rot.tolist()
                fdict['meta']['ego_id'] = ego_id
                fdict['meta']['ego_lidar_pose'] = ego_lidar_pose
            save_json(sdict, os.path.join(path_out, f'{s}.json'))


if __name__=="__main__":
    opv2v_to_cosense(
        "/koko/v2vreal",
        "/koko/cosense3d/v2vreal",
        isSim=False
    )
