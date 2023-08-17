import sys, os
from glob import glob

import numpy as np
from scipy.spatial.transform import Rotation as R

from cosense3d.dataset.toolkit.cosense import OBJ_ID_MAP
from cosense3d.utils.misc import list_dirs, load_yaml, save_json
from cosense3d.utils.vislib import o3d_draw_frame_data, plt_draw_frame_data


def convert_meta(root_dir, meta_dir):
    os.makedirs(meta_dir, exist_ok=True)
    # load all paths of different scenarios
    scenarios = list_dirs(root_dir)[21:]

    # loop over all scenarios
    print("Loading meta data...")
    for (i, scenario) in enumerate(scenarios):
        print(f"scenario: {scenario}")
        scenario_folder = os.path.join(root_dir, scenario)
        scenario_meta_file = os.path.join(meta_dir, scenario[20:] + '.yaml')
        scenario_dict = {}
        # read meta for current scenario
        agents = list_dirs(scenario_folder)
        for agent in agents:
            yamls = sorted(glob(os.path.join(scenario_folder, agent, "*.yaml")))
            for y in yamls[:10]:
                meta = load_yaml(y)
                frame = os.path.basename(y)[:-5]
                pose = meta.get('gps', None)
                if pose is not None:
                    pose = [float(p) for p in pose]
                transf_matrix = meta['lidar_pose']
                # transf_matrix = np.linalg.inv(transf_matrix)
                lidar_pose = transformation_mat2pose(transf_matrix)
                objects = meta['vehicles']
                objects = objects_to_cosense(objects, transf_matrix)
                frame_dict = {
                    agent: {
                        'pose': pose,
                        'time': None,  # timestamp for the current vehicle pose
                        'lidar0': {
                            0: {
                                'pose': lidar_pose,
                                'time': None,  # timestamp for the current lidar0 triggering round
                                'filename': y.replace('yaml', 'pcd').replace(root_dir, ''),
                            }
                        },
                        'camera': {},  # TODO API for cameras
                        'objects': objects
                    }
                }
                if frame in scenario_dict:
                    scenario_dict[frame].update(frame_dict)
                else:
                    scenario_dict[frame] = frame_dict
        # save_json(scenario_dict, scenario_meta_file)

        for frame, frame_dict in scenario_dict.items():
            o3d_draw_frame_data(frame_dict, root_dir)
            plt_draw_frame_data(frame_dict, root_dir)


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

    # object pose is list while lidar0 pose is transformation matrix
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        world_to_x2 = x2
        transformation_matrix = np.dot(world_to_x2, x1_to_world)
    # both are numpy matrix
    else:
        world_to_x2 = np.linalg.inv(x2)
        transformation_matrix = np.dot(world_to_x2, x1)

    return transformation_matrix


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
        sys.exit('Unknown order')


def project_world_objects(transformation_matrix):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    transformation_matrix : np.ndarray
        From current object to ego.

    Returns
    ----------
    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).
    """
    output_dict = {}
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        ass_id = object_content.get('ass_id', object_id)

        obj_type = {
            'car': 'vehicle.car',
            'van': 'vehicle.van',
            'truck': 'vehicle.truck',
            'bus': 'vehicle.bus',
            'concretetruck': 'vehicle.truck',
            'bicyclerider': 'vehicle.cyclist',
            'scooter': 'vehicle.scooter',
            'scooterrider': 'vehicle.scooter',
            'pedestrian': 'human.pedestrian'
        }[object_content.get('obj_type', "car").lower()]

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, transformation_matrix)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order='lwh')

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: {'coord': bbx_lidar,
                                            'ass_id': ass_id,
                                            'object_type': obj_type}})
    return output_dict


def project_object_to_world(object_dict, transformation_matrix):
    output_dict = {}
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        ass_id = object_content.get('ass_id', object_id)

        obj_type = {
            'car': 'vehicle.car',
            'van': 'vehicle.van',
            'truck': 'vehicle.truck',
            'bus': 'vehicle.bus',
            'concretetruck': 'vehicle.truck',
            'bicyclerider': 'vehicle.cyclist',
            'scooter': 'vehicle.scooter',
            'scooterrider': 'vehicle.scooter',
            'pedestrian': 'human.pedestrian'
        }[object_content.get('obj_type', "car").lower()]

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x_to_world(object_pose)
        object2world = transformation_matrix @ object2lidar

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = (np.dot(object2world, bbx)).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order='lwh')

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: {'coord': bbx_lidar,
                                            'ass_id': ass_id,
                                            'object_type': obj_type}})
    return output_dict


def objects_to_cosense(objects, transf_matrix):
    output_dict = project_object_to_world(objects, transf_matrix)
    objects_list = []
    for i, content in output_dict.items():
        objects_list.append([
            i, OBJ_ID_MAP[content['object_type'].lower()]]
                            + content['coord'].squeeze().tolist())
    return objects_list


def transformation_mat2pose(matrix):
    r = R.from_matrix(matrix[:3, :3]).as_euler('xyz')
    t = matrix[:3, 3]
    return t.tolist() + r.tolist()


if __name__=="__main__":
    convert_meta(
        "/data/v2vreal/train",
        "dataset/metas/v2vreal"
    )