import random

import torch
import numpy as np

from cosense3d.model.pre_process import PreProcessorBase
from cosense3d.utils import pclib, box_utils


class GeoAugmentation(PreProcessorBase):
    """
    This module generates an overall rotation matrix from the sub
    rotation matrices defined by the random rotation around xyz axis,
    random flip along x, y or xy axis, and the random scaling.
    In this module, the out point cloud will be rotated to the global
    augmented coordinate system. However the translation will be unchanged.
    We call this state of point cloud as PC_rot_aug. Once the translation
    is performed, we get PC_aug.
    With the new lidar0 pose (translation from rot_aug to aug, rotation angles
    are all set to zeros), the PC_rot_aug can be translated to PC_aug.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.operations = [
            'rotate',
            'flip',
            'scale',
        ]

    def __call__(self, data_dict):
        # get overall tf from rotate, flip and scale
        aug_tf = np.eye(4)
        for op in self.operations:
            if hasattr(self, op):
                aug_tf = getattr(self, f'fn_{op}')(aug_tf)
        aug_tf = torch.from_numpy(aug_tf).float().to(data_dict['objects'].device)

        # process lidar data
        if data_dict['pcds'] is not None:
            pcds = data_dict['pcds']

            # rotate pcds to global rotation-coords
            if data_dict['projected'] or data_dict['tf_cav2ego'] is None:
                pcds_global = pcds
            else:
                tfs = data_dict['tf_cav2ego']
                pcds_global = []
                for i in sorted(torch.unique(pcds[:, 0])):
                    tf = tfs[int(i)]
                    pcd = pcds[pcds[:, 0] == i]
                    if len(tf) == 4:
                        pcd[:, 1:4] = (tf[:3, :3] @ pcd[:, 1:4].T).T
                        pcd[:, 1:4] = pcd[:, 1:4] + tf[:3, 3].reshape(1, 3)
                    else:
                        pcd[:, 1:4] = pclib.rotate_points_batch(
                            pcd[:, 1:4].reshape(1, -1, 3),
                            tf[3:]
                        )
                        pcd[:, 1:4] = pcd[:, 1:4] + tf[:3].reshape(1, 3)
                    pcds_global.append(pcd)
                pcds_global = torch.cat(pcds_global, dim=0)
                data_dict['projected'] = True

            # rotate pcds to augmented coords
            pcds_global[:, 1:4] = (aug_tf[:3, :3] @ pcds_global[:, 1:4].T).T
            # lidar_poses[:, :3] = (aug_tf[:3, :3] @ lidar_poses[:, :3].T).T
            # lidar_poses[:, 3:] = 0

            data_dict['pcds'] = pcds_global

        # process object data
        boxes = data_dict['objects']  # in global coords
        boxes_corner = box_utils.boxes_to_corners_3d(boxes[:, 3:])  # (N, 8, 3)
        # rotate bbx to augmented coords
        boxes_corner = (aug_tf[:3, :3] @ boxes_corner.reshape(-1, 3).T
                        ).T.reshape(len(boxes_corner), 8, 3)
        boxes_center = box_utils.corners_to_boxes_3d(boxes_corner)
        boxes[:, 3:] = boxes_center
        data_dict['objects'] = boxes

        # update camera extrinsics
        if data_dict.get('cam_params', None) is not None:
            for i, param in enumerate(data_dict['cam_params']):
                extrinsic = param['extrinsic']
                data_dict['cam_params'][i]['extrinsic'] = \
                    (np.array(extrinsic) @ np.linalg.inv(aug_tf)).tolist()

    def fn_rotate(self, tf):
        # param: [roll, pitch, yaw] in degree
        angles = []
        for angle in self.rotate:
            angle = angle / 180 * np.pi * random.random()
            angles.append(angle)
        angles = np.array(angles)
        rot = pclib.rotation_matrix(angles)
        tf[:3, :3] = rot @ tf[:3, :3]
        return tf

    def fn_flip(self, tf):
        rot = np.eye(3)
        flip = np.random.choice(4, 1)
        # flip =1 : flip x
        # flip =2 : flip y
        # flip =3 : flip x & y

        # flip x
        if 'x' in getattr(self, 'flip', 'xy') and (flip == 1 or flip == 3):
            rot[0, 0] *= -1
        # flip y
        if 'y' in getattr(self, 'flip', 'xy') and (flip == 2 or flip == 3):
            rot[1, 1] *= -1
        tf[:3, :3] = rot @ tf[:3, :3]
        return tf

    def fn_scale(self, tf):
        scale = np.eye(3)
        scale_ratio = np.random.uniform(1.0 - self.scale, 1.0 + self.scale, (1, 3))
        scale[[0, 1, 2], [0, 1, 2]] = scale_ratio
        tf[:3, :3] = scale @ tf[:3, :3]
        return tf
