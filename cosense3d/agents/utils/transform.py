import torch
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from torch_scatter import scatter_mean

from cosense3d.utils import pclib, box_utils


def add_rotate(tf, rot):
    if isinstance(rot, list) and len(rot) == 3:
        # param: [roll, pitch, yaw] in radian
        rot = pclib.rotation_matrix(rot, degrees=False)
        rot = torch.from_numpy(rot).to(tf.device)
        tf[:3, :3] = rot @ tf[:3, :3]
    elif isinstance(rot, torch.Tensor):
        assert rot.shape[0] == 4
        tf = rot.to(tf.device) @ tf
    else:
        raise NotImplementedError
    return tf


def add_flip(tf, flip_idx, flip_axis='xy'):
    # flip_idx =1 : flip x
    # flip_idx =2 : flip y
    # flip_idx =3 : flip x & y
    rot = torch.eye(4).to(tf.device)
    # flip x
    if 'x' in flip_axis and (flip_idx == 1 or flip_idx == 3):
        rot[0, 0] *= -1
    # flip y
    if 'y' in flip_axis and (flip_idx == 2 or flip_idx == 3):
        rot[1, 1] *= -1
    tf = rot @ tf
    return tf


def add_scale(tf, scale_ratio):
    scale = torch.eye(4).to(tf.device)
    scale[[0, 1, 2], [0, 1, 2]] = scale_ratio
    tf = scale @ tf
    return tf


def apply_transform(data, transform, key):
    if (transform.cpu() == torch.eye(4)).all():
        return
    if key == 'points':
        C = data['points'].shape[-1]
        points = data['points'][:, :3]
        points = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1).T
        points = (transform @ points).T

        if C > 3:
            data['points'] = torch.cat([points[:, :3],
                                        data['points'][:, 3:]], dim=-1)
        else:
            data['points'] = points
    elif 'annos_global' == key or 'annos_local' == key:
        box_key = f"{key.split('_')[1]}_bboxes_3d"
        if box_key not in data:
            return
        boxes = data[box_key]
        boxes_corner = box_utils.boxes_to_corners_3d(boxes[:, :7])  # (N, 8, 3)
        boxes_corner = boxes_corner.reshape(-1, 3).T  # (N*8, 3)
        boxes_corner = torch.cat([boxes_corner, torch.ones_like(boxes_corner[:1])], dim=0)
        # rotate bbx to augmented coords
        boxes_corner = (transform @ boxes_corner)[:3].T.reshape(len(boxes), 8, 3)
        data[box_key][:, :7] = box_utils.corners_to_boxes_3d(boxes_corner, mode=7)
    elif key == 'img':
        for i in range(len(data['img'])):
            data['extrinsics'][i] = data['extrinsics'][i] @ transform.inverse()
            data['lidar2img'][i] = data['intrinsics'][i] @ data['extrinsics'][i]


def filter_range(data, lidar_range, key):
    if key == 'points':
        mask = filter_range_mask(data['points'], lidar_range)
        points = data['points'][mask]
        if len(points) == 0:
            # pad empty point cloud with random points to ensure batch norm validity
            points = data['points'].new_zeros((1024, points.shape[-1]))
            points[:, :2] = torch.rand_like(points[:, :2]) * 40
            points[:, 3] = -1
        data['points'] = points
    elif 'annos_global' == key or 'annos_local' == key:
        coor = key.split('_')[1]
        if f'{coor}_bboxes_3d' not in data:
            return
        mask = filter_range_mask(data[f'{coor}_bboxes_3d'][:, :3], lidar_range)
        data[f'{coor}_bboxes_3d'] = data[f'{coor}_bboxes_3d'][mask]
        data[f'{coor}_labels_3d'] = data[f'{coor}_labels_3d'][mask]
        data[f'{coor}_names'] = [data[f'{coor}_names'][i] for i, m in enumerate(mask) if m]


def filter_range_mask(points, lidar_range, eps=1e-4):
    lr = lidar_range.to(points.device)
    mask = (points[:, :3] > lr[:3].view(1, 3) + eps) & (points[:, :3] < lr[3:].view(1, 3) - eps)
    return mask.all(dim=-1)


class DataOnlineProcessor:

    @staticmethod
    def update_transform_with_aug(transform, aug_params):
        if 'rot' in aug_params:
            transform = add_rotate(transform, aug_params['rot'])
        if 'flip' in aug_params:
            transform = add_flip(transform, **aug_params['flip'])
        if 'scale' in aug_params:
            transform = add_scale(transform, aug_params['scale'])
        return transform

    @staticmethod
    def apply_transform(data, transform, apply_to=['points']):
        for k in apply_to:
            apply_transform(data, transform, k)

    @staticmethod
    def cav_aug_transform(data, transform, aug_params,
                          apply_to=['points', 'imgs', 'annos_global']):
        # augmentation
        if aug_params is not None:
            transform = DataOnlineProcessor.update_transform_with_aug(transform, aug_params)

        DataOnlineProcessor.apply_transform(data, transform, apply_to)

    @staticmethod
    def filter_range(data, lidar_range, apply_to):
        for k in apply_to:
            filter_range(data, lidar_range, k)

    @staticmethod
    @torch.no_grad()
    def free_space_augmentation(data, d=10, h=1.5, step=1.5):
        lidar = data['points']
        # get point lower than z_min=1.5m
        m = lidar[:, 2] < h
        points = lidar[m][:, :3]

        # generate free space points based on points
        dists = torch.norm(points[:, :2], dim=1).reshape(-1, 1)
        delta_d = torch.arange(1, d, step,
                               device=lidar.device).reshape(1, -1)
        steps = delta_d.shape[1]
        tmp = (dists - delta_d) / dists  # Nxsteps
        xyz_new = points[:, None, :] * tmp[:, :, None]  # Nxstepsx3

        # 1.remove free space points with negative distances to lidar center
        # 2.remove free space points higher than z_min
        # 3.remove duplicated points with resolution 1m
        xyz_new = xyz_new[tmp > 0]
        xyz_new = xyz_new[(xyz_new[..., 2] < h)]
        xyz_new = xyz_new[torch.randperm(len(xyz_new))]
        selected = torch.unique(torch.floor(xyz_new / 2).long(), return_inverse=True, dim=0)[1]
        xyz_new = scatter_mean(src=xyz_new, index=selected, dim=0)

        # pad free space point intensity as -1
        xyz_new = torch.cat([xyz_new, - torch.ones_like(xyz_new[:, :1])], dim=-1)
        data['points'] = torch.cat([lidar, xyz_new], dim=0)

    @staticmethod
    @torch.no_grad()
    def adaptive_free_space_augmentation(data, min_h=-1.5, steps=20, alpha=0.05, time_idx=None):
        """
        Add free space points according to the distance of points to the origin.

        lidar origin ->  *
                      *  *
                   *     * h
                *  ele   *
              ************
                     d

        Assume the theta = pi / 2 - ele (elevation angle),
        alpha = average angle between two lidar rings,
        d_k is the ground distance of the n_th lidar ring to lidar origin, k=1,...,n,
        delta_d is the distance between two neighboring lidar rings,
        then
        d = h*tan(theta)
        delta_d = d_n - d_{n-1} = dn - h*tan(arctan(h / d_n) - alpha)
        we sample free space points in the ground distance of delta_d relative to each ring
        with the given 'step' distance.

        Parameters
        ----------
        data: input data dict containing 'points'
        min_h: mininum sample height relative to lidar origin
        steps: number of points to be sampled for each lidar ray
        alpha: average angle offset between two neighboring lidar casting rays
        time_idx: if provided, time will be copied from the original points to free space points

        Returns
        -------

        """
        lidar = data['points']
        # get point lower than z_min=1.5m
        m = lidar[:, 2] < min_h
        points = lidar[m]

        # generate free space points based on points
        dn = torch.norm(points[:, :2], dim=1).view(-1, 1)
        dn1 = - points[:, 2:3] * torch.tan(torch.atan2(dn, -points[:, 2:3]) - alpha)
        delta_d = dn - dn1
        steps_arr = torch.linspace(0, 1, steps + 1)[:-1].view(1, steps).to(delta_d.device)
        tmp = (dn - steps_arr * delta_d) / dn  # Nxsteps
        xyz_new = points[:, None, :3] * tmp[:, :, None]  # Nxstepsx3
        if time_idx is not None:
            times = points[:, time_idx].view(-1, 1, 1).repeat(1, steps, 1)
            xyz_new = torch.cat([xyz_new, times], dim=-1)

        # 1.remove free space points with negative distances to lidar center
        # 2.remove free space points higher than z_min
        # 3.remove duplicated points with resolution 1m
        xyz_new = xyz_new[tmp > 0]
        # xyz_new = xyz_new[(xyz_new[..., 2] < min_h)]
        xyz_new = xyz_new[torch.randperm(len(xyz_new))]
        selected = torch.unique(torch.floor(xyz_new[..., :3] / 2).long(), return_inverse=True, dim=0)[1]
        xyz_new = scatter_mean(src=xyz_new, index=selected, dim=0)

        # pad free space point intensity as -1
        xyz_new = torch.cat([xyz_new[:, :3], - torch.ones_like(xyz_new[:, :1]), xyz_new[:, 3:]], dim=-1)
        data['points'] = torch.cat([lidar, xyz_new], dim=0)










