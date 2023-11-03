import torch
from scipy.spatial.transform.rotation import Rotation as R


def transform_points(data, transform, scale):
    C = data['points'].shape[-1]
    points = data['points'][:, :3]
    points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1).T
    points_homo = transform @ points_homo

    if scale is not None:
        points_homo[:2] *= scale

    if C > 3:
        data['points'] = torch.cat([points_homo[:3].T,
                                           data['points'][:, 3:]], dim=-1)
    else:
        data['points'] = points_homo[:3].T


def transform_bboxes_3d(boxes, transform, scale):
    yaw = R.from_matrix(transform[:3, :3].cpu().numpy()).as_euler('xyz')[-1]
    boxes[:, :3] = (transform[:3, :3] @ boxes[:, :3].T + transform[:3, 3].view(3, 1)).T
    boxes[:, 6] += yaw
    if scale is not None:
        boxes[:, :6] *= scale
        if boxes.shape[-1] == 9:
            # scale velocity
            boxes[:, 7:] *= scale
    return boxes


def transform_annos_global(data, transform, scale):
    if 'global_bboxes_3d' not in data:
        return
    data['global_bboxes_3d'] = transform_bboxes_3d(data['global_bboxes_3d'], transform, scale)


def transform_annos_local(data, transform, scale):
    if 'local_bboxes_3d' not in data:
        return
    data['local_bboxes_3d'] = transform_bboxes_3d(data['local_bboxes_3d'], transform, scale)


def filter_range_mask(points, lidar_range):
    lr = lidar_range.to(points.device)
    mask = (points[:, :3] > lr[:3].view(1, 3)) & (points[:, :3] < lr[3:].view(1, 3))
    return mask.all(dim=-1)


def filter_range_points(data, lidar_range):
    mask = filter_range_mask(data['points'], lidar_range)
    data['points'] = data['points'][mask]


def filter_range_annos_global(data, lidar_range):
    if 'global_bboxes_3d' not in data:
        return
    mask = filter_range_mask(data['global_bboxes_3d'][:, :3], lidar_range)
    data['global_bboxes_3d'] = data['global_bboxes_3d'][mask]
    data['global_labels_3d'] = data['global_labels_3d'][mask]
    data['global_names'] = [data['global_names'][i] for i, m in enumerate(mask) if m]


def filter_range_annos_local(data, lidar_range):
    if 'local_bboxes_3d' not in data:
        return
    mask = filter_range_mask(data['local_bboxes_3d'][:, :3], lidar_range)
    data['local_bboxes_3d'] = data['local_bboxes_3d'][mask]
    data['local_labels_3d'] = data['local_labels_3d'][mask]
    data['local_names'] = [data['local_names'][i] for i, m in enumerate(mask) if m]


class DataOnlineProcessor:

    @staticmethod
    def cav_aug_transform(data, transform, aug_params,
                          apply_to=['points', 'imgs', 'annos_global']):
        # augmentation
        if 'rot' in aug_params:
            transform = aug_params['rot'].to(transform.device) @ transform
        scale = aug_params['scale'].item() if 'scale' in aug_params else None
        for k in apply_to:
            func = globals().get(f'transform_{k}', False)
            if not func:
                raise NotImplementedError(f"function `transform_{k}` not implemented.")
            else:
                func(data, transform, scale)

    @staticmethod
    def filter_range(data, lidar_range, apply_to):
        for k in apply_to:
            func = globals().get(f'filter_range_{k}', False)
            if not func:
                raise NotImplementedError(f"function `filter_range_{k}` not implemented.")
            else:
                func(data, lidar_range)








