import torch
from cosense3d.model.utils import metric2indices


def mink_coor_limit(lidar_range, voxel_size, stride):
    if not isinstance(voxel_size, list):
        voxel_size = [voxel_size, voxel_size]
    lr = lidar_range
    x_max = (round(lr[3] / voxel_size[0]) - 1) // stride * stride  # relevant to ME
    x_min = (round(lr[0] / voxel_size[0]) + 1) // stride * stride - stride  # relevant to ME
    y_max = (round(lr[4] / voxel_size[1]) - 1) // stride * stride
    y_min = (round(lr[1] / voxel_size[1]) + 1) // stride * stride - stride
    return [x_min, x_max, y_min, y_max]


def update_me_essentials(self, data_info, stride=None):
    """
    Update essential variables for ME-based models
    Args:
        self: instance of a python class
        data_info:
            det_r: float
            lidar_range: [xmin, ymin, zmin, xmax, ymax, zmax]
            voxel_size: [vx, vy, vz]
        stride: int
    """
    for k, v in data_info.items():
        setattr(self, k, v)

    if getattr(self, 'det_r', False):
        lr = [-self.det_r, -self.det_r, 0, self.det_r, self.det_r, 0]
    elif getattr(self, 'lidar_range', False):
        lr = self.lidar_range
    else:
        raise NotImplementedError
    setattr(self, 'lidar_range', lr)

    if stride is not None:
        setattr(self, 'stride', stride)
        setattr(self, 'res', (self.stride * self.voxel_size[0], self.stride * self.voxel_size[1]))
        setattr(self, 'mink_xylim', mink_coor_limit(lr, self.voxel_size, self.stride))
        setattr(self, 'size_x', round((lr[3] - lr[0]) / self.res[0]))
        setattr(self, 'size_y', round((lr[4] - lr[1]) / self.res[1]))
        setattr(self, 'offset_sz_x', round(lr[0] / self.res[0]))
        setattr(self, 'offset_sz_y', round(lr[1] / self.res[1]))


def bev_sparse_to_dense(self, preds):
    conf, unc = preds['conf'], preds['unc'],
    ctrs = preds['centers'][:, :3]  # N 2
    batch_size = ctrs[:, 0].max().int() + 1
    conf_map = torch.zeros((batch_size, self.size_x, self.size_y, 2),
                           device=conf.device)
    unc_map = torch.ones((batch_size, self.size_x, self.size_y),
                         device=unc.device)
    inds = metric2indices(ctrs, self.res).T
    inds[1] -= self.offset_sz_x
    inds[2] -= self.offset_sz_y
    conf_map[inds[0], inds[1], inds[2]] = conf
    unc_map[inds[0], inds[1], inds[2]] = unc
    return conf_map, unc_map







