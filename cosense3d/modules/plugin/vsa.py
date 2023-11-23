import copy
import random

import torch
import torch.nn as nn

from cosense3d.ops import pointnet2_utils
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.utils.common import get_voxel_centers, cat_coor_with_idx

sa_layer_default=dict(
    raw_points=dict(
    mlps=[[16, 16], [16, 16]],
    pool_radius=[0.4, 0.8],
    n_sample=[16, 16],
    ),
    x_conv1=dict(
        downsample_factor=1,
        mlps=[[16, 16], [16, 16]],
        pool_radius=[0.4, 0.8],
        n_sample=[16, 16],
    ),
    x_conv2=dict(
        downsample_factor=2,
        mlps=[[32, 32], [32, 32]],
        pool_radius=[0.8, 1.2],
        n_sample=[16, 32],
    ),
    x_conv3=dict(
        downsample_factor=4,
        mlps=[[64, 64], [64, 64]],
        pool_radius=[1.2, 2.4],
        n_sample=[16, 32],
    ),
    x_conv4=dict(
        downsample_factor=8,
        mlps=[[64, 64], [64, 64]],
        pool_radius=[2.4, 4.8],
        n_sample=[16, 32],
    )
)

default_feature_source = ['bev', 'x_conv1', 'x_conv2', 'x_conv3', 'x_conv4', 'raw_points']

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class VoxelSetAbstraction(nn.Module):
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 num_keypoints=4096,
                 num_out_features=32,
                 point_source='raw_points',
                 features_source=None,
                 num_bev_features=128,
                 bev_stride=8,
                 num_rawpoint_features=3,
                 enlarge_selection_boxes=True,
                 sa_layer=None,
                 **kwargs):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.features_source = default_feature_source \
            if features_source is None \
            else features_source
        self.num_keypoints = num_keypoints
        self.num_out_features = num_out_features
        self.point_source = point_source
        self.num_bev_features = num_bev_features
        self.bev_stride = bev_stride
        self.num_rawpoint_features = num_rawpoint_features
        self.enlarge_selection_boxes = enlarge_selection_boxes

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        sa_layer = sa_layer_default if sa_layer is None else sa_layer
        for src_name in self.features_source :
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = sa_layer[src_name]['downsample_factor']
            mlps = copy.copy(sa_layer[src_name]['mlps'])
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_utils.StackSAModuleMSG(
                radii=sa_layer[src_name]['pool_radius'],
                nsamples=sa_layer[src_name]['n_sample'],
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'bev' in self.features_source:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.features_source:
            mlps = copy.copy(sa_layer['raw_points']['mlps'])
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_utils.StackSAModuleMSG(
                radii=sa_layer['raw_points']['pool_radius'],
                nsamples=sa_layer['raw_points']['n_sample'],
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.num_out_features, bias=False),
            nn.BatchNorm1d(self.num_out_features),
            nn.ReLU(),
        )
        self.num_point_features = self.num_out_features
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, keypoints_list, bev_features):
        B = len(bev_features)
        point_bev_features_list = []
        for i in range(B):
            keypoints = keypoints_list[i]
            x_idxs = (keypoints[..., 0] - self.point_cloud_range[0]) / self.voxel_size[0]
            y_idxs = (keypoints[..., 1] - self.point_cloud_range[1]) / self.voxel_size[1]
            x_idxs = x_idxs / self.bev_stride
            y_idxs = y_idxs / self.bev_stride
            cur_bev_features = bev_features[i].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, x_idxs, y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_sampled_points(self, batch_size, points, voxel_coords):
        B = len(points)
        keypoints_list = []
        for i in range(B):
            if self.point_source == 'raw_points':
                src_points = points[i]
            else:
                raise NotImplementedError
            # # generate random keypoints in the perception view field
            # keypoints = torch.randn((self.num_keypoints, 4), device=src_points.device)
            # keypoints[..., 0] = keypoints[..., 0] * 140
            # keypoints[..., 1] = keypoints[..., 1] * 40
            # # points with height flag 10 are padding/invalid, for later filtering
            # keypoints[..., 2] = 10.0

            sampled_points = src_points.unsqueeze(dim=0)  # (1, N, 3)
            # sample points with FPS
            # some cropped pcd may have very few points, select various number
            # of points to ensure similar sample density
            # 50000 is approximately the number of points in one full pcd
            num_kpts = int(self.num_keypoints * sampled_points.shape[1] / 50000) + 1
            num_kpts = min(num_kpts, self.num_keypoints)
            cur_pt_idxs = pointnet2_utils.furthest_point_sample(
                sampled_points[..., :3].contiguous(), num_kpts
            ).long()

            if sampled_points.shape[1] < num_kpts:
                empty_num = num_kpts - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            # keypoints[:len(kpts[0]), :] = kpts
            keypoints_list.append(keypoints)

        # keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints_list

    def forward(self, det_out, bev_feat, voxel_feat, points):
        B = len(points)
        preds = [x['preds'] for x in det_out]
        keypoints = self.get_sampled_points(B, points, voxel_feat) # BxNx4

        # Only select the points that are in the predicted bounding boxes
        boxes = cat_coor_with_idx([x['box'] for x in preds])
        keypoints = cat_coor_with_idx(keypoints)
        if self.enlarge_selection_boxes:
            boxes[:, 4:7] += 0.5
        kpt_mask = points_in_boxes_gpu(keypoints, boxes, batch_size=B) > 0

        # Ensure there are more than 2 points are selected to satisfy the
        # condition of batch norm in the FC layers of feature fusion module
        if (kpt_mask).sum() < 2:
            kpt_mask[torch.randint(0, 1024, (2,))] = True

        point_features_list = []
        if 'bev' in self.features_source:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints[..., :3], bev_feat, B,
                bev_stride=8
            )
            point_features_list.append(point_bev_features[kpt_mask])

        batch_size, num_keypoints, _ = keypoints.shape

        new_xyz = keypoints[kpt_mask]
        new_xyz_batch_cnt = torch.tensor([(mask).sum() for mask in kpt_mask], device=new_xyz.device).int()

        if 'raw_points' in self.features_source:
            raw_points = points
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            indices = raw_points[:, 0].long()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (indices == bs_idx).sum()
            point_features = None

            pooled_points, pooled_features = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz[:, :3].contiguous(),
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
            )
            point_features_list.append(pooled_features)

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = feature_dict['multi_scale_features'][src_name].indices
            xyz = get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz[:, :3].contiguous(),
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=feature_dict['multi_scale_features'][src_name].features.contiguous(),
            )

            point_features_list.append(pooled_features)

        point_features = torch.cat(point_features_list, dim=1)

        out_dict = {}
        out_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        cur_idx = 0
        out_dict['point_features'] = []
        out_dict['point_coords'] = []
        for num in new_xyz_batch_cnt:
            out_dict['point_features'].append(point_features[cur_idx:cur_idx + num])
            out_dict['point_coords'].append(new_xyz[cur_idx:cur_idx + num])
            cur_idx += num

        return out_dict
