

def get_det_anchor_dense_cfg(gather_keys, scatter_keys, gt_keys,
        voxel_size, point_cloud_range, in_channels=256, stride=2):
    return dict(
            type='heads.det_anchor_dense.DetAnchorDense',
            gather_keys=gather_keys,
            scatter_keys=scatter_keys,
            gt_keys=gt_keys,
            in_channels=in_channels,
            target_assigner=dict(
                type='target_assigners.BoxAnchorAssigner',
                box_size=[3.9, 1.6, 1.56],
                dirs=[0, 90],
                voxel_size=voxel_size,
                lidar_range=point_cloud_range,
                stride=stride,
                pos_threshold=0.6,
                neg_threshold=0.45,
                score_thrshold=0.25,
                box_coder=dict(type='ResidualBoxCoder', mode='simple_dist')
            ),
            loss_cls = dict(type='FocalLoss', use_sigmoid=True,
                            gamma=2.0, alpha=0.25, loss_weight=0.25),
            loss_box = dict(type='SmoothL1Loss', loss_weight=1.0),
        )