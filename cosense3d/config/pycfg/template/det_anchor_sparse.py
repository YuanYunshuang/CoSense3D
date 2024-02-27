

def get_det_anchor_sparse_cfg(voxel_size, point_cloud_range,
                              in_channels=256, stride=2,
                              generate_roi_scr=False,
                              gather_keys=[], scatter_keys=[], gt_keys=[]):
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return dict(
                type='heads.det_center_sparse.DetCenterSparse',
                gather_keys=gather_keys,
                scatter_keys=scatter_keys,
                gt_keys=gt_keys,
                data_info=data_info,
                generate_roi_scr=generate_roi_scr,
                input_channels=in_channels,
                shared_conv_channel=256,
                get_predictions=True,
                stride=stride,
                cls_head_cfg=dict(name='UnitedClsHead'),
                reg_head_cfg=dict(name='UnitedRegHead', combine_channels=True, sigmoid_keys=['scr']),
                class_names_each_head=[['vehicle.car']],
                reg_channels=['box:6', 'dir:8', 'scr:4'],
                cls_assigner=dict(
                    type='target_assigners.BEVHardCenternessAssigner',
                    n_cls=1,
                    min_radius=1.0,
                    pos_neg_ratio=2,
                    max_mining_ratio=0,
                ),
                box_assigner=dict(
                    type='target_assigners.BoxCenterAssigner',
                    voxel_size=voxel_size,
                    lidar_range=point_cloud_range,
                    stride=stride,
                    detection_benchmark='Car',
                    class_names_each_head=[['vehicle.car']],
                    center_threshold=0.5,
                    box_coder=dict(type='CenterBoxCoder'),
                ),
                loss_cls=dict(type='EDLLoss', activation='exp', annealing_step=20, n_cls=2, loss_weight=5.0),
                loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
            )