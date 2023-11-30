from collections import OrderedDict

# point_cloud_range = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
# point_cloud_range_enlarged = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
point_cloud_range = [-144, -41.6, -3.0, 144, 41.6, 3.0]
point_cloud_range_test = [-140.8, -40, -3, 140.8, 40, 1]
voxel_size = [0.2, 0.2, 0.2]
data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
out_stride = 2

"""
gather_keys: 
    keys to gather data from cavs, key order is important, should match the forward input arguments order.
scatter_keys: 
    1st key in the list is used as the key for scattering and storing module output data to cav.
"""
shared_modules = OrderedDict(
    pts_backbone = dict(
        type='backbone3d.mink_unet.MinkUnet',
        gather_keys=['points'],
        scatter_keys=['pts_feat'],
        d=3,
        cache_strides=[2],
        in_dim=4,
        stride=out_stride,
        floor_height=point_cloud_range[2],
        data_info=data_info,
        height_compression=OrderedDict(p2=dict(channels=[128, 128, 128], steps=[5, 3]))
    ),

    fusion = dict(
        type='fusion.naive_fusion.NaiveFusion',
        gather_keys=['pts_feat', 'received_response'],
        scatter_keys=['fused_feat'],
        data_info=data_info,
        stride=out_stride,
        dim=128
    ),

    fusion_neck = dict(
        type='necks.dilation_spconv.DilationSpconv',
        gather_keys=['fused_feat'],
        scatter_keys=['fused_neck_feat'],
        data_info=data_info,
        d=2,
        convs=dict(p2=dict(kernels=[5, 5, 3], in_dim=128, out_dim=128))
    ),

    bev_head = dict(
        type='heads.bev.BEV',
        gather_keys=['fused_neck_feat'],
        scatter_keys=['bev'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d'],
        data_info=data_info,
        stride=out_stride,
        in_dim=128,
        target_assigner=dict(type='target_assigners.BEVPointAssigner'),
        loss_cls=dict(type='EDLLoss', activation='exp', annealing_step=40, n_cls=2, loss_weight=1.0),
    ),

    detection_head = dict(
        type='heads.det_center_sparse.DetCenterSparse',
        gather_keys=['fused_neck_feat'],
        scatter_keys=['detection'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d'],
        data_info=data_info,
        input_channels=128,
        shared_conv_channel=128,
        get_predictions=True,
        stride=out_stride,
        cls_head_cfg=dict(name='UnitedClsHead'),
        reg_head_cfg=dict(name='UnitedRegHead', combine_channels=True, sigmoid_keys=['scr']),
        class_names_each_head=[['vehicle.car']],
        reg_channels=['box:6', 'dir:8', 'scr:4'],
        cls_assigner=dict(
            type='target_assigners.BEVHardCenternessAssigner',
            n_cls=1,
            min_radius=1.0,
            pos_neg_ratio=5,
            mining_thr=0.5,
            max_mining_ratio=0.1,
            mining_start_epoch=10,
        ),
        box_assigner=dict(
            type='target_assigners.BoxCenterAssigner',
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            stride=out_stride,
            detection_benchmark='Car',
            class_names_each_head=[['vehicle.car']],
            center_threshold=0.5,
            box_coder=dict(type='CenterBoxCoder'),
        ),
        loss_cls=dict(type='EDLLoss', activation='exp', annealing_step=20, n_cls=2, loss_weight=1.0),
        loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
    ),
)

train_hooks = [
        dict(type='MemoryUsageHook'),
        dict(type='TrainTimerHook'),
        dict(type="CheckPointsHook", epoch_every=10)
    ]


test_hooks = [
        dict(type="DetectionNMSHook", nms_thr=0.1, pre_max_size=500),
        dict(type="EvalDetectionHook", save_result=True, pc_range=point_cloud_range_test, metrics=['OPV2V', 'CoSense3D']),
        dict(type="BEVSparseToDenseHook", lidar_range=point_cloud_range_test, voxel_size=voxel_size, stride=4),
        dict(type="EvalDenseBEVHook", thr=0.5)
    ]

plots = [
    dict(title='BEVSparseCanvas', lidar_range=point_cloud_range, width=10, height=4, nrows=1, ncols=1,
         data_keys=['bev', 'global_labels']),
    dict(title='DetectionCanvas', lidar_range=point_cloud_range, width=10, height=4, nrows=1, ncols=1,
         data_keys=['detection', 'global_labels'])
]