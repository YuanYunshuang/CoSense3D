from collections import OrderedDict

# point_cloud_range = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
# point_cloud_range_enlarged = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
point_cloud_range = [-140.8, -40, -3, 140.8, 40, 1] # [-144, -41.6, -3.0, 144, 41.6, 3.0]
point_cloud_range_test = [-140.8, -38.4, -3.0, 140.8, 38.4, 3.0]
voxel_size = [0.4, 0.4, 6]
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
        type='backbone3d.pillar_bev.PillarBEV',
        gather_keys=['points'],
        scatter_keys=['bev_feat'],
        in_channels=64,
        layer_nums=[3, 5, 8],
        layer_strides=[2, 2, 2],
        downsample_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        upsample_channels=[128, 128, 128],
        voxel_generator=dict(
            type='voxel_generator.VoxelGenerator',
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            max_points_per_voxel=32,
            max_voxels_train=32000,
            max_voxels_test=70000
        ),
        pillar_encoder=dict(
            type='pillar_encoder.PillarEncoder',
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            features=['xyz', 'intensity', 'absolute_xyz'],
            channels=[64]
        ),
        bev_shrinker=dict(
            type='downsample_conv.DownsampleConv',
            in_channels=384,  # 128 * 3
            dims=[256]
        ),
    ),

    fusion = dict(
        type='fusion.dense_bev_fusion.BEVMaxoutFusion',
        gather_keys=['bev_feat', 'received_response'],
        scatter_keys=['bev_feat_fused'],
    ),

    detection_head = dict(
        type='heads.det_anchor_dense.DetAnchorDense',
        gather_keys=['bev_feat_fused'],
        scatter_keys=['detection_out'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d'],
        in_channels=256,
        target_assigner=dict(
            type='target_assigners.BoxAnchorAssigner',
            box_size=[3.9, 1.6, 1.56],
            dirs=[0, 90],
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            stride=2,
            pos_threshold=0.6,
            neg_threshold=0.45,
            box_coder=dict(type='ResidualBoxCoder', mode='simple_dist')
        ),
        loss_cls = dict(type='FocalLoss', use_sigmoid=True,
                        gamma=2.0, alpha=0.25, loss_weight=0.25),
        loss_box = dict(type='SmoothL1Loss', loss_weight=1.0),
    )
)

train_hooks = [
        dict(type='MemoryUsageHook'),
        dict(type='TrainTimerHook'),
        dict(type="CheckPointsHook")
    ]


test_hooks = [
        dict(type="DetectionNMSHook", nms_thr=0.1, pre_max_size=500),
        dict(type="EvalOPV2VDetectionHook"),
        dict(type="BEVSparseToDenseHook", lidar_range=point_cloud_range_test, voxel_size=voxel_size, stride=4),
        dict(type="EvalDenseBEVHook", thr=0.5)
    ]

plots = [
    dict(title='BEVSparseCanvas', width=10, height=4, nrows=1, ncols=1),
    dict(title='DetectionCanvas', width=10, height=4, nrows=1, ncols=1)
]