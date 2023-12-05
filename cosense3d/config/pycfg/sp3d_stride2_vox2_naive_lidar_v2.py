from collections import OrderedDict

# point_cloud_range = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
# point_cloud_range_enlarged = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
point_cloud_range = [-144, -41.6, -3.0, 144, 41.6, 3.0]
point_cloud_range_test = [-140.8, -38.4, -3.0, 140.8, 38.4, 1.0]
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

    detection_head = dict(
        type='heads.det_anchor_sparse.DetAnchorSparse',
        gather_keys=['fused_neck_feat'],
        scatter_keys=['detection'],
        gt_keys=['fused_neck_feat', 'global_bboxes_3d', 'global_labels_3d'],
        in_channels=128,
        target_assigner=dict(
            type='target_assigners.BoxSparseAnchorAssigner',
            box_size=[3.9, 1.6, 1.56],
            dirs=[0, 90],
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            stride=2,
            pos_threshold=0.6,
            neg_threshold=0.45,
            score_thrshold=0.25,
            box_coder=dict(type='ResidualBoxCoder', mode='simple_dist')
        ),
        loss_cls = dict(type='FocalLoss', use_sigmoid=True,
                        gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_box = dict(type='SmoothL1Loss', loss_weight=2.0),
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
        # dict(type="BEVSparseToDenseHook", lidar_range=point_cloud_range_test, voxel_size=voxel_size, stride=4),
        # dict(type="EvalDenseBEVHook", thr=0.5)
    ]

plots = [
    # dict(title='BEVSparseCanvas', lidar_range=point_cloud_range, width=10, height=4, nrows=1, ncols=1,
    #      data_keys=['bev', 'global_labels']),
    dict(title='DetectionCanvas', lidar_range=point_cloud_range, width=10, height=4, nrows=1, ncols=1,
         data_keys=['detection', 'global_labels'])
]