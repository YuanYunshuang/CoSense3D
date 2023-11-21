from collections import OrderedDict

# point_cloud_range = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
# point_cloud_range_enlarged = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
point_cloud_range = [-140.8, -41.6, -3, 140.8, 41.6, 1] # [-144, -41.6, -3.0, 144, 41.6, 3.0]
point_cloud_range_test = [-140.8, -38.4, -3.0, 140.8, 38.4, 3.0]
voxel_size = [0.1, 0.1, 0.1]
data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
out_stride = 2

pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(),
    LoadAnnotations=dict(load3d_global=True, min_num_pts=2, load3d_local=True),
)

inference_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(),
)

data_manager = dict(
    train=dict(
        aug=dict(
            rot_range=[-1.57, 1.57],
            flip='xy',
            scale_ratio_range=[0.95, 1.05],
        ),
        pre_process=['remove_empty_boxes']
    ),
    test=dict()
)

"""
gather_keys: 
    keys to gather data from cavs, key order is important, should match the forward input arguments order.
scatter_keys: 
    1st key in the list is used as the key for scattering and storing module output data to cav.
"""
shared_modules = OrderedDict(
    pts_backbone = dict(
        type='backbone3d.spconv.Spconv',
        gather_keys=['points'],
        scatter_keys=['voxel_feat', 'bev_feat'],
        in_channels=4,
        out_channels=64,
        voxel_generator=dict(
            type='voxel_generator.VoxelGenerator',
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            max_points_per_voxel=32,
            max_voxels_train=32000,
            max_voxels_test=70000
        ),
        voxel_encoder=dict(
            type='voxel_encoder.MeanVFE',
            num_point_features=4,
        ),
        bev_neck=dict(type='ssfa.SSFA', in_channels=64, out_channels=128),
    ),

    detection_head_local = dict(
        type='heads.det_anchor_dense.DetAnchorDense',
        gather_keys=['bev_feat'],
        scatter_keys=['detection_local'],
        gt_keys=['local_bboxes_3d', 'local_labels_3d'],
        in_channels=128,
        get_boxes_when_training=True,
        target_assigner=dict(
            type='target_assigners.BoxAnchorAssigner',
            box_size=[3.9, 1.6, 1.56],
            dirs=[0, 90],
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            stride=8,
            pos_threshold=0.6,
            neg_threshold=0.45,
            score_thrshold=0.25,
            box_coder=dict(type='ResidualBoxCoder', mode='simple_dist')
        ),
        loss_cls = dict(type='FocalLoss', use_sigmoid=True,
                        gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_box = dict(type='SmoothL1Loss', loss_weight=2.0),
    ),

    cpm_composer=dict(
        type='necks.cpm_composer.KeypointComposer',
        gather_keys=['detection_local', 'bev_feat', "voxel_feat", 'points'],
        scatter_keys=['keypoint_feat'],
        vsa=dict(
            type='vsa.VoxelSetAbstraction',
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            num_keypoints=4096,
            num_out_features=32,
            num_bev_features=128,
            num_rawpoint_features=3,
            enlarge_selection_boxes=True,
        )
    ),

    # fusion=dict(
    #     type='fusion.keypoints.VoxelKeypointsFusion',
    #     gather_keys=['bev_feat', 'received_response'],
    #     scatter_keys=['bev_feat_fused'],
    #     feature_dim=128,
    # ),

    # detection_head_global = dict(
    #     type='heads.det_roi_refine.DetROIRefine',
    # )
)

train_hooks = [
        dict(type='MemoryUsageHook'),
        dict(type='TrainTimerHook'),
        dict(type="CheckPointsHook", epoch_every=10)
    ]


test_hooks = [
        dict(type="DetectionNMSHook", nms_thr=0.15, pre_max_size=100),
        dict(type="EvalOPV2VDetectionHook", save_result=True),
    ]

plots = [
    dict(title='DetectionCanvas', width=10, height=4, nrows=1, ncols=1)
]