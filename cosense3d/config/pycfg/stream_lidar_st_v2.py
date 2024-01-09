import copy
from collections import OrderedDict
from cosense3d.utils.misc import update_dict


point_cloud_range = [-144, -41.6, -3.0, 144, 41.6, 1.0]
point_cloud_range_test = [-140.8, -38.4, -3.0, 140.8, 38.4, 1.0]
voxel_size = [0.4, 0.4, 0.4]
data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
out_stride = 2

pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time']),
    LoadAnnotations=dict(load3d_global=True, load3d_local=True,
                         load_global_time=True, with_velocity=True, min_num_pts=3),
)

inference_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time']),
)

data_manager = dict(
    train=dict(
        aug=dict(
            rot_range=[-1.57, 1.57],
            flip='xy',
            scale_ratio_range=[0.95, 1.05],
        ),
        pre_process=['remove_local_empty_boxes',
                     'remove_global_empty_boxes']
    ),
    test=dict(
        aug=dict()
        # aug=dict(
        #     rot_range=[-1.57, 1.57],
        #     flip='xy',
        #     scale_ratio_range=[0.95, 1.05],
        # ),
    )
)

bev_head_cfg = dict(
    type='heads.bev.BEV',
    data_info=data_info,
    stride=out_stride,
    in_dim=256,
    target_assigner=dict(type='target_assigners.BEVPointAssigner'),
    loss_cls=dict(type='EDLLoss', activation='relu', annealing_step=50, n_cls=2, loss_weight=1.0),
)

det_head_cfg = dict(
    type='heads.det_center_sparse.DetCenterSparse',
    data_info=data_info,
    input_channels=256,
    shared_conv_channel=256,
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
        pos_neg_ratio=2,
        max_mining_ratio=0,
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
    loss_cls=dict(type='EDLLoss', activation='exp', annealing_step=20, n_cls=2, loss_weight=5.0),
    loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
)


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
        scatter_keys=['bev_feat'],
        d=3,
        cache_strides=[2, 8],
        kernel_size_layer1=5,
        in_dim=4,
        stride=out_stride,
        floor_height=point_cloud_range[2],
        data_info=data_info,
        height_compression=OrderedDict(
            p2=dict(channels=[128, 256, 384], steps=[5, 2]),
            p8=dict(channels=[128, 256], steps=[2])
        )
    ),

    backbone_neck = dict(
        type='necks.dilation_spconv.DilationSpconv',
        gather_keys=['bev_feat'],
        scatter_keys=['bev_feat'],
        data_info=data_info,
        d=2,
        convs=dict(
            p2=dict(kernels=[3, 3, 3], in_dim=384, out_dim=256),
            p8=dict(kernels=[3, 3, 3], in_dim=256, out_dim=256)
        )
    ),

    roi_head = dict(
        type='heads.multitask_head.MultiTaskHead',
        gather_keys=['bev_feat'],
        scatter_keys=['det_local', 'bev_local'],
        gt_keys=['local_bboxes_3d', 'local_labels_3d'],
        heads=[update_dict(copy.copy(det_head_cfg), dict(generate_roi_scr=True)),
               bev_head_cfg],
        strides=[2, 8],
        losses=[True, False]
    ),

    temporal_fusion = dict(
        type='fusion.temporal_fusion.LocalTemporalFusion',
        gather_keys=['det_local', 'bev_local', 'bev_feat', 'memory'],
        scatter_keys=['temp_fusion_feat'],
        in_channels=256,
        ref_pts_stride=2,
        feature_stride=8,
        transformer_itrs=1,
        global_ref_time=0.05,
        lidar_range=point_cloud_range,
        transformer=dict(
            type='transformer.PETRTemporalTransformer',
            decoder=dict(
                type='TransformerDecoder',
                return_intermediate=True,
                num_layers=1,
                transformerlayers=dict(
                    type='TransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention', #fp16 for 2080Ti training (save GPU memory).
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            fp16=False),
                        dict(
                            type='MultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        dropout=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    with_cp=False,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm',
                                     'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )
        ),

    ),

    spatial_fusion = dict(
        type='fusion.spatial_query_fusion.SpatialQueryFusion',
        gather_keys=['temp_fusion_feat', 'received_response'],
        scatter_keys=['spatial_fusion_feat'],
        in_channels=256,
        pc_range=point_cloud_range,
        resolution=0.8
    ),

    det1_head = dict(
        type='heads.query_guided_petr_head.QueryGuidedPETRHead',
        gather_keys=['temp_fusion_feat'],
        scatter_keys=['detection_local'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d', 'detection_local'],
        sparse=False,
        embed_dims=256,
        num_reg_fcs=1,
        num_pred=1,
        pc_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        num_classes=1,
        reg_channels=['box:6', 'dir:8', 'scr:4', 'vel:2'],
        cls_assigner=dict(
            type='target_assigners.BEVHardCenternessAssigner',
            n_cls=1,
            min_radius=1.0,
            pos_neg_ratio=0,
            mining_thr=0,
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
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                      gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
    ),

    det2_head=dict(
        type='heads.query_guided_petr_head.QueryGuidedPETRHead',
        gather_keys=['spatial_fusion_feat'],
        scatter_keys=['detection'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d', 'detection'],
        sparse=True,
        embed_dims=256,
        num_reg_fcs=1,
        num_pred=1,
        pc_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        num_classes=1,
        reg_channels=['box:6', 'dir:8', 'scr:4', 'vel:2'],
        cls_assigner=dict(
            type='target_assigners.BEVHardCenternessAssigner',
            n_cls=1,
            min_radius=1.0,
            pos_neg_ratio=0,
            mining_thr=0,
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
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                      gamma=2.0, alpha=0.25, loss_weight=2.0),
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
    ]

plots = [
    # dict(title='BEVSparseCanvas', lidar_range=point_cloud_range_test, width=10, height=4, nrows=1, ncols=1,
    #      data_keys=['bevseg_local', 'local_labels']),
    # dict(title='DetectionScoreMap', lidar_range=point_cloud_range_test, width=10, height=4, nrows=1, ncols=1,
    #      data_keys=['detection_local']),
    dict(title='DetectionCanvas', lidar_range=point_cloud_range_test, width=10, height=4, nrows=1, ncols=1,
         data_keys=['detection', 'global_labels'])
]