from collections import OrderedDict

# point_cloud_range = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
# point_cloud_range_enlarged = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
point_cloud_range = [-51.2, -38.4, -5, 51.2, 38.4, 3]
position_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
voxel_size = [0.2, 0.2, 8]
data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
img_size = (384, 768)
num_classes = 1

"""
gather_keys: 
    keys to gather data from cavs, key order is important, should match the forward input arguments order.
scatter_keys: 
    1st key in the list is used as the key for scattering and storing module output data to cav.
"""
shared_modules = OrderedDict(
    img_backbone = dict(
        type='backbone2d.resnet_encoder.ResnetEncoder',
        gather_keys=['img'],
        scatter_keys=['img_feat', 'img_coor'],
        num_layers=18,
        feat_indices=(3, 4),
        out_index=3,
        img_size=img_size,
        neck=dict(
            type='fpn.FPN',
            in_channels=[256, 512],
            out_channels=128,
            num_outs=2
        )
    ),

    img_roi = dict(
        type='heads.img_focal.ImgFocal',
        gather_keys=['img_feat', 'img_coor'],
        scatter_keys=['img_roi'],
        gt_keys=['labels2d', 'centers2d', 'bboxes2d', 'img_size'],
        in_channels=128,
        embed_dims=128,
        num_classes=num_classes,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        center_assigner= dict(type='target_assigners.HeatmapAssigner'),
        box_assigner=dict(
            type='target_assigners.HungarianAssigner2D',
            cls_cost = dict(type='focal_loss', weight=2.),
            reg_cost = dict(type='bboxl1', weight=5.0, box_format='xywh'),
            iou_cost = dict(type='giou', weight=2.0),
            centers2d_cost = dict(type='l1', weight=10.0)
        ),
    ),

    img2bev = dict(
        type='projection.petr.PETR',
        gather_keys=['img_feat', 'img_roi', 'img_coor', 'img_size', 'intrinsics', 'lidar2img'],
        scatter_keys=['petr_feat'],
        in_channels=128,
        LID=True,
        position_range=position_range,
        transformer=dict(
            type='transformer.PETRTransformer',
            decoder=dict(
                type='TransformerDecoder',
                return_intermediate=True,
                num_layers=3,
                transformerlayers=dict(
                    type='TransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention',  # fp16 for 2080Ti training (save GPU memory).
                             embed_dims=128,
                             num_heads=8,
                             dropout=0.1)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=128,
                        feedforward_channels=1024,
                        num_fcs=2,
                        dropout=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    with_cp=False,  ###use checkpoint to save memory
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')),
                )
            )
        ),

    detection = dict(
        type='heads.petr_head.PETRHead',
        gather_keys=['petr_feat'],
        scatter_keys=['petr_out'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d'],
        embed_dims=128,
        pc_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        num_classes=1,
        box_assigner=dict(
            type='target_assigners.HungarianAssigner3D',
            cls_cost=dict(type='focal_loss', weight=2.),
            reg_cost=dict(type='l1', weight=.25),
            iou_cost=dict(type='iou', weight=0.0),
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True,
                      gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        # loss_iou=dict(type='GIoULoss', loss_weight=0.0),
    )

    )

train_hooks = [
        dict(type='MemoryUsageHook'),
        dict(type='TrainTimerHook'),
        dict(type="CheckPointsHook", epoch_every=10)
    ]


test_hooks = [
        dict(type="EvalOPV2VDetectionHook"),
        dict(type="EvalDenseBEVHook", thr=0.5)
    ]


plots = [
    # dict(title='BEVSparseCanvas', width=10, height=4, nrows=1, ncols=1),
    # dict(title='DetectionCanvas', width=10, height=4, nrows=1, ncols=1)
]