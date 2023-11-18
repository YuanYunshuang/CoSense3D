from collections import OrderedDict

point_cloud_range = [-50, -50, -3, 50, 50, 1]
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
        num_layers=34,
        feat_indices=(2, 3, 4),
        out_index=(2, 3, 4),
        img_size=img_size,
        # neck=dict(
        #     type='fpn.FPN',
        #     in_channels=[256, 512],
        #     out_channels=128,
        #     num_outs=2
        # )
    ),

    img2bev = dict(
        type='projection.fax.FAXModule',
        gather_keys=['img_feat'],
        scatter_keys=['bev_feat'],
        dim=[128, 128, 128],
        middle=[2, 2, 2],
        bev_embedding=dict(
            sigma=1.0,
            bev_height=256,
            bev_width=256,
            h_meters=100,
            w_meters=100,
            offset=0.0,
            upsample_scales=[2, 4, 8],
        ),
        cross_view=dict(
            img_size=img_size,
            no_image_features=False,
            skip=True,
            heads=[4, 4, 4],
            dim_head=[32, 32, 32],
            qkv_bias=True,
        ),
        cross_view_swap=dict(
            rel_pos_emb=False,
            q_win_size=[[16, 16], [16, 16], [32, 32]],
            feat_win_size=[[8, 8], [8, 8], [16, 16]],
            bev_embedding_flag=[True, False, False],
        ),
        self_attn=dict(
            dim_head=32,
            dropout=0.1,
            window_size=32,
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