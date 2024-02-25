from collections import OrderedDict

voxel_size = [0.4, 0.4, 4]
out_stride = 2


def get_shared_modules(point_cloud_range,
                       attn1='MultiheadFlashAttention',
                       global_ref_time=0.0):
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    bev_head_cfg = dict(
        type='heads.bev_dense.BevRoIDenseHead',
        in_dim=256,
    )

    det_head_cfg = dict(
            type='heads.det_anchor_dense.DetAnchorDense',
            gather_keys=['bev_feat'],
            scatter_keys=['detection'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            in_channels=256,
            target_assigner=dict(
                type='target_assigners.BoxAnchorAssigner',
                box_size=[3.9, 1.6, 1.56],
                dirs=[0, 90],
                voxel_size=voxel_size,
                lidar_range=point_cloud_range,
                stride=2,
                pos_threshold=0.3,
                neg_threshold=0.1,
                score_thrshold=0.15,
                box_coder=dict(type='ResidualBoxCoder', mode='simple_dist')
            ),
            loss_cls = dict(type='FocalLoss', use_sigmoid=True,
                            gamma=2.0, alpha=0.25, loss_weight=0.25),
            loss_box = dict(type='SmoothL1Loss', loss_weight=1.0),
        )


    """
    gather_keys: 
        keys to gather data from cavs, key order is important, should match the forward input arguments order.
    scatter_keys: 
        1st key in the list is used as the key for scattering and storing module output data to cav.
    """
    return OrderedDict(
        pts_backbone = dict(
            type='backbone3d.pillar_bev.PillarBEV',
            gather_keys=['points'],
            scatter_keys=['bev_feat', 'multi_scale_bev_feat'],
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

        roi_head = dict(
            type='heads.multitask_head.MultiTaskHead',
            gather_keys=['multi_scale_bev_feat'],
            scatter_keys=['det_local_dense', 'bev_local_dense'],
            gt_keys=['local_bboxes_3d', 'local_labels_3d'],
            heads=[det_head_cfg, bev_head_cfg],
            strides=[2, 8],
            losses=[True, False],
        ),

        formatting = dict(
            type='necks.formatting.DenseToSparse',
            gather_keys=['multi_scale_bev_feat', 'det_local_dense', 'bev_local_dense', 'points'],
            scatter_keys=['multi_scale_bev_feat', 'det_local_sparse', 'bev_local_sparse'],
            data_info=data_info,
            strides=[2, 8]
        ),

        temporal_fusion = dict(
            type='fusion.temporal_fusion.LocalTemporalFusion',
            gather_keys=['det_local_sparse', 'bev_local_sparse', 'multi_scale_bev_feat', 'memory'],
            scatter_keys=['temp_fusion_feat'],
            in_channels=256,
            ref_pts_stride=2,
            feature_stride=8,
            transformer_itrs=1,
            global_ref_time=global_ref_time,
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
                                type=attn1,
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1,
                                fp16=False if attn1 == 'MultiheadAttention' else True
                            ),
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
            type='fusion.maxout_fusion.SparseBEVMaxoutFusion',
            gather_keys=['temp_fusion_feat', 'received_response'],
            scatter_keys=['spatial_fusion_feat'],
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
                box_coder=dict(type='CenterBoxCoder', with_velo=True),
            ),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                          gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
        ),

        det2_head=dict(
            type='heads.query_guided_petr_head.QueryGuidedPETRHead',
            gather_keys=['spatial_fusion_feat'],
            scatter_keys=['detection'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
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
                box_coder=dict(type='CenterBoxCoder', with_velo=True),
            ),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                          gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
        ),

    )

