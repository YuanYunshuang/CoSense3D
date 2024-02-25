from collections import OrderedDict
from cosense3d.config.pycfg.template.petr_transformer import get_petr_transformer_cfg
from cosense3d.config.pycfg.template.voxnet import get_voxnet_cfg
from cosense3d.config.pycfg.template.query_guided_petr_head import get_query_guided_petr_head_cfg
from cosense3d.config.pycfg.template.det_anchor_dense import get_det_anchor_dense_cfg

voxel_size = [0.4, 0.4, 0.4]
out_stride = 2


def get_shared_modules(point_cloud_range, attn1='MultiheadFlashAttention'):
    """
    gather_keys: 
        keys to gather data from cavs, key order is important, should match the forward input arguments order.
    scatter_keys: 
        1st key in the list is used as the key for scattering and storing module output data to cav.
    """
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return OrderedDict(
        pts_backbone=get_voxnet_cfg(
            gather_keys=['points'],
            scatter_keys=['bev_feat', 'multi_scale_bev_feat'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            sparse_cml=True,
            neck=dict(type='bev_rpn.CustomRPN', out_channels=256, num_layers=2),
        ),

        roi_head = dict(
            type='heads.multitask_head.MultiTaskHead',
            gather_keys=['multi_scale_bev_feat'],
            scatter_keys=['det_local_dense', 'bev_local_dense'],
            gt_keys=['local_bboxes_3d', 'local_labels_3d'],
            heads=[
                get_det_anchor_dense_cfg(
                    gather_keys=['bev_feat'],
                    scatter_keys=['detection'],
                    gt_keys=['global_bboxes_3d', 'global_labels_3d'],
                    voxel_size=voxel_size,
                    point_cloud_range=point_cloud_range,
                    pos_threshold=0.3,
                    neg_threshold=0.1,
                    score_thrshold=0.15,
                ),
                dict(
                    type='heads.bev_dense.BevRoIDenseHead',
                    in_dim=256,
                )
            ],
            strides=[2, 8],
            losses=[True, False],
        ),

        formatting = dict(
            type='necks.formatting.DenseToSparse',
            gather_keys=['multi_scale_bev_feat', 'det_local_dense', 'bev_local_dense'],
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
            global_ref_time=0.05,
            lidar_range=point_cloud_range,
            transformer=get_petr_transformer_cfg(attn1)
        ),

        spatial_fusion=dict(
            type='fusion.spatial_query_fusion.SpatialQueryFusion',
            gather_keys=['temp_fusion_feat', 'received_response'],
            scatter_keys=['spatial_fusion_feat'],
            in_channels=256,
            pc_range=point_cloud_range,
            resolution=0.8
        ),

        det1_head = get_query_guided_petr_head_cfg(
            gather_keys=['temp_fusion_feat'],
            scatter_keys=['detection_local'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=False,
        ),

        det2_head = get_query_guided_petr_head_cfg(
            gather_keys=['spatial_fusion_feat'],
            scatter_keys=['detection'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=True,
        ),

    )

