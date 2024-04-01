import copy
from collections import OrderedDict
from cosense3d.config.pycfg.base import opv2v, hooks
from cosense3d.config.pycfg.template.minkunet import get_minkunet_cfg
from cosense3d.config.pycfg.template.query_guided_petr_head import get_query_guided_petr_head_cfg
from cosense3d.config.pycfg.template.det_center_sparse import get_det_center_sparse_cfg
from cosense3d.config.pycfg.template.bev_head import get_bev_head_cfg


voxel_size = [0.2, 0.2, 0.2]
out_stride = 2


def get_shared_modules(point_cloud_range):
    """
    gather_keys: 
        keys to gather data from cavs, key order is important, should match the forward input arguments order.
    scatter_keys: 
        1st key in the list is used as the key for scattering and storing module output data to cav.
    """
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return OrderedDict(
        pts_backbone=get_minkunet_cfg(
            gather_keys=['points'],
            scatter_keys=['bev_feat'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            height_compression=True
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

        spatial_fusion=dict(
            type='fusion.spatial_query_fusion.SpatialQueryFusion',
            gather_keys=['temp_fusion_feat', 'received_response'],
            scatter_keys=['spatial_fusion_feat'],
            in_channels=256,
            pc_range=point_cloud_range,
            resolution=0.8
        ),

        det_head = get_query_guided_petr_head_cfg(
            gather_keys=['temp_fusion_feat'],
            scatter_keys=['detection_local'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=False,
        ),

        semseg_head = get_query_guided_petr_head_cfg(
            gather_keys=['spatial_fusion_feat'],
            scatter_keys=['detection'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=True,
        ),

    )

######################################################
#                     OPV2V
######################################################
test_hooks_opv2v = hooks.get_test_nms_eval_hooks(opv2v.point_cloud_range_test)
plots_opv2v = [hooks.get_detection_plot(opv2v.point_cloud_range_test)]
shared_modules_opv2v = get_shared_modules(opv2v.point_cloud_range)

#--------- Ablation 1 : No RoI regression-------------