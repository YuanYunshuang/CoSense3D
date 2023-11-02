from collections import OrderedDict

# point_cloud_range = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
# point_cloud_range_enlarged = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
point_cloud_range = [-144, -41.6, -5.0, 144, 41.6, 3.0]
point_cloud_range_test = [-140.8, -38.4, -3.0, 140.8, 38.4, 3.0]
voxel_size = [0.2, 0.2, 8]
data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)

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
        voxel_size=voxel_size,
        d=2,
        cache_strides=[4],
        in_dim=4,
        stride=4,
        floor_height=point_cloud_range[2]
    ),

    fusion = dict(
        type='fusion.naive_fusion.NaiveFusion',
        gather_keys=['pts_feat', 'received_response'],
        scatter_keys=['fused_feat'],
        data_info=data_info,
        stride=4,
        dim=128
    ),

    fusion_neck = dict(
        type='necks.dilation_spconv.DilationSpconv',
        gather_keys=['fused_feat'],
        scatter_keys=['fused_neck_feat'],
        data_info=data_info,
        d=2,
        convs=dict(p4=dict(kernels=[3, 3, 3], in_dim=128, out_dim=128))
    ),


    bev_head = dict(
        type='heads.bev.BEV',
        gather_keys=['fused_neck_feat'],
        scatter_keys=['bev_out'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d'],
        data_info=data_info,
        stride=4,
        annealing_step=50,
        in_dim=128,
        sampling=dict(annealing=False, topk=False),

    ),

    detection_head = dict(
        type='heads.det_center_sparse.DetCenterSparse',
        gather_keys=['fused_neck_feat'],
        scatter_keys=['detection_out'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d'],
        data_info=data_info,
        input_channels=128,
        shared_conv_channel=128,
        get_predictions=True,
        stride=4,
        cls_head_cfg=dict(name='UnitedClsHead'),
        reg_head_cfg=dict(name='UnitedRegHead', combine_channels=True, sigmoid_keys=['scr']),
        class_names_each_head=[['vehicle.car']],
        reg_channels=['box:6', 'dir:8', 'scr:4'],
        target_assigner=dict(
            data_info=data_info,
            meter_per_pixel=0.8,
            stride=4,
            detection_benchmark='Car',
            assigners=OrderedDict(
                points_centerness=dict(min_radius=1.6, batch_dict_key='det_center_head', pos_neg_ratio=2),
                encode_box=dict(_target_='modules.utils.box_coder.CenterBoxCoder', center_thresh=0.5)),
        ),
        loss_cfg=dict(center=dict(_target_='modules.losses.edl.edl_mse_loss', args=dict(annealing_step=5000)),
                      reg=dict(_target_='modules.losses.common.weighted_smooth_l1_loss')),
    ),
)