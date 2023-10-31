from collections import OrderedDict

# point_cloud_range = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
# point_cloud_range_enlarged = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
point_cloud_range = [-144, -41.6, -5.0, 144, 41.6, 3.0]
point_cloud_range_test = [-140.8, -38.4, -3.0, 140.8, 38.4, 3.0]
voxel_size = [0.2, 0.2, 8]
data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)

shared_modules = OrderedDict(
    pts_backbone = dict(
        type='backbone_3d.mink_unet.MinkUnet',
        voxel_size=voxel_size,
        d=2,
        cache_strides=[4],
        in_dim=4,
        stride=4,
        floor_height=point_cloud_range[2]
    ),

    fusion = dict(
        type='submodules.naive_fusion.NaiveFusion',
        data_info=data_info,
        stride=2,
        dim=128
    ),

    coor_dilation = dict(
        type='submodules.dilation_spconv.DilationSpconv',
        data_info=data_info,
        convs=dict(p2=dict(kernels=[5, 5, 3], in_dim=128, out_dim=128))
    ),


    bev_head = dict(
        type='heads.bev.Bev',
        data_info=data_info,
        stride=2,
        annealing_step=50,
        in_dim=128,
        sampling=dict(annealing=False, topk=False),

    ),

    detection_head = dict(
        type='heads.det_center_sparse.DetCenterSparse',
        data_info=data_info,
        input_channels=128,
        shared_conv_channel=128,
        get_predictions=True,
        stride=2,
        cls_head_cfg=dict(name='UnitedClsHead'),
        reg_head_cfg=dict(name='UnitedRegHead', combine_channels=True, sigmoid_keys=['scr']),
        class_names_each_head=[['vehicle.car']],
        reg_channels=['box:6', 'dir:8', 'scr:4'],
        target_assigner=dict(
            data_info=data_info,
            meter_per_pixel=0.4,
            stride=2,
            detection_benchmark='Car',
            assigners=OrderedDict(
                points_centerness=dict(min_radius=1.6, batch_dict_key='det_center_head', pos_neg_ratio=2),
                encode_box=dict(_target_='model.utils.box_coder.CenterBoxCoder', center_thresh=0.5)),
        ),
        loss_cfg=dict(center=dict(_target_='model.losses.edl.edl_mse_loss', args=dict(annealing_step=5000)),
                      reg=dict(_target_='model.losses.common.weighted_smooth_l1_loss')),
    ),
)