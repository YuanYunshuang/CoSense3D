from collections import OrderedDict

point_cloud_range = [-102.4, -41.6, -3.0, 102.4, 41.6, 1.0]
point_cloud_range_test = [-100, -38.4, -3.0, 100, 38.4, 1.0]

pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time'], time_offset=1.6261*1e9),
    LoadAnnotations=dict(load3d_global=True, load3d_local=True,
                         with_velocity=True, min_num_pts=3, load_global_time=True),
)

inference_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time']),
)

data_manager = dict(
    train=dict(
        aug=dict(
            rot_range=[-0.785, 0.785],
            flip='xy',
            scale_ratio_range=[0.95, 1.05],
        ),
        pre_process=['remove_local_empty_boxes',
                     'remove_global_empty_boxes']
    ),
    test=dict(
        aug=dict()
    )
)


def get_dairv2xt_cfg(voxel_size, seq_len):
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return dict(
        name='dairv2xt',
        dataset='temporal_cosense',
        meta_path='dairv2xt',
        data_path='dairv2xt',
        enable_split_sub_folder=True,
        DetectionBenchmark='Car',
        data_info=data_info,
        lidar_range=point_cloud_range,
        voxel_size=voxel_size,
        batch_size_train=4,
        batch_size_test=1,
        n_workers=4,
        max_num_cavs=7,
        com_range=200,
        seq_len=seq_len,
        train_pipeline=pipeline_cpu,
        test_pipeline=pipeline_cpu,
    )