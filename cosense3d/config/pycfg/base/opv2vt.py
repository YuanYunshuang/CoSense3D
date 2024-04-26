from collections import OrderedDict

point_cloud_range = [-144, -41.6, -3.0, 144, 41.6, 1.0]
point_cloud_range_test = [-140.8, -38.4, -3.0, 140.8, 38.4, 1.0]
global_ref_time = 0.05

pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time']),
    LoadAnnotations=dict(load3d_global=True, load3d_local=True,
                         load_global_time=True, with_velocity=True, min_num_pts=0),
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
    )
)


def get_opv2vt_cfg(seq_len, voxel_size, latency=0):
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return dict(
        name='opv2vt',
        dataset='temporal_cosense',
        meta_path='opv2vt',
        data_path='opv2vt',
        enable_split_sub_folder=False,
        DetectionBenchmark='Car',
        data_info=data_info,
        lidar_range=point_cloud_range,
        voxel_size=voxel_size,
        batch_size_train=4,
        batch_size_test=1,
        n_workers=4,
        max_num_cavs=7,
        com_range=70,
        latency=latency,
        seq_len=seq_len,
        train_pipeline=pipeline_cpu,
        test_pipeline=pipeline_cpu,
    )


seq4_pillar04 = get_opv2vt_cfg(4, [0.4, 0.4, 4])
seq4_vox04 = get_opv2vt_cfg(4, [0.4, 0.4, 0.4])
seq4_vox04_lat1 = get_opv2vt_cfg(4, [0.4, 0.4, 0.4], 1)
seq4_vox01 = get_opv2vt_cfg(4, [0.1, 0.1, 0.1])