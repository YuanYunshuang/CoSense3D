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