from collections import OrderedDict


point_cloud_range = [-144, -41.6, -3.0, 144, 41.6, 1.0]
point_cloud_range_test = [-140.8, -38.4, -3.0, 140.8, 38.4, 1.0]

pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time']),
    LoadAnnotations=dict(load3d_global=True, load3d_local=True,
                         load_global_time=True, with_velocity=True, min_num_pts=3),
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
        # aug=dict(
        #     rot_range=[-1.57, 1.57],
        #     flip='xy',
        #     scale_ratio_range=[0.95, 1.05],
        # ),
    )
)