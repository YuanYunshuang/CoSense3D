from collections import OrderedDict

pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'time']),
    LoadAnnotations=dict(load3d_global=True, load3d_local=True,
                         load3d_pred=True, with_velocity=True, min_num_pts=3),
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
