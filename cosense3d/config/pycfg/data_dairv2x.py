from collections import OrderedDict

train_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time'], time_offset=1.6261626 * 1e9),
    LoadAnnotations=dict(load3d_global=True),
)

test_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time'], time_offset=1.6261626 * 1e9),
    LoadAnnotations=dict(load3d_global=True),
)

data_manager = dict(
    train=dict(
        aug=dict(
            # rot_range=[-1.57, 1.57],
            # flip='xy',
            # scale_ratio_range=[0.95, 1.05],
        ),
        pre_process=['remove_empty_boxes']
    ),
    test=dict()
)

