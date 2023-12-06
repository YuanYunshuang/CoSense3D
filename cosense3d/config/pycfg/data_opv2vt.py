from collections import OrderedDict

train_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'time']),
    LoadAnnotations=dict(load3d_local=True, load3d_global=True),
)

test_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'time']),
    LoadAnnotations=dict(load3d_local=True, load3d_global=True),
)

data_manager = dict(
    train=dict(
        aug=dict(
            rot_range=[-1.57, 1.57],
            flip='xy',
            scale_ratio_range=[0.95, 1.05],
        ),
        pre_process=['remove_empty_boxes']
    ),
    test=dict()
)

