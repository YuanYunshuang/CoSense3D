from collections import OrderedDict

pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(),
    LoadAnnotations=dict(load2d=False, load3d_local=False),
)

inference_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(),
)

data_manager = dict(
    train=dict(
        aug=dict(
            rot_range=[-1.57, 1.57],
            flip='xy',
            scale_ratio_range=[0.95, 1.05],
        )
    ),
    test=dict()
)

