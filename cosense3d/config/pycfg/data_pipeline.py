from collections import OrderedDict

train_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(),
    LoadMultiViewImg=dict(),
    LoadAnnotations=dict(with_velocity=True),
    ResizeCropFlipRotImage=dict(
        training=True,
        data_aug_conf=dict(
            resize_lim=[0.8, 1.0],
            final_dim=[384, 768],
            bot_pct_lim=[0.0, 0.0],
            rot_lim=[0.0, 0.0],
            H=600,
            W=800,
            rand_flip=True,
        )
    ),
    Format2D=dict(),
)


test_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(),
    LoadMultiViewImg=dict(),
    LoadAnnotations=dict(with_velocity=True),
    ResizeCropFlipRotImage=dict(
        training=False,
        data_aug_conf=dict(
            resize_lim=[0.55, 0.7],
            final_dim=[256, 512],
            bot_pct_lim=[0.0, 0.0],
            rot_lim=[0.0, 0.0],
            H=600,
            W=800,
            rand_flip=False,
        )
    ),
    Format2D=dict(),
)


data_manager = dict(
    train=dict(
        aug=dict(
            rot_range=[-1.57, 1.57],
            flip='xy',
            scale_ratio_range=[0.95, 1.05],
        ),
    ),
    test=dict()
)


output_viewer = [
    dict(title='BEVSparseCanvas', width=10, height=4, nrows=1, ncols=1),
    dict(title='DetectionCanvas', width=10, height=4, nrows=1, ncols=1)
]