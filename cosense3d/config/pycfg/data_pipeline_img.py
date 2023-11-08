from collections import OrderedDict

pipeline_cpu = OrderedDict(
    LoadMultiViewImg=dict(),
    LoadAnnotations=dict(with_velocity=True, load3d_local=False),
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

inference_pipeline_cpu = OrderedDict(
    LoadMultiViewImg=dict(),
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

