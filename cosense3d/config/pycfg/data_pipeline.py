from collections import OrderedDict

train_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(),
    LoadMultiViewImg=dict(),
    LoadAnnotations=dict(with_velocity=True),
    ResizeCropFlipRotImage=dict(
        training=True,
        data_aug_conf=dict(
            resize_lim=[0.55, 0.7],
            final_dim=[256, 512],
            bot_pct_lim=[0.0, 0.0],
            rot_lim=[0.0, 0.0],
            H=600,
            W=800,
            rand_flip=True,
        )
    )
)

train_pipeline_gpu = OrderedDict(
    FormatSequenceData=dict(),
    ProjectToEgo=dict(),
    GlobalRotScaleTrans=dict(),
)

test_pipeline_gpu = OrderedDict(

)