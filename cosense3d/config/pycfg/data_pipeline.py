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
    ),
    Format2D=dict(),
)

train_data_manager = dict(
    aug=dict(
        rot_range=[-1.57, 1.57],
        scale_ratio_range=[0.95, 1.05],)
)

train_pipeline_gpu = OrderedDict(
    FormatSequenceData=dict(),
    ProjectToEgo=dict(),
    GlobalRotScaleTrans=dict(),
)

test_pipeline_gpu = OrderedDict(

)