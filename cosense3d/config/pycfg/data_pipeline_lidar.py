from collections import OrderedDict

train_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(),
    LoadAnnotations=dict(load2d=False, with_velocity=False),
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