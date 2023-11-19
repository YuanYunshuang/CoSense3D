from collections import OrderedDict

pipeline_cpu = OrderedDict(
    LoadMultiViewImg=dict(),
    LoadAnnotations=dict(with_velocity=True, load3d_local=False),
    ResizeImage=dict(img_size=(512, 512)),
    Format2D=dict(),
)

inference_pipeline_cpu = OrderedDict(
    LoadMultiViewImg=dict(),
    ResizeImage=dict(img_size=(512, 512)),
    Format2D=dict(),
)

data_manager = dict(
    train=dict(),
    test=dict()
)

