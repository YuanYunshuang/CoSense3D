from collections import OrderedDict

pipeline_cpu = OrderedDict(
    LoadMultiViewImg=dict(bgr2rgb=True),
    LoadOPV2VBevMaps=dict(keys=['bev_visibility_corp']),
    LoadAnnotations=dict(load_cam_param=True),
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

