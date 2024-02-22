from cosense3d.config.pycfg.base.opv2vt import *
from cosense3d.config.pycfg.base.hooks import *
from cosense3d.config.pycfg.nets.streamLTS_attnfusion import *
from cosense3d.utils.train_utils import get_gpu_architecture

gpu_arc = get_gpu_architecture()
if gpu_arc >= 75:
    shared_modules = get_shared_modules(point_cloud_range, 'MultiheadFlashAttention')
else:
    shared_modules = get_shared_modules(point_cloud_range, 'MultiheadAttention')
test_hooks = get_test_nms_eval_hooks(point_cloud_range_test)
plots = [get_detection_plot(point_cloud_range_test)]