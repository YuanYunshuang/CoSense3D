from .focal_loss import *
from .l1_loss import *
from .iou_loss import *
from .edl import *


def build_loss(type, **kwargs):
    return globals()[type](**kwargs)



