import torch
from torch import nn

from cosense3d.modules import BaseModule





class KeypointsComposer(BaseModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

