from torch import nn

from cosense3d import model


class SharedModules(nn.ModuleDict):
    def __init__(self, cfg):
        super().__init__()