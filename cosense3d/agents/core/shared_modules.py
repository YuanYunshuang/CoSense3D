from torch import nn

from cosense3d.modules import build_module



class SharedModules:
    def __init__(self, cfg):
        module_dict = {}
        self.module_keys = []
        for k, v in cfg.items():
            module_dict[k] = build_module(v)
            self.module_keys.append(k)

        self.modules = nn.ModuleDict(module_dict)

