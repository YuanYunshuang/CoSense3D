from torch import nn

from cosense3d.model import build_module



class SharedModules(nn.ModuleDict):
    def __init__(self, cfg):
        module_dict = {}
        self.module_keys = []
        for k, v in cfg.items():
            module_dict[k] = build_module(v)
            self.module_keys.append(k)

        super().__init__(module_dict)

