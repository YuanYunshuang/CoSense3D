import torch
from torch import nn
from typing import List, Dict

from cosense3d.modules.utils.common import cat_coor_with_idx


class BaseModule(nn.Module):
    def __init__(self, gather_keys, scatter_keys, **kwargs):
        super(BaseModule, self).__init__()
        self.gather_keys = gather_keys
        self.scatter_keys = scatter_keys

    def format_input(self, input: List[Dict]):
        pass

    def format_output(self, output: Dict, B):
        pass

    def data_from_list_by_key(self, input, key=None, pad_idx=False):
        data = [x[key] for x in input]
        if pad_idx:
            return cat_coor_with_idx(data)
        else:
            return torch.cat(data, dim=0)

    def coor_feat_to_list(self, coor, feat, key):
        pass

    def __repr__(self):
        def __repr__(self):
            repr_str = self.__class__.__name__
            repr_str += f'(gather_keys={self.gather_keys}, '
            repr_str += f'scatter_keys={self.scatter_keys})'
            return repr_str