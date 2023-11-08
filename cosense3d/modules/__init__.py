import torch
from torch import nn
from typing import List, Dict, Optional
import importlib

from cosense3d.modules.utils.common import cat_coor_with_idx
from cosense3d.modules.utils.me_utils import ME


def build_module(module_cfg):
    module_full_path=module_cfg['type']
    package, module_name = module_full_path.rsplit('.', 1)
    module = importlib.import_module(f'cosense3d.modules.{package}')
    cls_obj = getattr(module, module_name, None)
    assert cls_obj is not None, f'Class \'{module_name}\' not found.'
    inst = cls_obj(**module_cfg)
    return inst


class BaseModule(nn.Module):
    def __init__(self, gather_keys, scatter_keys, gt_keys=[], **kwargs):
        super(BaseModule, self).__init__()
        self.gather_keys = gather_keys
        self.scatter_keys = scatter_keys
        self.gt_keys = gt_keys

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        """This must be implemented in head module."""
        # TODO: Create Head base module.
        pass

    def prepare_vis_data(self):
        pass

    def format_input(self, input: List):
        pass

    def format_output(self, output, B):
        pass

    def data_from_list(self, input, key=None, pad_idx=False):
        if key is not None:
            data = [x[key] for x in input]
        else:
            data = input
        if pad_idx:
            return cat_coor_with_idx(data)
        else:
            return torch.cat(data, dim=0)

    def compose_imgs(self, img_list):
        imgs = [img for x in img_list for img in x]
        return torch.stack(imgs, dim=0)

    def compose_stensor(self, stensor_list, stride):
        coor = [stensor[f'p{stride}']['coor'] for stensor in stensor_list]
        coor = cat_coor_with_idx(coor)
        feat = [stensor[f'p{stride}']['feat'] for stensor in stensor_list]
        feat = torch.cat(feat, dim=0)
        return coor, feat

    def decompose_stensor(self, res, N):
        # decompose batch
        for k, v in res.items():
            if isinstance(v, ME.SparseTensor):
                coor, feat = v.decomposed_coordinates_and_features
            elif isinstance(v, dict):
                coor, feat = [], []
                for i in range(N):
                    mask = v['coor'][:, 0] == i
                    coor.append(v['coor'][mask, 1:])
                    feat.append(v['feat'][mask])
            else:
                raise NotImplementedError
            res[k] = {'coor': coor, 'feat': feat}

        # compose result list
        res_list = self.compose_result_list(res, N)
        return res_list

    def compose_result_list(self, res, N):
        """

        :param res: dict(k:list)
        :param N:
        :return:
        """
        keys = res.keys()
        res_list = []
        for i in range(N):
            cur_res = {}
            for k, v in res.items():
                if isinstance(v, dict):
                    cur_res[k] = {
                        'coor': v['coor'][i],
                        'feat': v['feat'][i]
                    }
                elif isinstance(v, list):
                    cur_res[k] = v[i]
                else:
                    raise NotImplementedError
            res_list.append(cur_res)
        return res_list

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gather_keys={self.gather_keys}, '
        repr_str += f'scatter_keys={self.scatter_keys})'
        return repr_str