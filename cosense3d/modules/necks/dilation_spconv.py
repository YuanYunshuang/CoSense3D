import functools
import torch

from cosense3d.modules import BaseModule, nn
from cosense3d.modules.utils.me_utils import mink_coor_limit, minkconv_conv_block, ME


class DilationSpconv(BaseModule):
    def __init__(self, data_info, convs, d=2, **kwargs):
        super(DilationSpconv, self).__init__(**kwargs)
        self.det_r = data_info.get('det_r', False)
        self.lidar_range = data_info.get('lidar_range', False)
        self.voxel_size = data_info['voxel_size']
        self.d = d
        self.convs = []
        for k, conv_args in convs.items():
            self.convs.append(k)
            setattr(self, f'convs_{k}', self.get_conv_layer(conv_args))
            stride = int(k[1])

            if self.det_r:
                lr = [-self.det_r, -self.det_r, 0, self.det_r, self.det_r, 0]
            elif self.lidar_range:
                lr = self.lidar_range
            else:
                raise NotImplementedError
            setattr(self, f'mink_xylim_{k}', mink_coor_limit(lr, self.voxel_size, stride))  # relevant to ME

    def forward(self, stensor_list):
        out_dict = {}
        for k in self.convs:
            stride = int(k[1])
            coor_cat = []
            feat_cat = []
            for i, stensor_dict in enumerate(stensor_list):
                stensor = stensor_dict[k]
                coor = stensor['coor'][:, :self.d]
                coor_cat.append(torch.cat([torch.ones_like(coor[:, :1]) * i, coor], dim=-1))
                feat_cat.append(stensor['feat'])

            coor_cat = torch.cat(coor_cat, dim=0)
            feat_cat = torch.cat(feat_cat, dim=0)
            stensor2d = ME.SparseTensor(
                coordinates=coor_cat.contiguous(),
                features=feat_cat,
                tensor_stride=[stride] * 2
            )

            stensor2d = getattr(self, f'convs_{k}')(stensor2d)
            # after coordinate expansion, some coordinates will exceed the maximum detection
            # range, therefore they are removed here.
            xylim = getattr(self, f'mink_xylim_{k}')
            mask = (stensor2d.C[:, 1] > xylim[0]) & (stensor2d.C[:, 1] <= xylim[1]) & \
                   (stensor2d.C[:, 2] > xylim[2]) & (stensor2d.C[:, 2] <= xylim[3])

            coor = stensor2d.C[mask]
            feat = stensor2d.F[mask]

            out_dict[k] = {
                'coor': coor,
                'feat': feat
            }
        out_list = self.to_batch_list(out_dict, len(stensor_list))
        return {self.scatter_keys[0]: out_list}

    def to_batch_list(self, out_dict, B):
        out_list = []
        for b in range(B):
            tmp = {}
            for k, out in out_dict.items():
                mask = out['coor'][:, 0] == b
                tmp[k] = {
                    'coor': out['coor'][mask, 1:],
                    'feat': out['feat'][mask]
                }
            out_list.append(tmp)
        return out_list

    def get_conv_layer(self, args):
        minkconv_layer = functools.partial(
            minkconv_conv_block, d=self.d, bn_momentum=0.1,
        )
        in_dim = args['in_dim']
        out_dim = args['out_dim']
        layers = [minkconv_layer(in_dim, out_dim, args['kernels'][0], 1,
                                 expand_coordinates=True)]
        for ks in args['kernels'][1:]:
            layers.append(minkconv_layer(out_dim, out_dim, ks, 1,
                                         expand_coordinates=True))
        return nn.Sequential(*layers)