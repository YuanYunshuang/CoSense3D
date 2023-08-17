import functools
from torch import nn
from cosense3d.model.utils import minkconv_conv_block, ME
from cosense3d.model.utils.me_utils import mink_coor_limit


class DilationSpconv(nn.Module):
    def __init__(self, cfgs):
        super(DilationSpconv, self).__init__()
        self.cfgs = cfgs
        self.det_r = cfgs['data_info'].get('det_r', False)
        self.lidar_range = cfgs['data_info'].get('lidar_range', False)
        self.voxel_size = cfgs['data_info']['voxel_size']
        self.convs = []
        for k, conv_args in cfgs['convs'].items():
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

    def forward(self, batch_dict):
        out_dict = {}
        for k in self.convs:
            stride = int(k[1])
            stensor3d = batch_dict[self.cfgs['feature_src']][k]
            coor = stensor3d['coor'][:, :3]
            feat = stensor3d['feat']
            stensor2d = ME.SparseTensor(
                coordinates=coor.contiguous(),
                features=feat,
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
        batch_dict['dilation_spconv'] = out_dict

    def get_conv_layer(self, args):
        minkconv_layer = functools.partial(
            minkconv_conv_block, d=2, bn_momentum=0.1,
        )
        in_dim = args['in_dim']
        out_dim = args['out_dim']
        layers = [minkconv_layer(in_dim, out_dim, args['kernels'][0], 1,
                                 expand_coordinates=args['expand_coordinates'])]
        for ks in args['kernels'][1:]:
            layers.append(minkconv_layer(out_dim, out_dim, ks, 1,
                                         expand_coordinates=args['expand_coordinates']))
        return nn.Sequential(*layers)