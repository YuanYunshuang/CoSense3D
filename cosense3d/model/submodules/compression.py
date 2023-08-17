import torch
from torch import nn
from cosense3d.model.utils import minkconv_conv_block
from cosense3d.model.utils.me_utils import update_me_essentials


class Compression(nn.Module):
    def __init__(self, cfgs):
        super(Compression, self).__init__()
        # self.voxel_size = cfgs['data_info']['voxel_size']
        # self.det_r = cfgs['data_info']['det_r']
        update_me_essentials(self, cfgs['data_info'])
        self.d = len(self.voxel_size)

        self.dense = cfgs['dense']
        self.stride_size_dict = {}
        for plane in cfgs['planes']:
            k = list(plane.keys())[0]
            v = plane[k]
            self.stride_size_dict[int(k[1])] = self.grid_size(int(k[1]))
            layers = []
            steps = v['steps']
            channels = v['channels']
            for i, s in enumerate(steps):
                step = [1] * self.d
                step[2] = s
                layers.append(
                    minkconv_conv_block(channels[i], channels[i+1],
                                        step, step, self.d, 0.1)
                )
            layers = nn.Sequential(*layers)
            setattr(self, f'{k}_compression', layers)

    def forward(self, batch_dict):
        stensors = batch_dict['backbone']
        out = {}

        for stride in self.stride_size_dict.keys():
            out_tensor = getattr(self, f'p{stride}_compression')(stensors[f'p{stride}'])
            assert len(out_tensor.C[:, 3].unique()) == 1, \
                (f"height is not fully compressed. "
                 f"Current z coords: {','.join([str(x.item()) for x in out_tensor.C[:, 3].unique()])}")
            if self.dense:
                out_tensor = self.to_dense(out_tensor).permute(0, 3, 1, 2)
            out[f'p{stride}'] = {'coor': out_tensor.C, 'feat': out_tensor.F}

        batch_dict['compression'] = out

    def to_dense(self, stensor):
        mask, indices = self.valid_coords(stensor)
        b = int(stensor.C[:, 0].max()) + 1
        d = stensor.F.shape[1]
        features = stensor.F[mask].view(-1, d)
        s = self.stride_size_dict[stensor.tensor_stride[0]]
        dtensor = features.new_zeros((b, s[0], s[1], d))
        dtensor[indices[0], indices[1], indices[2]] = features
        return dtensor

    def valid_coords(self, stensor):
        stride = stensor.tensor_stride
        s = self.stride_size_dict[stride[0]]
        # remove voxels that are outside range
        xi = torch.div(stensor.C[:, 1], stride[0], rounding_mode='floor') + s[0] / 2
        yi = torch.div(stensor.C[:, 2], stride[1], rounding_mode='floor') + s[1] / 2

        mask = (xi >= 0) * (xi < s[0]) * (yi >= 0) * (yi < s[1])
        indices = (stensor.C[:, 0][mask].long(),
                   xi[mask].long(),
                   yi[mask].long()
                   )
        # if the backbone uses 4d convs, last dim is time
        if stensor.C.shape[1] == 5:
            ti = stensor.C[:, 4]
            mask = mask * (ti >= 0) * (ti < self.seq_len)
            indices = indices + ti[mask].long()
        return mask, indices

    def grid_size(self, stride):
        x_range = self.lidar_range[3] - self.lidar_range[0]
        y_range = self.lidar_range[4] - self.lidar_range[1]
        x_size = int(x_range / self.voxel_size[0]) // stride
        y_size = int(y_range / self.voxel_size[1]) // stride
        return (x_size, y_size)