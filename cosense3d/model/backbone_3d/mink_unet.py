import torch
from torch import nn
from cosense3d.model.utils.common import *
from cosense3d.model.utils.me_utils import *


class MinkUnet(nn.Module):
    QMODE = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    def __init__(self,
                 voxel_size,
                 stride,
                 d=3,
                 cache_strides=None,
                 floor_height=0,
                 **kwargs):
        super(MinkUnet, self).__init__()
        for name, value in kwargs.items():
            if name not in ["model", "__class__"]:
                setattr(self, name, value)
        self.voxel_size = voxel_size
        self.stride = stride
        self.floor_height = floor_height
        self.d = d
        if cache_strides is None:
            self.cache_strides = [stride]
            self.max_resolution = stride
        else:
            self.max_resolution = min(cache_strides)
            self.cache_strides = cache_strides
        self.enc_mlp = linear_layers([self.in_dim * 2, 16, 32])
        kernel = [3,] * min(self.d, 3)
        if self.d == 4:
            kernel = kernel + [1,]
        self.conv1 = minkconv_conv_block(32, 32, kernel, 1, self.d, 0.1)
        self.conv2 = get_conv_block([32, 32, 32], kernel, d=self.d)
        self.conv3 = get_conv_block([32, 64, 64], kernel, d=self.d)
        self.conv4 = get_conv_block([64, 128, 128], kernel, d=self.d)

        if self.max_resolution <= 4:
            self.trconv4 = get_conv_block([128, 64, 64], kernel, d=self.d, tr=True)
        if self.max_resolution <= 2:
            self.trconv3 = get_conv_block([128, 64, 64], kernel, d=self.d, tr=True)
        if self.max_resolution <= 1:
            self.trconv2 = get_conv_block([96, 64, 32], kernel, d=self.d, tr=True)
            self.out_layer = minkconv_conv_block(64, 32, kernel, 1, self.d, 0.1,
                                                 'ReLU', norm_before=True)

    def forward(self, points_list, pad_idx=False):
        if pad_idx:
            points_list = [torch.cat([torch.ones_like(points[:, :1]) * i,
                                      points], dim=-1) for i, points in enumerate(points_list)]
        x = prepare_input_data(points_list, self.voxel_size, self.QMODE, self.floor_height, self.d)
        x1, norm_points_p1, points_p1, count_p1, pos_embs = voxelize_with_centroids(x, self.enc_mlp)

        # convs
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        x4 = self.conv3(x2)
        p8 = self.conv4(x4)

        # transposed convs
        if self.max_resolution <= 4:
            p4 = self.trconv4(p8)
        if self.max_resolution <= 2:
            p2 = self.trconv3(ME.cat(x4, p4))
        if self.max_resolution <= 1:
            p1 = self.trconv2(ME.cat(x2, p2))
            p1 = self.out_layer(ME.cat(x1, p1))
        if self.max_resolution == 0:
            p0 = {'coor': torch.cat(points_list, dim=0), 'feat': devoxelize_with_centroids(p1, x, pos_embs)}

        vars = locals()
        return {f'p{k}': vars[f'p{k}'] for k in self.cache_strides}





