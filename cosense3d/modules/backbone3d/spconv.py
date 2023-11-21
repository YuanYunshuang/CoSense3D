from functools import partial

import spconv
import torch.nn as nn

from spconv.pytorch import  SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
from cosense3d.modules import BaseModule, plugin


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class Spconv(BaseModule):
    def __init__(self,
                 in_channels,
                 voxel_generator,
                 voxel_encoder,
                 num_point_features=128,
                 bev_neck=None,
                 bev_compressor=None,
                 **kwargs):
        super(Spconv, self).__init__(**kwargs)
        self.num_point_features = num_point_features
        self.voxel_generator = plugin.build_plugin_module(voxel_generator)
        self.voxel_encoder = plugin.build_plugin_module(voxel_encoder)
        self.grid_size = self.voxel_generator.grid_size
        if bev_neck is not None:
            self.bev_neck = plugin.build_plugin_module(bev_neck)
        if bev_compressor is not None:
            self.bev_compressor = plugin.build_plugin_module(bev_compressor)

    def _init_layers(self, in_channels):
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = self.grid_size[::-1] + [1, 0, 0]

        self.conv_input = SparseSequential(
            SubMConv3d(in_channels, 16, 3,
                       padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = SparseSequential(
            block(16, 16, 3,
                  norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3,
                  norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3,
                  norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3,
                  norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        self.conv_out = SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            SparseConv3d(64, self.num_point_features, (3, 1, 1),
                         stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'),
            norm_fn(self.num_point_features),
            nn.ReLU(),
        )

        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, points: list, **kwargs):
        B = len(points)
        voxels, coords, num_points = self.voxel_generator(points)
        coords = self.cat_data_from_list(coords, pad_idx=True)
        voxels = self.cat_data_from_list(voxels)
        num_points = self.cat_data_from_list(num_points)

        input_sp_tensor = SparseConvTensor(
            features=voxels,
            indices=coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=B
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        bev_feat = self.to_dense(out)

        if hasattr(self, 'bev_neck'):
            bev_feat = self.bev_neck(bev_feat)
        if hasattr(self, 'bev_compressor'):
            bev_feat = self.bev_compressor(bev_feat)

        return {self.scatter_keys[0]: bev_feat}

    def to_dense(self, stensor):
        spatial_features = stensor.dense()
        N, C, D, H, W = spatial_features.shape
        bev_featrues = spatial_features.view(N, C * D, H, W)
        return bev_featrues



