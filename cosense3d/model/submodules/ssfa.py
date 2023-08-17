import torch
from torch import nn
from cosense3d.model.utils import get_conv2d_layers, sparse_to_dense


class Ssfa(nn.Module):
    """Module SSFA"""
    def __init__(self, cfgs):
        super(Ssfa, self).__init__()
        self.stride = cfgs['stride']
        self.num_input_features = cfgs['feature_num']  # 128
        self.voxel_size = cfgs['data_info']['voxel_size']
        self.det_r = cfgs['data_info']['det_r']

        seq = [nn.ZeroPad2d(1)] + get_conv2d_layers(
            'Conv2d', 128, 128, n_layers=3, kernel_size=[3, 3, 3],
            stride=[1, 1, 1], padding=[0, 1, 1], sequential=False
        )
        self.bottom_up_block_0 = nn.Sequential(*seq)
        self.bottom_up_block_1 = get_conv2d_layers(
            'Conv2d', 128, 256, n_layers=3, kernel_size=[3, 3, 3],
            stride=[2, 1, 1], padding=[1, 1, 1]
        )

        self.trans_0 = get_conv2d_layers(
            'Conv2d', 128, 128, n_layers=1, kernel_size=[1], stride=[1], padding=[0]
        )
        self.trans_1 = get_conv2d_layers(
            'Conv2d', 256, 256, n_layers=1, kernel_size=[1], stride=[1], padding=[0]
        )

        self.deconv_block_0 = get_conv2d_layers(
            'ConvTranspose2d', 256, 128,
            n_layers=1, kernel_size=[3], stride=[2],
            padding=[1], output_padding=[1]
        )
        self.deconv_block_1 = get_conv2d_layers(
            'ConvTranspose2d', 256, 128,
            n_layers=1, kernel_size=[3], stride=[2],
            padding=[1], output_padding=[1]
        )

        self.conv_0 = get_conv2d_layers(
            'Conv2d', 128, 128, n_layers=1,
            kernel_size=[3], stride=[1], padding=[1]
        )
        self.conv_1 = get_conv2d_layers(
            'Conv2d', 128, 128, n_layers=1,
            kernel_size=[3], stride=[1], padding=[1]
        )

        self.w_0 = get_conv2d_layers(
            'Conv2d', 128, 1, n_layers=1,
            kernel_size=[1], stride=[1], padding=[0],
            relu_last=False
        )
        self.w_1 = get_conv2d_layers(
            'Conv2d', 128, 1, n_layers=1,
            kernel_size=[1], stride=[1], padding=[0],
            relu_last=False
        )

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        x = batch_dict['compression'][f'p{self.stride}']
        x_0 = self.bottom_up_block_0(x)
        x_1 = self.bottom_up_block_1(x_0)
        x_trans_0 = self.trans_0(x_0)
        x_trans_1 = self.trans_1(x_1)
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0
        x_middle_1 = self.deconv_block_1(x_trans_1)
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)
        x_output = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:, :, :]

        batch_dict['ssfa'] = {
            f'p{self.stride}': x_output.contiguous()
        }