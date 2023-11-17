import torch
from torch import nn
import torch.nn.functional as F


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class RPN(nn.Module):
    def __init__(self, anchor_num=2):
        super(RPN, self).__init__()
        self.anchor_num = anchor_num

        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0),
                                      nn.BatchNorm2d(256))


    def forward(self, x):
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        x = torch.cat((x_0, x_1, x_2), 1)
        return x