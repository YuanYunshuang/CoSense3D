import functools
from cosense3d.model.utils import *
from cosense3d.model.losses.edl import edl_mse_loss


class SparseConv(nn.Module):
    QMODE = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE

    def __init__(self, cfgs):
        super(SparseConv, self).__init__()
        for name, value in cfgs.items():
            if name not in ["model", "__class__"]:
                setattr(self, name, value)
        self.d = getattr(self, 'd', 3)

        self.enc_mlp = linear_layers([self.in_dim, 16, 32])

        self.conv_input = minkconv_conv_block(32, 16, 3, 1, 3, 0.1)
        self.conv1 = minkconv_conv_block(16, 16, 3, 1, 3, 0.1)
        self.conv2 = get_conv_block([16, 32, 32])
        self.conv3 = get_conv_block([32, 64, 64])
        self.conv4 = get_conv_block([64, 64, 64])
        self.conv_out = minkconv_conv_block(64, 128, (1, 1, 3), (1, 1, 2), 3, 0.1)

    def forward(self, batch_dict):
        batch_dict = prepare_input_data(batch_dict, self.QMODE)
        x = batch_dict['in_data']
        x1, norm_points_p1, points_p1, count_p1, pos_embs = voxelize_with_centroids(x, self.enc_mlp)

        # convs
        p1 = self.conv_input(x1)
        p1 = self.conv1(p1)
        p2 = self.conv2(p1)
        p4 = self.conv3(p2)
        p8 = self.conv4(p4)
        p8 = self.conv_out(p8)
        vars_local = locals()
        batch_dict['backbone'] = {k: vars_local[k] for k in self.cache}





