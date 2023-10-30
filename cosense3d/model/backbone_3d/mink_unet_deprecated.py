import functools
from cosense3d.model.utils import *
from cosense3d.model.losses.edl import edl_mse_loss


class MinkUnet(nn.Module):
    QMODE = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    def __init__(self, cfgs):
        super(MinkUnet, self).__init__()
        for name, value in cfgs.items():
            if name not in ["model", "__class__"]:
                setattr(self, name, value)
        self.d = getattr(self, 'd', 3)
        self.max_resolution = min([int(p[1:]) for p in self.cache])
        self.enc_mlp = linear_layers([self.in_dim, 16, 32])
        kernel = [3, 3, 3]
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
            self.out_layer = minkconv_conv_block(64, 32, kernel, 1, 3, 0.1,
                                                 'ReLU', norm_before=True)

    def forward(self, batch_dict):
        # from cosense3d.utils.vislib import draw_points_boxes_plt
        # points = batch_dict['pcds']
        # points = points[points[:, 0] < batch_dict['num_cav'][0]][:, 1:4].cpu().numpy()
        # objects = batch_dict['objects']
        # objects = objects[objects[:, 0] == 0][:, [3, 4, 5, 6, 7, 8, 11]].cpu().numpy()
        #
        # draw_points_boxes_plt(
        #     pc_range=[-140.8, -38.4, -5, 140.8, 38.4, 3],
        #     points=points,
        #     boxes_gt=objects,
        #     filename='/home/yuan/Downloads/tmp.png'
        # )

        batch_dict = prepare_input_data(batch_dict, self.QMODE)
        x = batch_dict['in_data']
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
            p0 = devoxelize_with_centroids(p1, x, pos_embs)

        vars = locals()
        batch_dict['backbone'] = {k: vars[k] for k in self.cache}





