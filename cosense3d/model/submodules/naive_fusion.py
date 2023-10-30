import torch
from torch import nn
from cosense3d.model.utils.me_utils import update_me_essentials


class NaiveFusion(nn.Module):
    def __init__(self, cfgs=None, **kwargs):
        super(NaiveFusion, self).__init__()
        # self.voxel_size = cfgs['data_info']['voxel_size']
        # self.det_r = cfgs['data_info']['det_r']
        if cfgs is None:
            cfgs = kwargs  # for old version
        update_me_essentials(self, cfgs['data_info'], stride=cfgs['stride'])
        self.d = len(self.voxel_size)
        self.feature_scr = cfgs.get('feature_src', None)  # for old version

    def forward(self, batch_dict, stensor=None):
        if stensor is None:
            # for old version
            assert self.feature_scr is not None
            stensor = batch_dict[self.feature_scr][f'p{self.stride}']

        coor = stensor['coor'][:, :3]
        feat = stensor['feat']

        mask, indices = self.valid_coords(coor)
        d = feat.shape[1]
        coor = coor[mask]
        feat = feat[mask].view(-1, d)

        num_cav = batch_dict['num_cav']
        for i, n in enumerate(num_cav):
            ptr = sum(num_cav[:i])
            cur_mask = (indices[:, 0] >= ptr) & (indices[:, 0] < ptr + n)
            coor[cur_mask, 0] = i

            # if i==0:
            #     import matplotlib.pyplot as plt
            #     points = cur_coords.detach().cpu().numpy()
            #     fig = plt.figure(figsize=(14, 4))
            #     plt.plot(points[:, 1], points[:, 2], '.', markersize=.5)
            #     plt.show()
            #     plt.close()

        batch_dict['naive_fusion'] = {
            f'p{self.stride}': {
                'coor': coor,
                'feat': feat
            }
        }

    def valid_coords(self, coor):
        # remove voxels that are outside range
        xi = torch.div(coor[:, 1], self.stride, rounding_mode='floor') - self.offset_sz_x
        yi = torch.div(coor[:, 2], self.stride, rounding_mode='floor') - self.offset_sz_y

        mask = (xi >= 0) * (xi < self.size_x) * (yi >= 0) * (yi < self.size_y)
        indices = torch.stack(
            [coor[:, 0][mask].long(),
             xi[mask].long(),
             yi[mask].long()],
             dim=1)

        return mask, indices


