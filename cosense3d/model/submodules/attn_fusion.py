import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from cosense3d.model.submodules.attention import CrossAttention
from cosense3d.model.utils.me_utils import update_me_essentials


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttnFusion(nn.Module):
    def __init__(self, cfgs):
        super(AttnFusion, self).__init__()
        # self.voxel_size = cfgs['data_info']['voxel_size']
        # self.det_r = cfgs['data_info']['det_r']
        update_me_essentials(self, cfgs['data_info'], stride=cfgs['stride'])
        self.d = len(self.voxel_size)
        self.feature_scr = cfgs['feature_src']
        self.attn = ScaledDotProductAttention(cfgs['dim'])

    def forward(self, batch_dict):
        stensor = batch_dict[self.feature_scr][f'p{self.stride}']
        coor = stensor['coor']
        feat = stensor['feat']

        mask, indices = self.valid_coords(coor)
        d = feat.shape[1]
        coor = coor[mask]
        feat = feat[mask].view(-1, d)

        num_cav = batch_dict['num_cav']
        out_features = []
        out_coords = []
        for i, n in enumerate(num_cav):
            ptr = sum(num_cav[:i])
            cur_mask = (indices[:, 0] >= ptr) & (indices[:, 0] < ptr + n)
            cur_indices = indices[cur_mask]
            cur_indices[:, 0] -= ptr
            cur_coor = coor[cur_mask][:, :3]
            cur_feat = feat[cur_mask]
            # make feat map as attn input
            feat_map = torch.zeros((n, self.size_x, self.size_y, d), device=feat.device)
            feat_map[cur_indices[:, 0], cur_indices[:, 1], cur_indices[:, 2]] = cur_feat
            xx = feat_map.permute(1, 2, 0, 3).view(-1, n, d)
            # do attn and retrive sparse result from feat map
            feat_map = self.attn(xx, xx, xx)[:, 0].view(self.size_x, self.size_y, d)
            fused_indices, reverse_indices = cur_indices[:, 1:].unique(dim=0, return_inverse=True)
            cur_feat = feat_map[fused_indices[:, 0], fused_indices[:, 1]]
            # get fused coordinates
            cur_coor[:, 0] = i
            cur_coor = scatter_mean(cur_coor, reverse_indices, dim=0)

            # if i==0:
            #     import matplotlib.pyplot as plt
            #     points = cur_coords.detach().cpu().numpy()
            #     fig = plt.figure(figsize=(14, 4))
            #     plt.plot(points[:, 1], points[:, 2], '.', markersize=.5)
            #     plt.show()
            #     plt.close()

            out_coords.append(cur_coor)  # ME coords
            out_features.append(cur_feat)
        batch_dict['attn_fusion'] = {
            f'p{self.stride}': {
                'coor': torch.cat(out_coords, dim=0),
                'feat': torch.cat(out_features, dim=0)
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
