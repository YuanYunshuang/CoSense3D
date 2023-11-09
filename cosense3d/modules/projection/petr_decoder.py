from typing import List

import torch
from torch import nn

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.transformer import build_module
from cosense3d.modules.utils.common import inverse_sigmoid
from cosense3d.modules.utils.misc import SELayer_Linear, MLN


class PETRDecoder(BaseModule):
    def __init__(self, in_channels, decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = build_module(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.img_position_dim = 128
        self.num_pose_feat = 64
        self.in_channels = in_channels

        self.img_position_encoder = nn.Sequential(
            nn.Linear(self.img_position_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.img_memory_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        # can be replaced with MLN
        self.featurized_img_pe = SELayer_Linear(self.embed_dims)

        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat*3, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(8, f_dim=self.embed_dims)
        self.pts_spatial_alignment = MLN(2, f_dim=self.embed_dims)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.utils.init.xavier_uniform_(m)
        self._is_init = True

    def forward(self, img_feat, img_size, intrinsics, lidar2img, **kwargs):
        img_memory, attn_mask = self.format_input(img_feat)
        img_coors = self.img_coors(img_size, img_feat)

        img_pos_emb, cone = self.img_position_embeding(img_feat, img_coors, img_size, intrinsics, lidar2img)
        img_memory = self.img_memory_embed(img_memory)

        # spatial_alignment in focal petr
        img_memory = self.spatial_alignment(img_memory, cone)
        img_pos_emb = self.featurized_img_pe(img_pos_emb, img_memory)

        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)
        outs_dec_img, _ = self.img_transformer(img_memory, tgt, query_pos, img_pos_emb, attn_mask)
        return outs_dec_img

    def format_input(self, input: List):
        memory = []
        for x in input:
            x = x.permute(0, 2, 3, 1).flatten(0, 2)
            memory.append(x)
        max_l = max([m.shape[0] for m in memory])
        out = x.new_zeros(len(memory), max_l, x.shape[-1])
        mask = x.new_ones(len(memory), max_l)
        for i, m in enumerate(memory):
            out[i, :len(m)] = m
            mask[i, :len(m)] = False
        return out, mask

    def img_coors(self, img_size, img_feat):
        H, W = img_size[0][0]
        h, w = img_feat[0].shape[2:]
        device = img_feat[0].device
        stride = H // h

        shifts_x = (torch.arange(
            0, stride * w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / W
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / H
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        coors = torch.stack((shift_x, shift_y), dim=1)

        coors = coors.reshape(h, w, 2)
        return coors

    def decode(self, memory, tgt, query_pos, pos_embed, attn_masks=None, query_mask=None):
        """

        Parameters
        ----------
        memory: (B, L, D)
        tgt: (B, S, D)
        query_pos: (B, S, D)
        pos_embed: (B, S, D)
        attn_masks: (B, S, L)
        query_mask: (B, S)

        Returns
        -------

        """
        memory = memory.transpose(0, 1).contiguous()
        query_pos = query_pos.transpose(0, 1).contiguous()
        pos_embed = pos_embed.transpose(0, 1).contiguous()

        n, bs, c = memory.shape

        if tgt is None:
            tgt = torch.zeros_like(query_pos)
        else:
            tgt = tgt.transpose(0, 1).contiguous()

        # out_dec: [num_layers, num_query, bs, dim]
        if not isinstance(attn_masks, list):
            attn_masks = [attn_masks, None]
        out_dec = self.decoder(
            query=tgt,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_pos,
            query_key_padding_mask=query_mask,
            attn_masks=attn_masks,
        )
        out_dec = out_dec.transpose(1, 2).contiguous()
        memory = memory.reshape(-1, bs, c).transpose(0, 1).contiguous()
        return out_dec, memory

    def img_position_embeding(self, img_feat, img_coor, img_size, intrinsic, lidar2img):
        eps = 1e-5
        B = len(img_feat)
        H, W = img_size[0][0]
        n_imgs = sum([len(x) for x in img_feat])

        # [alpha_x, alpha_y]
        intrinsic = torch.stack([intrinsic[..., 0, 0], intrinsic[..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(n_imgs, -1, 2)

        # transform memery_centers from ratio to pixel
        h, w = img_coor.shape[:2]
        img_coor[..., 0] = img_coor[..., 0] * w
        img_coor[..., 1] = img_coor[..., 1] * h

        D = self.coords_d.shape[0]
        num_sample_tokens = h * w

        memory_centers = img_coor.detach().view(n_imgs, -1, 1, 2)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1).to(img_coor.device)
        coords = torch.cat([img_coor, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = lidar2img.inverse()
        img2lidars = img2lidars.view(n_imgs, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(n_imgs, H*W, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        self.position_range = self.position_range.to(coords3d.device)
        coords3d[..., 0:3] = ((coords3d[..., 0:3] - self.position_range[0:3]) /
                              (self.position_range[3:6] - self.position_range[0:3]))
        coords3d = coords3d.reshape(B, -1, D*3)

        pos_embed = inverse_sigmoid(coords3d)
        coords_position_embeding = self.img_position_encoder(pos_embed)

        # for spatial alignment in focal petr
        # TODO: alpha_x and alpha_y, far bound at about 60m and near bound at about 18m (why?)
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)

        return coords_position_embeding, cone

