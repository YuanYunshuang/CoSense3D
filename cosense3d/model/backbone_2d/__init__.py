from importlib import import_module

import torch
import torch.nn as nn
import torch.utils.checkpoint

from cosense3d.model.utils.grid_mask import GridMask
from cosense3d.model.utils import instantiate


class Backbone2d(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.num_frame_backbone_grads = 2
        self.use_grid_mask = True
        self.with_img_neck = True
        self.position_level = 0
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.img_backbone = instantiate(**cfgs['backbone'])
        if self.with_img_neck:
            self.img_neck = instantiate(**cfgs['neck'])


    def extract_feat(self, img, len_queue=1, training_mode=False):
        """
        Extract image features with backbone network.

        Parameters
        ----------
        img: Tensor(B, T, L, C, H, W)
        len_queue: int, queue length that should get gradiants
        training_mode: bool, whether run the operation in training mode

        Returns
        -------
        img_feats: Tensor(B, T, L, C, H, W)
        """
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN /B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN / B /len_queue), C, H, W)


        return img_feats_reshaped

    def forward(self, batch_dict):
        T = batch_dict['seq_len']

        prev_img = batch_dict['img'][:, :-self.num_frame_backbone_grads]
        rec_img = batch_dict['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T- self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T - self.num_frame_backbone_grads, True)
            self.train()
            batch_dict['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            batch_dict['img_feats'] = rec_img_feats

        # x = batch_dict['imgs'].permute(0, 1, 4, 2, 3).contiguous().flatten(0, 1)
        # batch_dict['backbone_2d'] = self.forward_pass(x)