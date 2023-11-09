import torch
import torch.nn as nn
import torchvision.models as models

from einops import rearrange

from cosense3d.modules import BaseModule
from cosense3d.modules.plugin import build_plugin_module
from cosense3d.modules.utils.positional_encoding import img_locations


class ResnetEncoder(BaseModule):
    """Resnet family to encode image."""
    def __init__(self, num_layers, feat_indices, out_index, img_size,
                 pretrained=True, neck=None, **kwargs):
        super(ResnetEncoder, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.feat_indices = sorted(feat_indices)
        self.out_index = out_index
        self.img_size = img_size
        self.stride = 2 ** (self.out_index + 1)
        self.feat_size = (img_size[0] // self.stride, img_size[1] // self.stride)
        self.img_locations = nn.Parameter(img_locations(img_size, self.feat_size))
        self.pretrained = pretrained

        resnet = getattr(models, f'resnet{self.num_layers}', None)

        if resnet is None:
            raise ValueError(f"{self.num_layers} is not a valid number of resnet ""layers")

        resnet_weights = getattr(models, f"ResNet{self.num_layers}_Weights")
        self.encoder = resnet(weights=resnet_weights.DEFAULT)
        self.neck = build_plugin_module(neck) if neck is not None else None

    def forward(self, input_images, **kwargs):
        num_imgs = [len(x) for x in input_images]
        imgs = self.compose_imgs(input_images)
        b, h, w, c = imgs.shape

        # b, h, w, c -> b, c, h, w
        imgs = imgs.permute(0, 3, 1, 2).contiguous()

        x = self.encoder.conv1(imgs)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        out = []
        for i in range(1, 5):
            x = getattr(self.encoder, f'layer{i}')(x)
            if i in self.feat_indices:
                out.append(x)

        if self.neck is not None:
            out = self.neck(out)
        out = out[self.feat_indices.index(self.out_index)]
        return self.format_output(out, num_imgs)

    def format_output(self, output, num_imgs):
        ptr = 0
        output_list = []
        coor_list = []
        for n in num_imgs:
            if isinstance(output, (tuple, list)):
                output_list.append(tuple(out[ptr:ptr+n] for out in output))
            else:
                output_list.append(output[ptr:ptr + n])
            if 'img_coor' in self.scatter_keys:
                coor_list.append(self.img_locations.unsqueeze(0).repeat(n, 1, 1, 1))
            ptr += n
        out_dict = {}
        if 'img_feat' in self.scatter_keys:
            out_dict['img_feat'] = output_list
        if 'img_coor' in self.scatter_keys:
            out_dict['img_coor'] = coor_list

        return out_dict