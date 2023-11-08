import torch
import torch.nn as nn
import torchvision.models as models

from einops import rearrange

from cosense3d.modules import BaseModule


class ResnetEncoder(BaseModule):
    """Resnet family to encode image."""
    def __init__(self, num_layers, pretrained=True, **kwargs):
        super(ResnetEncoder, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.pretrained = pretrained

        resnet = getattr(models, f'resnet{self.num_layers}', None)

        if resnet is None:
            raise ValueError(f"{self.num_layers} is not a valid number of resnet ""layers")

        resnet_weights = getattr(models, f"ResNet{self.num_layers}_Weights")
        self.encoder = resnet(weights=resnet_weights.DEFAULT)

    def forward(self, input_images, **kwargs):
        num_imgs = [len(x) for x in input_images]
        imgs = self.compose_imgs(input_images)
        b, h, w, c = imgs.shape

        # b, h, w, c -> b, c, h, w
        imgs = imgs.permute(0, 3, 1, 2).contiguous()

        x = self.encoder.conv1(imgs)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        return self.format_output(x, num_imgs)

    def format_output(self, output, num_imgs):
        ptr = 0
        output_list = []
        for n in num_imgs:
            output_list.append(output[ptr:ptr+n])
            ptr += n
        return {self.scatter_keys[0]: output_list}