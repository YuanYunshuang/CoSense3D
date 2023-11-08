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

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if self.num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet "
                "layers".format(self.num_layers))

        self.encoder = resnets[self.num_layers](self.pretrained)

    def forward(self, input_images, **kwargs):
        """
        Compute deep features from input images.

        Parameters
        ----------
        input_images : List[torch.Tensor]
            The input list has length L, and each image has the shape of (H,W,3)

        Returns
        -------
        features: torch.Tensor
            The deep features for each image
        """
        b, l, m, h, w, c = input_images.shape
        input_images = input_images.view(b*l*m, h, w, c)
        # b, h, w, c -> b, c, h, w
        input_images = input_images.permute(0, 3, 1, 2).contiguous()

        x = self.encoder.conv1(input_images)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = rearrange(x, '(b l m) c h w -> b l m c h w',
                      b=b, l=l, m=m)

        return x