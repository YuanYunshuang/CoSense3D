import torch
from torch import nn
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck

from cosense3d.modules.plugin.cobevt import CrossViewSwapAttention, Attention, BEVEmbedding
ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


class FAXModule(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()

        middle = config['middle']
        dim = config['dim']
        self.backbone_output_shape = config['backbone_output_shape']
        assert len(middle) == len(self.backbone_output_shape)

        cross_view = config['cross_view']
        cross_view_swap = config['cross_view_swap']

        cross_views = list()
        layers = list()
        downsample_layers = list()

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone_output_shape, middle)):
            _, _, _, feat_dim, feat_height, feat_width = \
                torch.zeros(feat_shape).shape

            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim,
                                         dim[i], i,
                                         **cross_view, **cross_view_swap)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(dim[i], dim[i] // 4,
                                  kernel_size=3, stride=1,
                                  padding=1, bias=False),
                        nn.PixelUnshuffle(2),
                        nn.Conv2d(dim[i+1], dim[i+1],
                                  3, padding=1, bias=False),
                        nn.BatchNorm2d(dim[i+1]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim[i+1],
                                  dim[i+1], 1, padding=0, bias=False),
                        nn.BatchNorm2d(dim[i+1])
                        )))

        self.bev_embedding = BEVEmbedding(dim[0], **config['bev_embedding'])
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.self_attn = Attention(dim[-1], **config['self_attn'])

    def forward(self, batch):
        b, l, n, _, _, _ = batch['inputs'].shape

        I_inv = \
            rearrange(batch['intrinsic'], 'b l m h w -> (b l) m h w').inverse()
        E_inv = rearrange(batch['extrinsic'],
                          'b l m h w -> (b l) m h w')
        features = batch['features']

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b * l)  # b*l d H W

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, 'b l n ... -> (b l) n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        x = self.self_attn(x)
        x = rearrange(x, '(b l) ... -> b l ...', b=b, l=l)
        return x

