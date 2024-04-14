import math
from collections import OrderedDict


def get_minkunet_cfg(gather_keys, scatter_keys, voxel_size, point_cloud_range,
                     in_dim=4, dim=3, out_stride=2, height_compression=[2, 8],
                     compression_kernel_size_xy=1, cache_strides=[2, 8]):
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    if len(height_compression) > 0:
        hc = OrderedDict()
        height = (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]
        for stride in height_compression:
            downx = math.ceil(height / stride)
            if downx > 4:
                hc[f'p{stride}'] = dict(channels=[128, 256, 384], steps=[5, downx // 5])
            else:
                hc[f'p{stride}'] = dict(channels=[128, 256], steps=[downx])
    else:
        hc = None
    return dict(
        type='backbone3d.mink_unet.MinkUnet',
        gather_keys=gather_keys,
        scatter_keys=scatter_keys,
        d=dim,
        cache_strides=cache_strides,
        kernel_size_layer1=5,
        in_dim=in_dim,
        stride=out_stride,
        floor_height=point_cloud_range[2],
        data_info=data_info,
        height_compression=hc,
        compression_kernel_size_xy=compression_kernel_size_xy
    )