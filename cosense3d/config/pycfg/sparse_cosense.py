from collections import OrderedDict

# point_cloud_range = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
# point_cloud_range_enlarged = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
point_cloud_range = [-144, -41.6, -5.0, 144, 41.6, 3.0]
point_cloud_range_test = [-140.8, -38.4, -3.0, 140.8, 38.4, 3.0]
voxel_size = [0.2, 0.2, 8]

pts_backbone = dict(
    type='MinkUnet',
    voxel_size=voxel_size,
    d=2,
    cache_strides=[4],
    in_dim=3,
    stride=4,
    floor_height=point_cloud_range[2]
)

pts_neck = dict(
    type='MinkExpand',
    voxel_size=voxel_size,
    lidar_range=point_cloud_range,
    stride=4,
    convs=dict(kernels=[5, 5, 3], in_dim=64, out_dim=128)
)