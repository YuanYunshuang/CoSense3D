from collections import OrderedDict

# point_cloud_range = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
# point_cloud_range_enlarged = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]
point_cloud_range = [-51.2, -38.4, -5, 51.2, 38.4, 3]
voxel_size = [0.2, 0.2, 8]
data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)

"""
gather_keys: 
    keys to gather data from cavs, key order is important, should match the forward input arguments order.
scatter_keys: 
    1st key in the list is used as the key for scattering and storing module output data to cav.
"""
shared_modules = OrderedDict(
    img_backbone = dict(
        type='backbone2d.resnet_encoder.ResnetEncoder',
        gather_keys=['img'],
        scatter_keys=['img_feat'],
        num_layers=18,
    ),

    img2bev = dict(
        type='projection.petr_decoder.PETRDecoder',
        gather_keys=['img_feat', 'img_size', 'intrinsics', 'lidar2img'],
        scatter_keys=['bev_feat'],
        decoder=dict(
            type='TransformerDecoder',
            return_intermediate=True,
            num_layers=3,
            transformerlayers=dict(
                type='TransformerDecoderLayer',
                attn_cfgs=[
                    dict(type='MultiheadAttention',  # fp16 for 2080Ti training (save GPU memory).
                         embed_dims=128,
                         num_heads=8,
                         dropout=0.1)
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=128,
                    feedforward_channels=1024,
                    num_fcs=2,
                    dropout=0.,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=1024,
                ffn_dropout=0.1,
                with_cp=False,  ###use checkpoint to save memory
                operation_order=('cross_attn', 'norm', 'ffn', 'norm')),
        )
    ),

)

train_hooks = [
        dict(type='TrainTimerHook'),
        dict(type="CheckPointsHook", epoch_every=10)
    ]


test_hooks = [
        dict(type="EvalOPV2VDetectionHook"),
        dict(type="EvalDenseBEVHook", thr=0.5)
    ]


plots = [
    # dict(title='BEVSparseCanvas', width=10, height=4, nrows=1, ncols=1),
    # dict(title='DetectionCanvas', width=10, height=4, nrows=1, ncols=1)
]