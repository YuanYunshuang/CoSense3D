from cosense3d.config import add_cfg_keys

@add_cfg_keys
def get_bev_semseg_head_cfg(
        semseg_head_type,
        in_dim,
        data_info,
        stride,
        tgt_assigner_type,
):
    return dict(
            type=semseg_head_type,
            data_info=data_info,
            in_dim=in_dim,
            stride=stride,
            # dynamic_head=False,
            target_assigner=dict(
                type=tgt_assigner_type,
                down_sample=False,
                data_info=data_info,
                stride=stride,
                tgt_range=51.2
            ),
            loss_cls=dict(type='EDLLoss', activation='relu', annealing_step=20,
                          n_cls=2, loss_weight=1.0),
        )


