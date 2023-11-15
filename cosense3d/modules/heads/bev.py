from cosense3d.modules import BaseModule
from cosense3d.modules.utils.me_utils import *
from cosense3d.modules.utils.common import pad_r, linear_last, cat_coor_with_idx
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.losses import edl, build_loss
from cosense3d.modules.plugin import build_plugin_module


class BEV(BaseModule):
    def __init__(self,
                 data_info,
                 in_dim,
                 stride,
                 target_assigner,
                 loss_cls,
                 class_names_each_head=None,
                 **kwargs):
        super(BEV, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.class_names_each_head = class_names_each_head
        self.stride = stride
        for k, v in data_info.items():
            setattr(self, k, v)
        update_me_essentials(self, data_info, self.stride)

        self.reg_layer = linear_last(in_dim, 32, 2, bias=True)

        self.tgt_assigner = build_plugin_module(target_assigner)
        self.loss_cls = build_loss(**loss_cls)

    def forward(self, stensor_list, **kwargs):
        coor, feat = self.format_input(stensor_list)

        if self.training:
            coor, feat = self.down_sample(coor, feat)

        centers = indices2metric(coor, self.voxel_size)
        reg = self.reg_layer(feat)
        conf, unc = edl.evidence_to_conf_unc(reg.relu())

        out = {
            'center': centers,
            'reg': reg,
            'conf': conf,
            'unc': unc
        }

        return self.format_output(out, len(stensor_list))

    def format_input(self, stensor_list):
        return self.compose_stensor(stensor_list, self.stride)

    def format_output(self, output, B=None):
        # decompose batch
        output_new = {k: [] for k in output.keys()}
        for i in range(B):
            mask = output['center'][:, 0] == i
            output_new['center'].append(output['center'][mask, 1:])
            output_new['reg'].append(output['reg'][mask])
            output_new['conf'].append(output['conf'][mask])
            output_new['unc'].append(output['unc'][mask])
        output = {self.scatter_keys[0]: self.compose_result_list(output_new, B)}
        return output

    def down_sample(self, coor, feat):
        keep = torch.rand_like(feat[:, 0]) > 0.5
        coor = coor[keep]
        feat = feat[keep]

        return coor, feat

    def loss(self, batch_list, gt_boxes, gt_labels, **kwargs):
        tgt_pts = self.cat_data_from_list(batch_list, 'center', pad_idx=True)
        gt_boxes = self.cat_data_from_list(gt_boxes, pad_idx=True)
        conf = self.cat_data_from_list(batch_list, 'conf')
        tgt_pts, tgt_label, valid = self.tgt_assigner.assign(
            tgt_pts, gt_boxes, len(batch_list), conf, **kwargs)
        epoch_num = kwargs.get('epoch', 0)
        reg = self.cat_data_from_list(batch_list, 'reg')
        avg_factor = max(tgt_label.sum(), 1)
        loss_cls = self.loss_cls(
            reg[valid],
            tgt_label,
            temp=epoch_num,
            avg_factor=avg_factor
        )
        loss_dict = {'bev_loss': loss_cls}
        return loss_dict





