import torch
from cosense3d.model.utils import meshgrid_cross


class AnchorGenerator(object):
    def __init__(self, cfg, coor_lim, stride, voxel_size):
        self.cfg = cfg
        self.coor_lim = coor_lim
        self.stride = stride
        self.voxel_size = voxel_size
        self.template = self.generate_anchor_template()

        self.iou_match = torch.Tensor(self.iou_match)
        self.iou_unmatch = torch.Tensor(self.iou_unmatch)

    def generate_anchor_template(self):
        x_min, y_min, x_max, y_max = self.coor_lim
        xy = meshgrid_cross([x_min, y_min], [x_max, y_max],
                            steps=[self.stride, self.stride],
                            )  # h w 2
        # ME floors the input coordinates,
        # the real metric center of the current voxel should plus
        # the half of the current voxel size which is
        # original voxel size * stride / 2 = 0.1 * 8 / 2
        xy[..., 0] = (xy[..., 0] + self.stride / 2) * self.voxel_size[0]
        xy[..., 1] = (xy[..., 1] + self.stride / 2) * self.voxel_size[1]
        h, w, _ = xy.shape
        anchors = []
        self.num_cls = 0
        self.iou_match = []
        self.iou_unmatch = []
        for k, v in self.cfg.items():
            num = len(v['box_angles']) * len(v['box_dim']) * len(v['box_z'])
            self.num_cls += num
            self.iou_match += [v['iou_match']] * num
            self.iou_unmatch += [v['iou_unmatch']] * num

            cur_anchors = torch.zeros((h, w, len(v['box_angles']), 7))  # h w 2 7
            cur_anchors[..., :2] = torch.tile(xy.unsqueeze(2), (1, 1, 2, 1))  # set x y
            # TODO support multiple z and box dimensions, currently only support one
            cur_anchors[..., 2] = v['box_z'][0]  # set z
            cur_anchors[..., 3] = v['box_dim'][0][0]  # set l
            cur_anchors[..., 4] = v['box_dim'][0][1]  # set w
            cur_anchors[..., 5] = v['box_dim'][0][2]  # set h
            cur_anchors[..., 6] = torch.deg2rad(torch.Tensor(v['box_angles']))  # set angle
            anchors.append(cur_anchors)
        anchors = torch.cat(anchors, dim=-2)
        return anchors

    def anchors(self, coords=None):
        if coords is not None:
            anchors = self.template[coords[:, 0], coords[:, 1]]
        else:
            anchors = self.template
        return anchors