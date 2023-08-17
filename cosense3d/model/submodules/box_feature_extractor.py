import torch
import torch_scatter

from cosense3d.model.utils import *
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.ops.iou3d_nms_utils import aligned_boxes_iou3d_gpu, boxes_iou3d_gpu
from cosense3d.model.losses.common import weighted_smooth_l1_loss


class BoxFeatureExtractor(nn.Module):
    def __init__(self, cfgs):
        super(BoxFeatureExtractor, self).__init__()

        self.in_channels = cfgs['in_channels']
        self.grid_size = cfgs.get('rcnn_grid_size', 6)
        grid = (self.grid_size, ) * 3 + (self.in_channels,)
        self.grid_emb = nn.Parameter(torch.randn(grid))
        self.pos_emb_layer = linear_layers([3, 32, 32])
        self.attn_weight = linear_layers([self.in_channels, 32, 1],
                                         ['ReLU', 'Sigmoid'])

        self.proj = linear_layers([self.in_channels, 32])
        self.fc_layer = linear_layers([64, 64])
        self.fc_out = linear_layers([64, 64])

        self.empty = False

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                roi:
                    box: (N, 8), 1st column is batch index
                    scr: (N), optional
                gt_boxes: (N, 8)
        Returns:

        """
        coords = self.get_coords(batch_dict)
        assert 'roi' in batch_dict, "no ROI result from stage 1."
        assert 'p0' in batch_dict['backbone'], "no p0 features in found."
        preds = batch_dict['roi']
        boxes = preds['box'].clone()
        if len(boxes) == 0:
            batch_dict['box_features'] = []
            return batch_dict

        boxes[:, 4:7] *= 1.5
        batch_size = batch_dict['batch_size'] * batch_dict.get('seq_len', 1)
        boxes_decomposed, box_idxs_of_pts = points_in_boxes_gpu(
            coords[:, :4], boxes, batch_size
        )
        in_box_mask = box_idxs_of_pts >= 0
        new_idx = box_idxs_of_pts[in_box_mask]

        new_xyz = coords[in_box_mask, 1:4]
        features = batch_dict['backbone']['p0'][in_box_mask]
        mapped_boxes = boxes_decomposed[new_idx]

        # canonical transformation
        new_xyz = new_xyz - mapped_boxes[:, 1:4]
        xyz = new_xyz.clone()
        st = torch.sin(-mapped_boxes[:, -1])
        ct = torch.cos(-mapped_boxes[:, -1])
        new_xyz[:, 0] = xyz[:, 0] * ct - xyz[:, 1] * st
        new_xyz[:, 1] = xyz[:, 0] * st + xyz[:, 1] * ct

        # import matplotlib.pyplot as plt
        # points = new_xyz[new_idx==0]
        # points = torch.div(points, mapped_boxes[:, 4:7][new_idx==0]).cpu().numpy() * 3
        # plt.plot(points[:, 0], points[:, 1], '.')
        # plt.plot([-1, 1, 1, -1., -1.], [-1., -1., 1., 1., -1.])
        # plt.show()
        # plt.close()

        # grid size minus 1e-4 to ensure positive coords
        new_tfield = torch.div(new_xyz, mapped_boxes[:, 4:7]) * self.grid_size + 3
        new_tfield = torch.clamp(new_tfield, 0, self.grid_size - 1e-4)

        new_tfield = ME.TensorField(
            coordinates=torch.cat([new_idx.view(-1, 1), new_tfield], dim=1),
            features=features,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
        )
        voxel_embs = self.voxelize_with_centroids(new_tfield, new_xyz)
        voxel_idx = voxel_embs.C[:, 0].long()
        num_box = len(boxes_decomposed)
        pos = voxel_embs.C.T[1:].long()
        # if pos.min() < 0 or pos.max() > self.grid_size - 1:
        #     print('d')
        pos_emb = self.grid_emb[pos[0], pos[1], pos[2]]
        voxel_embs = voxel_embs.F.contiguous() + pos_emb
        weights = self.attn_weight(voxel_embs)
        weighted_voxel_features = weights * voxel_embs
        out = torch.zeros_like(weighted_voxel_features[:num_box])
        torch_scatter.scatter_add(weighted_voxel_features,
                                  voxel_idx, dim=0, out=out)

        out = self.fc_out(out)
        batch_dict['box_features'] = out
        return batch_dict

    def get_coords(self, batch_dict):
        coords = batch_dict['pcds']
        if 'num_cav' in batch_dict:
            # merge coordinates from different cavs
            if len(coords[:, 0].unique()) > batch_dict['batch_size']:
                for i, c in enumerate(batch_dict['num_cav']):
                    idx_start = sum(batch_dict['num_cav'][:i])
                    mask = torch.logical_and(
                        coords[:, 0] >= idx_start,
                        coords[:, 0] < idx_start + c
                    )
                    coords[mask, 0] = i
        return coords

    def voxelize_with_centroids(self, x: ME.TensorField, coords: torch.Tensor):
        cm = x.coordinate_manager
        features = x.F
        # coords = x.C[:, 1:]

        out = x.sparse()
        size = torch.Size([len(out), len(x)])
        tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
        coords_p1, count_p1 = downsample_points(coords, tensor_map, field_map, size)
        norm_coords = normalize_points(coords, coords_p1, tensor_map)
        pos_emb = self.pos_emb_layer(norm_coords)
        feat_enc = self.proj(features)

        voxel_embs = self.fc_layer(torch.cat([feat_enc, pos_emb], dim=1))
        down_voxel_embs = downsample_embeddings(voxel_embs, tensor_map, size, mode="max")
        out = ME.SparseTensor(down_voxel_embs,
                              coordinate_map_key=out.coordinate_key,
                              coordinate_manager=cm)
        return out

