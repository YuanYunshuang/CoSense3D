import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from .multi_modal_cav import BaseCAV


class StreamLidarCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lidar_range = torch.nn.Parameter(self.lidar_range)
        self.prepare_data_keys = ['points', 'annos_local', 'annos_global']
        self.data['memory'] = None

    def refresh_memory(self, prev_exists):
        x = prev_exists.float()
        if self.data['memory'] is None:
            self.data['memory'] = {
                'embeddings': x.new_zeros(self.memory_len, self.memory_num_propagated, self.memory_emb_dims),
                'ref_pts': x.new_zeros(self.memory_len, self.memory_num_propagated, self.ref_pts_dim),
                'timestamp': x.new_zeros(self.memory_len, self.memory_num_propagated, 1),
                'pose': x.new_zeros(self.memory_len, self.memory_num_propagated, 4, 4) ,
                'velo': x.new_zeros(self.memory_len, self.memory_num_propagated, 2),
            }
        else:
            for k, v in self.data['memory'].items():
                self.data['memory'][k] = self.data['memory'][k][:self.memory_len] * x
        if not x:
            # pose will all become zeros after refreshing, recover recent pose to eye
            self.data['memory']['pose'][0] = self.data['memory']['pose'][0] + \
                                          torch.eye(4, device=x.device).unsqueeze(0)
        self.data['memory']['prev_exists'] = x

    def prepare_data(self):
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def get_response_cpm(self):
        cpm = {}
        # for k in ['bev_feat']:
        #     if k in self.data:
        #         cpm[k] = self.data[k]
        return cpm

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '01:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '02:backbone_neck', {}))
        tasks[grad_mode].append((self.id, '03:roi_head', {}))
        tasks[grad_mode].append((self.id, '04:temp_fusion', {}))

    def forward_fusion(self, tasks, training_mode):
        # if self.is_ego:
        #     tasks['with_grad'].append((self.id, '11:fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '12:detection_head', {}))
        return tasks

    def loss(self, tasks):
        if self.is_ego:
            tasks['loss'].append((self.id, '21:roi_head', {}))
            tasks['loss'].append((self.id, '22:detection_head', {}))
        return tasks

    def pre_update_memory(self):
        """Update memory before each forward run of a single frame."""
        if not self.is_ego:
            return
        if self.data['memory'] is not None:

            pose_inv = self.lidar_pose.inverse()

            self.data['memory']['timestamp'] += self.timestamp
            self.data['memory']['pose'] = pose_inv @ self.data['memory']['pose']
            self.data['memory']['ref_pts'] = self.transform_ref_pts(
                self.data['memory']['ref_pts'], pose_inv)

        self.refresh_memory(self.data['prev_exists'])

    def post_update_memory(self):
        """Update memory after each forward run of a single frame."""
        x = self.data['petr_out']
        scores = x['all_cls_scores'][-1].sigmoid().topk(1, dim=-1).values[..., 0]
        topk = torch.topk(scores, k=self.memory_num_propagated).indices

        ref_pts = x['all_bbox_preds'][-1][:, :self.ref_pts_dim]
        velo = x['all_bbox_preds'][-1][:, -2:]
        embeddings = self.data['temp_fusion_feat']['outs_dec'][-1]
        timestamp = torch.zeros_like(scores).view(-1, 1)
        pose = torch.eye(4, device=ref_pts.device).unsqueeze(0).repeat(
            timestamp.shape[0], 1, 1)

        vars = locals()
        for k, v in self.data['memory'].items():
            if k == 'prev_exists':
                continue
            rec_topk = vars[k][topk].unsqueeze(0)
            self.data['memory'][k] = torch.cat([rec_topk, v], dim=0)

        self.data['memory']['ref_pts'] = self.transform_ref_pts(
            self.data['memory']['ref_pts'], self.lidar_pose)
        self.data['memory']['timestamp'] -= self.timestamp
        self.data['memory']['pose'] = self.lidar_pose[(None,)*2] @ self.data['memory']['pose']

    def transform_ref_pts(self, reference_points, matrix):
        reference_points = torch.cat(
            [reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
        reference_points = matrix.unsqueeze(0) @ reference_points.permute(0, 2, 1)
        return reference_points.permute(0, 2, 1)[..., :3]

    @property
    def timestamp(self):
        if 'timestamp' in self.data:
            timestamp = self.data['timestamp']
        else:
            timestamp = float(self.data['frame']) * 0.1
        return timestamp




