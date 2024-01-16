import torch
import torch_scatter
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from cosense3d.agents.cav_prototype.base_cav import BaseSeqCAV


class StreamLidarCAV(BaseSeqCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lidar_range = torch.nn.Parameter(self.lidar_range)
        self.prepare_data_keys = ['points', 'annos_local', 'annos_global']
        self.memory = None
        self.aug_transform = None
        self.T_aug2g = {}
        self.T_g2aug = {}
        self.T_e2g = {}
        self.use_aug = True
        self.cur_seq_idx = 0

    def refresh_memory(self, prev_exists):
        x = prev_exists.float()
        init_pose = torch.eye(4, device=x.device).unsqueeze(0).unsqueeze(0)
        if self.memory is None:
            self.memory = {
                'embeddings': x.new_zeros(self.memory_len, self.memory_num_propagated, self.memory_emb_dims),
                'ref_pts': x.new_zeros(self.memory_len, self.memory_num_propagated, self.ref_pts_dim),
                'timestamp': x.new_zeros(self.memory_len, self.memory_num_propagated, 1),
                'pose': x.new_zeros(self.memory_len, self.memory_num_propagated, 4, 4) ,
                'pose_no_aug': x.new_zeros(self.memory_len, self.memory_num_propagated, 4, 4) ,
                'velo': x.new_zeros(self.memory_len, self.memory_num_propagated, 2),
            }
            self.memory['pose_no_aug'] = self.memory['pose'] + init_pose
        else:
            for k, v in self.memory.items():
                self.memory[k] = self.memory[k][:self.memory_len] * x
            if not x:
                self.memory['pose_no_aug'][0] = init_pose[0].repeat(self.memory_num_propagated, 1, 1)
        self.memory['prev_exists'] = x

    def apply_transform(self, seq_idx):
        data = self.data[seq_idx]
        lidar_pose = data['lidar_poses']
        if self.use_aug:
            if self.is_ego:
                T_e2g = lidar_pose
                T_g2e = T_e2g.inverse()
                T_c2e = torch.eye(4).to(T_e2g.device)
            else:
                # cav to ego
                T_e2g = data['received_request']['lidar_poses']
                T_g2e = data['received_request']['lidar_poses'].inverse()
                T_c2e = T_g2e @ lidar_pose

            if self.aug_transform is None:
                self.aug_transform = DOP.update_transform_with_aug(
                    torch.eye(4).to(lidar_pose.device), data['augment_params'])
                T_e2aug = self.aug_transform
            else:
                # adapt aug params to the current ego frame
                T_e2aug = self.T_g2aug[seq_idx-1] @ T_e2g

            T_c2aug = T_e2aug @ T_c2e
            T_g2aug = T_e2aug @ T_g2e

            DOP.apply_transform(data, T_c2aug, apply_to=self.prepare_data_keys)
            # if self.data['prev_exists']:
            #     self.memory['pose_no_aug'] = T_g2e @ self.memory['pose_no_aug']
            #     self.memory['ref_pts'] = self.transform_ref_pts(
            #         self.memory['ref_pts'], T_g2aug)
            # self.memory['pose'] = self.aug_transform @ self.memory['pose_no_aug']


            self.T_e2g[seq_idx] = T_e2g
            self.T_g2aug[seq_idx] = T_g2aug
            self.T_aug2g[seq_idx] = T_g2aug.inverse() # ego aug to global

        else:
            if self.is_ego:
                transform = torch.eye(4).to(lidar_pose.device)
            else:
                # cav to ego
                request = data['received_request']
                transform = request['lidar_poses'].inverse() @ lidar_pose

            T_c2aug = DOP.update_transform_with_aug(transform, data['augment_params'])
            DOP.apply_transform(data, T_c2aug, apply_to=self.prepare_data_keys)
            self.T_aug2g[seq_idx] = T_c2aug

    def prepare_data(self, seq_idx):
        data = self.data[seq_idx]
        # hash time
        azi = torch.arctan2(data['points'][:, 1], data['points'][:, 0])
        azi, inds = (torch.rad2deg(azi) + 180).floor().long().unique(return_inverse=True)
        times = torch.zeros_like(azi).float()
        torch_scatter.scatter_mean(data['points'][:, -1], inds, dim=0, out=times)
        data['time_scale'] = times
        data['time_scale_reduced'] = times - self.timestamp(seq_idx) / 2
        # self.data['points'] = self.data['points'][:, :-1]
        DOP.adaptive_free_space_augmentation(data, time_idx=-1)
        self.apply_transform(seq_idx)
        DOP.filter_range(data, self.lidar_range, apply_to=self.prepare_data_keys)

        # self.vis_data('transformed', 4)

    def get_response_cpm(self):
        cpm = {}
        for k in ['temp_fusion_feat']:
            if k in self.data[0]:
                cpm[k] = {i: d[k] for i, d in self.data.items()}
        return cpm

    def forward_local(self, tasks, training_mode, seq_idx):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.task_id(seq_idx), '01:pts_backbone', {}))
        tasks[grad_mode].append((self.task_id(seq_idx), '02:backbone_neck', {}))
        tasks[grad_mode].append((self.task_id(seq_idx), '03:roi_head', {}))
        tasks[grad_mode].append((self.task_id(seq_idx), '11:temporal_fusion', {}))
        tasks[grad_mode].append((self.task_id(seq_idx), '12:det1_head', {}))

    def forward_fusion(self, tasks, training_mode, seq_idx):
        if self.is_ego:
            tasks['with_grad'].append((self.task_id(seq_idx), '21:spatial_fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode, seq_idx):
        if self.is_ego:
            tasks['with_grad'].append((self.task_id(seq_idx), '22:det2_head', {}))
        return tasks

    def loss(self, tasks, seq_idx):
        if self.is_ego:
            tasks['loss'].append((self.task_id(seq_idx), '31:roi_head', {}))
            tasks['loss'].append((self.task_id(seq_idx), '32:det1_head', {}))
            tasks['loss'].append((self.task_id(seq_idx), '33:det2_head', {}))
        return tasks

    def init_memory(self, seq_idx, **kwargs):
        x = self.data[seq_idx]['prev_exists']
        init_pose = torch.eye(4, device=x.device).unsqueeze(0).unsqueeze(0)
        self.memory = {
            'embeddings': x.new_zeros(self.memory_len, self.memory_num_propagated, self.memory_emb_dims),
            'ref_pts': x.new_zeros(self.memory_len, self.memory_num_propagated, self.ref_pts_dim),
            'timestamp': x.new_zeros(self.memory_len, self.memory_num_propagated, 1),
            'pose': x.new_zeros(self.memory_len, self.memory_num_propagated, 4, 4),
            'pose_no_aug': x.new_zeros(self.memory_len, self.memory_num_propagated, 4, 4),
            'velo': x.new_zeros(self.memory_len, self.memory_num_propagated, 2),
        }
        self.memory['pose_no_aug'] = self.memory['pose'] + init_pose

    def pre_update_memory(self, seq_idx, **kwargs):
        """Update memory before each forward run of a single frame."""
        if seq_idx > 0:
            self.memory['timestamp'] += self.timestamp(seq_idx) / 2

        self.refresh_memory(self.data[seq_idx]['prev_exists'])

    def post_update_memory(self, seq_idx, **kwargs):
        """Update memory after each forward run of a single frame."""
        x = self.data[seq_idx]['detection_local']
        scores = x['all_cls_scores'][-1][...,
                 min(x['all_cls_scores'][-1].shape[-1] - 1, 1):].topk(1, dim=-1).values[..., 0]
        topk = torch.topk(scores, k=self.memory_num_propagated).indices

        ref_pts = x['all_bbox_preds'][-1][:, :self.ref_pts_dim]
        velo = x['all_bbox_preds'][-1][:, -2:]
        embeddings = self.data[seq_idx]['temp_fusion_feat']['outs_dec'][-1]
        # timestamp = torch.zeros_like(ref_pts[..., :1])
        # transform ref pts to coop coordinates
        transform = self.data[seq_idx]['lidar_poses'].inverse() @ self.T_aug2g[seq_idx]
        pts = self.transform_ref_pts(ref_pts, transform)

        # import matplotlib.pyplot as plt
        # from cosense3d.utils.vislib import draw_points_boxes_plt
        # pcd = self.transform_ref_pts(self.data['points'][:, :3], transform).detach().cpu().numpy()
        # vis_pts = pts.detach().cpu().numpy()
        # ax = draw_points_boxes_plt(
        #     pc_range=self.lidar_range.tolist(),
        #     points=pcd,
        #     return_ax=True,
        # )
        #
        # plt.plot(pts[:, 0], pts[:, 1], ".", markersize=2)
        # plt.show()
        # plt.close()

        timestamp = torch.rad2deg(torch.arctan2(pts[:, 1], pts[:, 0])) + 180
        timestamp = - self.data[seq_idx]['time_scale'][(timestamp % 360).floor().long()].unsqueeze(-1)
        pose_no_aug = torch.eye(4, device=ref_pts.device).unsqueeze(0).repeat(
            timestamp.shape[0], 1, 1)

        vars = locals()
        for k, v in self.memory.items():
            if k == 'prev_exists' or k == 'pose':
                continue
            rec_topk = vars[k][topk].unsqueeze(0)
            self.memory[k] = torch.cat([rec_topk, v], dim=0)

        # self.vis_ref_pts('post update')

        # ego aug to global
        self.memory['ref_pts'] = self.transform_ref_pts(
            self.memory['ref_pts'], self.T_aug2g[seq_idx])
        self.memory['timestamp'][1:] -= self.timestamp(seq_idx) / 2
        self.memory['pose_no_aug'] = self.T_e2g[seq_idx][(None,) * 2] @ self.memory['pose_no_aug'] # aug -->global

    def transform_ref_pts(self, reference_points, matrix):
        reference_points = torch.cat(
            [reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
        if reference_points.ndim == 3:
            reference_points = matrix.unsqueeze(0) @ reference_points.permute(0, 2, 1)
            reference_points = reference_points.permute(0, 2, 1)[..., :3]
        elif reference_points.ndim == 2:
            reference_points = matrix @ reference_points.T
            reference_points = reference_points.T[..., :3]
        else:
            raise NotImplementedError
        return reference_points

    def timestamp(self, seq_idx):
        if 'timestamp' in self.data[seq_idx]:
            timestamp = self.data[seq_idx]['timestamp']
        elif 'global_time' in self.data[seq_idx]:
            timestamp = self.data[seq_idx]['global_time']
        else:
            timestamp = float(self.data[seq_idx]['frame']) * 0.1
        return timestamp

    def vis_ref_pts(self, ax=None, label=None, his_len=1, **kwargs):
        import matplotlib.pyplot as plt
        from cosense3d.utils.vislib import draw_points_boxes_plt
        pcd = self.data['points'][:, :3].detach().cpu().numpy()
        gt_boxes = self.data['local_bboxes_3d'].detach().cpu().numpy()
        ax = draw_points_boxes_plt(
            pc_range=self.lidar_range.tolist(),
            boxes_gt=gt_boxes[:, :7],
            points=pcd,
            ax=ax,
            return_ax=True,
        )

        ref_pts = self.memory['ref_pts'].detach().cpu().numpy()
        markers = ['.r', '.m', '.b', '.c']
        for i in range(his_len):
            plt.plot(ref_pts[i, :, 0], ref_pts[i, :, 1], markers[i], markersize=2)
        ax.set_title(f"{label}: {self.data['scenario']}, {self.data['frame']}")

        return ax

    def vis_poses(self, ax=None, label=None, his_len=1, **kwargs):
        import matplotlib.pyplot as plt
        markers = ['r', 'm', 'b', 'c']
        mem_poses = self.memory['pose'][:, 0].detach().cpu()
        p0 = mem_poses[:his_len, :2, -1].numpy()
        p1 = mem_poses[:his_len] @ torch.tensor([1., 0., 0., 1.]).view(1, 4, 1).repeat(his_len, 1, 1)
        p2 = mem_poses[:his_len] @ torch.tensor([0., 1., 0., 1.]).view(1, 4, 1).repeat(his_len, 1, 1)
        p1 = p1.squeeze(-1)[:, :2].numpy()
        p2 = p2.squeeze(-1)[:, :2].numpy()

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.axis('equal')
        for i in range(his_len):
            ax.plot([p0[i, 0], p1[i, 0]], [p0[i, 1], p1[i, 1]], markers[i])
            ax.plot([p0[i, 0], p2[i, 0]], [p0[i, 1], p2[i, 1]], markers[i])
        return ax







