import torch
import torch.nn as nn

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.misc import SELayer_Linear, MLN
import cosense3d.modules.utils.positional_encoding as PE


class TemporalFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 transformer,
                 feature_stride,
                 lidar_range,
                 pos_dim=3,
                 num_pose_feat=64,
                 topk=2048,
                 num_propagated=256,
                 memory_len=1024,
                 num_query=644,
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = plugin.build_plugin_module(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_pose_feat = num_pose_feat
        self.pos_dim = pos_dim
        self.in_channels = in_channels
        self.feature_stride = feature_stride
        self.topk = topk
        self.num_query = num_query
        self.num_propagated = num_propagated
        self.memory_len = memory_len

        self.lidar_range = nn.Parameter(torch.tensor(lidar_range), requires_grad=False)

        self._init_layers()

    def _init_layers(self):
        self.position_embeding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.memory_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # can be replaced with MLN
        self.featurized_pe = SELayer_Linear(self.embed_dims)

        self.reference_points = nn.Embedding(self.num_query, self.pos_dim)
        self.pseudo_reference_points = nn.Embedding(self.num_propagated, self.pos_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        pose_nerf_dim = (3 + 3 * 4) * 12
        self.ego_pose_pe = MLN(pose_nerf_dim, f_dim=self.embed_dims)
        self.ego_pose_memory = MLN(pose_nerf_dim, f_dim=self.embed_dims)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.utils.init.xavier_uniform_(m)
        self._is_init = True

    def forward(self, rois, bev_feat, mem_dict, **kwargs):
        feat, ctr = self.gather_topk(rois, bev_feat)

        pos = ((ctr - self.lidar_range[:self.pos_dim]) /
               (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        pos_emb = self.position_embeding(self.embed_pos(pos))
        memory = self.memory_embed(feat)
        pos_emb = self.featurized_pe(pos_emb, memory)

        reference_points = self.reference_points.weight.unsqueeze(0).repeat(memory.shape[0], 1, 1)
        query_pos = self.query_embedding(self.embed_pos(reference_points))
        tgt = torch.zeros_like(query_pos)

        tgt, query_pos, reference_points, temp_memory, temp_pos = \
            self.temporal_alignment(query_pos, tgt, reference_points, mem_dict)
        mask_dict = [None, None]
        outs_dec, _ = self.transformer(memory, tgt, query_pos, pos_emb,
                                       mask_dict, temp_memory, temp_pos)

        outs = [
            {
                'outs_dec': outs_dec[:, i],
                'ref_pts': reference_points[i],
            } for i in range(len(rois))
        ]

        return {self.scatter_keys[0]: outs}

    def gather_topk(self, rois, bev_feats):
        topk_feat, topk_ctr = [], []
        for roi, bev_feat in zip(rois, bev_feats):
            ctr = bev_feat[f'p{self.feature_stride}']['ctr']
            feat = bev_feat[f'p{self.feature_stride}']['feat']
            scores = roi['scr']
            if scores.shape[0] < self.topk:
                raise NotImplementedError
            else:
                topk_inds = torch.topk(scores, k=self.topk).indices
                topk_ctr.append(ctr[topk_inds])
                topk_feat.append(feat[topk_inds])
        topk_ctr = torch.stack(topk_ctr, dim=0)
        topk_feat = torch.stack(topk_feat, dim=0)
        if topk_ctr.shape[-1] < self.pos_dim:
            pad_dim = self.pos_dim - topk_ctr.shape[-1]
            topk_ctr = torch.cat([topk_ctr, torch.zeros_like(topk_ctr[..., :pad_dim])], dim=-1)
        return topk_feat, topk_ctr

    def embed_pos(self, pos, dim=None):
        dim = self.num_pose_feat if dim is None else dim
        return getattr(PE, f'pos2posemb{pos.shape[-1]}d')(pos, dim)

    def temporal_alignment(self, query_pos, tgt, ref_pts, mem_dict):
        B = ref_pts.shape[0]
        mem_dict = self.stack_dict_list(mem_dict)
        x = mem_dict['prev_exists'].view(-1)
        # metric coords --> normalized coords
        temp_ref_pts = ((mem_dict['ref_pts'] - self.lidar_range[:self.pos_dim]) /
                        (self.lidar_range[3:3+self.pos_dim] - self.lidar_range[:self.pos_dim]))
        if not x.all():
            # pad the recent memory ref pts with pseudo points
            pseudo_ref_pts = self.pseudo_reference_points.weight.unsqueeze(0).repeat(B, 1, 1)
            x = x.view(*((-1,) + (1,) * (pseudo_ref_pts.ndim - 1)))
            temp_ref_pts[:, 0] = temp_ref_pts[:, 0] * x + pseudo_ref_pts * (1 - x)

        temp_pos = self.query_embedding(self.embed_pos(temp_ref_pts))
        temp_memory = mem_dict['embeddings']
        rec_pose = torch.eye(
            4, device=query_pos.device).reshape(1, 1, 4, 4).repeat(
            B, query_pos.size(1), 1, 1)

        # Get ego motion-aware tgt and query_pos for the current frame
        rec_motion = torch.cat(
            [torch.zeros_like(tgt[..., :3]),
             rec_pose[..., :3, :].flatten(-2)], dim=-1)
        rec_motion = PE.nerf_positional_encoding(rec_motion)
        tgt = self.ego_pose_memory(tgt, rec_motion)
        query_pos = self.ego_pose_pe(query_pos, rec_motion)

        # get ego motion-aware reference points embeddings and memory for past frames
        memory_ego_motion = torch.cat(
            [mem_dict['velo'], mem_dict['timestamp'],
             mem_dict['pose'][..., :3, :].flatten(-2)], dim=-1).float()
        memory_ego_motion = PE.nerf_positional_encoding(memory_ego_motion)
        temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
        temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        # get time-aware pos embeddings
        query_pos += self.time_embedding(
            self.embed_pos(torch.zeros_like(ref_pts[..., :1]), self.embed_dims))
        temp_pos += self.time_embedding(
            self.embed_pos(mem_dict['timestamp'], self.embed_dims).float())

        tgt = torch.cat([tgt, temp_memory[:, 0]], dim=1)
        query_pos = torch.cat([query_pos, temp_pos[:, 0]], dim=1)
        ref_pts = torch.cat([ref_pts, temp_ref_pts[:, 0]], dim=1)
        # rec_pose = torch.eye(
        #     4, device=query_pos.device).reshape(1, 1, 4, 4).repeat(
        #     B, query_pos.shape[1] + temp_pos[:, 0].shape[1], 1, 1)
        temp_memory = temp_memory[:, 1:].flatten(1, 2)
        temp_pos = temp_pos[:, 1:].flatten(1, 2)

        return tgt, query_pos, ref_pts, temp_memory, temp_pos





