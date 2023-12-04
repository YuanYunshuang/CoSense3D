import torch
import torch.nn as nn

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.misc import SELayer_Linear, MLN
from cosense3d.modules.utils.positional_encoding import \
    pos2posemb2d, pos2posemb1d, nerf_positional_encoding


class TemporalFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 transformer,
                 feature_stride,
                 lidar_range,
                 topk=2048,
                 num_propagated=256,
                 memory_len=1024,
                 num_query=644,
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = plugin.build_plugin_module(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_pose_feat = 64
        self.pos_dim = 2
        self.in_channels = in_channels
        self.feature_stride = feature_stride
        self.topk = topk
        self.num_query = num_query
        self.num_propagated = num_propagated
        self.memory_len = memory_len

        self.lidar_range = nn.Parameter(torch.tensor(lidar_range), requires_grad=False)
        self.reference_points = nn.Embedding(self.num_query, self.pos_dim)

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

        self.pseudo_reference_points = nn.Embedding(self.num_propagated, self.pos_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        self.ego_pose_pe = MLN(180)
        self.ego_pose_memory = MLN(180)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.utils.init.xavier_uniform_(m)
        self._is_init = True

    def forward(self, rois, bev_feat, mem_dict, pose, **kwargs):
        feat, ctr = self.gather_topk(rois, bev_feat)

        pos = ((ctr - self.lidar_range[:2]) /
               (self.lidar_range[3:5] - self.lidar_range[:2]))
        pos_emb = self.position_embeding(pos2posemb2d(pos, self.num_pose_feat))
        memory = self.memory_embed(feat)
        pos_emb = self.featurized_pe(pos_emb, memory)

        reference_points = (self.reference_points.weight).unsqueeze(0).repeat(memory.shape[0], 1, 1)
        query_pos = self.query_embedding(pos2posemb2d(reference_points, self.num_pose_feat))
        tgt = torch.zeros_like(query_pos)

        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = \
            self.temporal_alignment(query_pos, tgt, reference_points, mem_dict, pose)
        outs_dec, _ = self.transformer(memory, tgt, query_pos, pos_emb, None, temp_memory, temp_pos)

        outs = [
            {
                'outs_dec': outs_dec[:, i],
                'ref_pts': reference_points[i],
            } for i in range(len(rois))
        ]

        return {self.scatter_keys[0]: outs}

    def format_input(self, input):
        memory = []
        for x in input:
            x = x.permute(0, 2, 3, 1).flatten(0, 2)
            memory.append(x)
        max_l = max([m.shape[0] for m in memory])
        out = x.new_zeros(len(memory), max_l, x.shape[-1])
        mask = x.new_ones(len(memory), max_l)
        for i, m in enumerate(memory):
            out[i, :len(m)] = m
            mask[i, :len(m)] = False
        return out, mask

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
        return topk_feat, topk_ctr

    def temporal_alignment(self, query_pos, tgt, ref_pts, mem_dict, pose):
        B = ref_pts.shape[0]
        if len(mem_dict['ref_pts']) == 0:
            temp_ref_pts = self.pseudo_reference_points.weight
            temp_ref_pts = temp_ref_pts.unsqueeze(0).repeat(B, 1, 1)
            temp_memory = ref_pts.new_zeros(B, self.memory_len, self.embed_dims)
            timestamp = ref_pts.new_zeros(B, self.memory_len, 1)
            pose = ref_pts.new_zeros(B, self.memory_len, 4, 4)
            velo = ref_pts.new_zeros(B, self.memory_len, 2)
        else:
            temp_ref_pts = ((mem_dict['ref_pts'] - self.lidar_range[:2]) /
                            (self.lidar_range[3:5] - self.lidar_range[:2]))
            temp_memory = mem_dict['embedding']
            timestamp = mem_dict['timestamp']
            pose = mem_dict['pose']
            velo = mem_dict['velo']

        temp_pos = self.query_embedding(pos2posemb2d(temp_ref_pts))
        rec_pose = torch.eye(
            4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(
            B, query_pos.size(1), 1, 1)

        # Get ego motion-aware tgt and query_pos for the current frame
        rec_motion = torch.cat(
            [torch.zeros_like(ref_pts[..., :3]),
             rec_pose[..., :3, :].flatten(-2)], dim=-1)
        rec_motion = nerf_positional_encoding(rec_motion)
        tgt = self.ego_pose_memory(tgt, rec_motion)
        query_pos = self.ego_pose_pe(query_pos, rec_motion)

        # get ego motion-aware reference points embeddings and memory for past frames
        memory_ego_motion = torch.cat(
            [self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
        memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
        temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
        temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        # get time-aware pos embeddings
        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(ref_pts[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
        query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
        ref_pts = torch.cat([ref_pts, temp_ref_pts[:, :self.num_propagated]], dim=1)
        rec_pose = torch.eye(
            4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(
            B, query_pos.shape[1] + self.num_propagated, 1, 1)
        temp_memory = temp_memory[:, self.num_propagated:]
        temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, ref_pts, temp_memory, temp_pos, rec_pose





