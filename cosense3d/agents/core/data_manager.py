import random

import torch
import torch_scatter

from cosense3d.ops.utils import points_in_boxes_gpu


class DataManager:
    def __init__(self, cav_manager, lidar_range, voxel_size=None, aug=None, pre_process=[]):
        self.cav_manager = cav_manager
        self.lidar_range = lidar_range
        self.voxel_size = voxel_size
        self.aug = aug
        self.pre_process = pre_process

    def apply_preprocess(self):
        if isinstance(self.pre_process, list):
            for p in self.pre_process:
                getattr(self, p)()
        elif isinstance(self.pre_process, dict):
            for p, args in self.pre_process.items():
                getattr(self, p)(**args)

    def remove_empty_boxes(self):
        for cavs in self.cav_manager.cavs:
            points = torch.cat([cav.data['points'] for cav in cavs], dim=0)
            assert cavs[0].is_ego
            global_boxes = cavs[0].data['global_bboxes_3d']
            box_idx = points_in_boxes_gpu(points.unsqueeze(0)[..., :3], global_boxes.unsqueeze(0))[0]
            box_idx = box_idx[box_idx > -1]
            num_pts = torch.zeros_like(global_boxes[:, 0]).long()
            torch_scatter.scatter_add(torch.ones_like(box_idx), box_idx, dim=0, out=num_pts)
            mask = num_pts > 3
            cavs[0].data['global_bboxes_3d'] = global_boxes[mask]
            cavs[0].data['global_labels_3d'] = cavs[0].data['global_labels_3d'][mask]

    def remove_global_empty_boxes(self):
        self.remove_empty_boxes()

    def remove_local_empty_boxes(self, ego_only=False):
        for cavs in self.cav_manager.cavs:
            for cav in cavs:
                if not cav.is_ego and ego_only:
                    continue
                points = cav.data['points']
                local_boxes = cav.data['local_bboxes_3d']
                box_idx = points_in_boxes_gpu(points.unsqueeze(0)[..., :3], local_boxes.unsqueeze(0))[0]
                box_idx = box_idx[box_idx > -1]
                num_pts = torch.zeros_like(local_boxes[:, 0]).long()
                torch_scatter.scatter_add(torch.ones_like(box_idx), box_idx, dim=0, out=num_pts)
                mask = num_pts > 3
                cav.data['local_bboxes_3d'] = local_boxes[mask]
                cav.data['local_labels_3d'] = cav.data['local_labels_3d'][mask]

    def distribute_to_seq_list(self, batch_dict, seq_len):
        result = []
        for l in range(seq_len):
            res = {}
            for k, v in batch_dict.items():
                res[k] = [x[l] for x in v]
            result.append(res)
        return result

    def distribute_to_cav(self, valid_agent_ids=None, **data):
        cavs = self.cav_manager.cavs
        global_data_list = []
        for b, agent_ids in enumerate(valid_agent_ids):
            global_data = {}
            for j, ai in enumerate(agent_ids):
                assert cavs[b][j].id == f'{b}.{ai}'
                for k, v in data.items():
                    if isinstance(v[b], list) and len(v[b]) == len(agent_ids):
                        cavs[b][j].data[k] = v[b][j]
                    elif k == 'chosen_cams':
                        cavs[b][j].data[k] = v[b][ai]
                    elif k == 'augment_params':
                        cavs[b][j].data[k] = v[b]
                        global_data[k] = v[b]
                    elif cavs[b][j].is_ego:
                        cavs[b][j].data[k] = v[b]
            global_data_list.append(global_data)
        return global_data_list

    def generate_augment_params(self, batch_dict, seq_len):
        B = len(batch_dict['scenario'])
        if self.aug is None:
            rand_aug = [[None] * seq_len] * B
        else:
            rand_aug = []
            def rand_from_range(r):
                return torch.rand(1) * (r[1] - r[0]) + r[0]
            for i in range(B):
                cur_aug = {}
                if 'rot_range' in self.aug:
                    theta = rand_from_range(self.aug['rot_range'])
                    ct = torch.cos(theta)
                    st = torch.sin(theta)
                    transform = torch.eye(4)
                    transform[0, 0] = ct
                    transform[0, 1] = -st
                    transform[1, 0] = st
                    transform[1, 1] = ct
                    cur_aug['rot'] = transform
                if 'trans_std' in self.aug:
                    cur_aug['trans'] = torch.randn(len(self.aug['trans_std'])) * torch.tensor(self.aug['trans_std'])
                if 'scale_ratio_range' in self.aug:
                    cur_aug['scale'] = rand_from_range(self.aug['scale_ratio_range'])
                if 'flip' in self.aug:
                    cur_aug['flip'] = {'flip_idx': random.randint(0, 3), 'flip_axis': self.aug['flip']}
                rand_aug.append([cur_aug for _ in range(seq_len)])
        batch_dict['augment_params'] = rand_aug

    def gather(self, cav_list, data_keys):
        data_list = []
        for k in data_keys:
            data = []
            for cav_id in cav_list:
                data.append(self.cav_manager.get_cav_with_id(cav_id).data[k])
            data_list.append(data)
        return data_list

    def scatter(self, cav_list, data_dict):
        for k, data_list in data_dict.items():
            for cav_id, data in zip(cav_list, data_list):
                self.update(cav_id, k, data)

    def update(self, cav_id, data_key, data):
        self.cav_manager.get_cav_with_id(cav_id).data[data_key] = data

    def gather_batch(self, batch_idx, key, to_numpy=False):
        data = {}
        for cav in self.cav_manager.cavs[batch_idx]:
            if key not in cav.data:
                continue
            d = cav.data[key]
            if isinstance(d, torch.Tensor) and to_numpy:
                d = d.cpu().numpy()
            elif isinstance(d, list) and len(d) > 0 and isinstance(d[0], torch.Tensor):
                d = [x.cpu().numpy() for x in d]
            data[cav.id] = d
        return data

    def gather_ego_data(self, key):
        data = {}
        for cavs in self.cav_manager.cavs:
            assert cavs[0].is_ego
            if key not in cavs[0].data:
                continue
            d = cavs[0].data[key]
            data[cavs[0].id] = d
        return data

    def boxes_to_vis_format(self, boxes, labels):
        boxes_vis = {}
        gt_labels = labels.tolist()
        for i, box in enumerate(boxes.tolist()):
            boxes_vis[i] = [gt_labels[i]] + box[:6] + [0, 0] + [box[6]]
        return boxes_vis

    def get_gt_boxes_as_vis_format(self, batch_idx, coor='global'):
        gt_boxes = self.gather_batch(batch_idx, f'{coor}_bboxes_3d' )
        gt_labels = self.gather_batch(batch_idx, f'{coor}_labels_3d')
        labels = {}
        for k in gt_boxes.keys():
            labels[k] = self.boxes_to_vis_format(gt_boxes[k], gt_labels[k])
        return labels

    def gather_vis_data(self, batch_idx=0, keys=['points']):
        gather_dict = {}
        for k in keys:
            if k in ['global_labels', 'local_labels']:
                ref_coor = k.split('_')[0]
                gather_dict[f'{ref_coor}_labels'] = (
                    self.get_gt_boxes_as_vis_format(batch_idx, ref_coor))
            elif 'detection' in k:
                detection = self.gather_ego_data(k)
                for cav_id, det in detection.items():
                    if 'preds' in det:
                        det = det['preds'] # todo: without nms hook, keywork preds is not removed
                    detection[cav_id]['labels'] = self.boxes_to_vis_format(det['box'], det['lbl'])
                gather_dict[k] = detection
            else:
                gather_dict[k] = self.gather_batch(batch_idx, k, True)
        return gather_dict

    def get_vis_data_input(self, batch_idx=0, keys=None):
        """

        Parameters
        ----------
        batch_idx
        key: additional gt keys that are not standarlized in consense3d data API

        Returns
        -------

        """
        pcds = self.gather_batch(batch_idx, 'points', True)
        imgs = self.gather_batch(batch_idx, 'img', True)
        global_labels = self.get_gt_boxes_as_vis_format(batch_idx, 'global')
        local_labels = self.get_gt_boxes_as_vis_format(batch_idx, 'local')
        bboxes2d = self.gather_batch(batch_idx, 'bboxes2d', True)
        lidar2img = self.gather_batch(batch_idx, 'lidar2img', True)
        out_dict = {
            'pcds': pcds,
            'imgs': imgs,
            'bboxes2d': bboxes2d,
            'lidar2img': lidar2img,
            'global_labels': global_labels,
            'local_labels': local_labels
        }
        if keys is not None:
            for k in keys:
                out_dict[k] = self.gather_batch(batch_idx, k, True)
        return out_dict

    def get_vis_data_detection(self, batch_idx=0, keys='detection'):
        """

        Parameters
        ----------
        batch_idx: batch index
        key: the default key for detection is 'detection', customized key can also be used,
        depending on which key is used for saving detection result in the CAV data pool.

        Returns
        -------
            detection: result with boxes and labels converted to the visualizing format.
        """
        detection = self.gather_batch(batch_idx, 'detection')
        for cav_id, det in detection.items():
            detection[cav_id]['labels'] = self.boxes_to_vis_format(det['box'], det['lbl'])
        return detection

    def get_vis_data_bev(self, batch_idx=0, keys='bev'):
        return self.gather_batch(batch_idx, 'bev')

    def get_vis_data_meta(self, batch_idx=0, keys=None):
        return {
            'scenario': self.gather_batch(batch_idx, 'scenario'),
            'frame': self.gather_batch(batch_idx, 'frame')
        }



