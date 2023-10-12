import torch


class FormatFrameData:

    def __init__(self):
        pass

    def __call__(self, batch_dict):
        point_inds = []
        img_inds = []

        for b, cavs in enumerate(batch_dict['valid_agent_ids']):
            for i, cav in enumerate(cavs):
                cam_ids = batch_dict[b][cav]
                point_inds.append([b, i, ])


class FormatSequenceData:

    def __init__(self):
        pass

    def __call__(self, batch_dict):
        point_inds = []
        img_inds = []

        for b, seqs in enumerate(batch_dict['valid_agent_ids']):
            for s, cavs in enumerate(seqs):
                for i, cav in enumerate(cavs):
                    cam_ids = batch_dict['chosen_cams'][b][s][cav]
                    point_inds.append([b, s, i])
                    img_inds.extend([[b, s, i] for _ in enumerate(cam_ids)])

        # cat img data
        for k in batch_dict.keys():
            if k not in ['scenario', 'frame', 'valid_agent_ids', 'chosen_cams', 'scene_tokens', 'prev_exists',
                         'ego_poses', 'global_bboxes_3d', 'global_labels_3d', 'global_names']:
                batch_dict[k] = self.cat_list(batch_dict[k])

        batch_dict['points_inds'] = torch.tensor(point_inds, device=batch_dict['points'][0].device)
        batch_dict['img_inds'] = torch.tensor(img_inds, device=batch_dict['img'][0].device)
        return batch_dict

    def cat_list(self, ls):
        if not isinstance(ls, list):
            return ls
        ls_cat = []
        for l in ls:
            if isinstance(l, list):
                ls_cat.extend(self.cat_list(l))
            elif isinstance(l, str):
                return [ls]
            else:
                return ls
        return ls_cat

