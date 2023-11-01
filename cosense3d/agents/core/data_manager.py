import torch

from cosense3d.model.pre_process import PreProcess
from cosense3d.model.post_process import PostProcess


class DataManager:
    def __init__(self, cav_manager, data_info, aug=None, post_process=None):
        self.cav_manager = cav_manager
        self.data_info = data_info
        self.aug = aug
        if post_process is not None:
            self.postP = PostProcess(post_process)

    def post_process(self, batch_dict):
        if not hasattr(self, 'postP'):
            return batch_dict
        else:
            return self.postP(batch_dict)

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
                assert cavs[b][j].id == ai
                for k, v in data.items():
                    if isinstance(v[b], list) and len(v[b]) == len(agent_ids):
                        cavs[b][j].data[k] = v[b][j]
                    elif k == 'chosen_cams':
                        cavs[b][j].data[k] = v[b][ai]
                    elif k == 'augment_params':
                        cavs[b][j].data[k] = v[b]
                        global_data[k] = v[b]
                    else:
                        global_data[k] = v[b]
            global_data_list.append(global_data)
        return global_data_list

    def generate_augment_params(self, batch_dict, seq_len):
        if self.aug is None:
            rand_aug = None
        else:
            B = len(batch_dict['scenario'])
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




