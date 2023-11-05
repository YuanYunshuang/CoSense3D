import torch

from cosense3d.agents.cav_prototype import get_prototype

class CAVManager:
    def __init__(self, lidar_range, prototype=None, memory_len=1, **kwargs):
        self.lidar_range = torch.tensor(lidar_range)
        self.memory_len = memory_len
        self.cavs = []
        self.cav_dict = {}
        assert prototype is not None, "CAV prototype should be defined."
        self.prototype = get_prototype(prototype)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(cavs={self.cav_dict.keys()})'
        return repr_str

    def reset(self):
        self.cavs = []
        self.cav_dict = {}

    def update_cav_info(self, valid_agent_ids=None, lidar_poses=None, **data):
        B = len(valid_agent_ids)  # batch_size
        cavs = []
        cav_dict = {}
        for b in range(B):
            batch_cavs = []
            for i, cav_id in enumerate(valid_agent_ids[b]):
                is_ego = True if i==0 else False  # assume the first car is ego car
                # pad id with batch idx to avoid duplicated ids across different batches
                cav_id = f'{b}.{cav_id}'
                cav = self.get_cav_with_id(cav_id)
                if not cav:
                    cav = self.prototype(cav_id, i, is_ego, lidar_poses[b][i],
                                         self.lidar_range, self.memory_len)
                else:
                    cav.update(lidar_poses[b][i])
                batch_cavs.append(cav)
                cav_dict[cav_id] = (b, i)
            if sum([int(cav.is_ego) for cav in batch_cavs]) > 1:
                print('d')
            cavs.append(batch_cavs)
        self.cavs = cavs
        self.cav_dict = cav_dict

    def has_cav(self, cav_id):
        return cav_id in self.cav_dict

    def get_cav_with_id(self, id):
        if id not in self.cav_dict:
            return False
        b, i = self.cav_dict[id]
        return self.cavs[b][i]

    def send_request(self):
        request = []
        for b, cavs in enumerate(self.cavs):
            req = {}
            for cav in cavs:
                if cav.is_ego:
                    req[cav.id] = cav.get_request_cpm()
            request.append(req)
        return request

    def receive_request(self, request):
        for b, req in enumerate(request):
            for ai, req_cpm in req.items():
                for cav in self.cavs[b]:
                    if ai != cav.id:
                        cav.receive_request(req_cpm)

    def send_response(self):
        response = []
        for b, cavs in enumerate(self.cavs):
            ans = {}
            for cav in cavs:
                if cav.has_request():
                    ans[cav.id] = cav.get_response_cpm()
            response.append(ans)
        return response

    def receive_response(self, response):
        for cavs, resp in zip(self.cavs, response):
            for cav in cavs:
                if cav.is_ego:
                    cav.receive_response(resp)

    def forward(self, with_loss, training_mode):
        tasks = {'with_grad': [], 'no_grad': [], 'loss': []}
        for i, cavs in enumerate(self.cavs):
            for cav in cavs:
                cav.forward_local(tasks, training_mode)
                cav.forward_fusion(tasks, training_mode)
                cav.forward_head(tasks, training_mode)
                if with_loss:
                    cav.loss(tasks)
        return tasks

    def apply_cav_function(self, func_name):
        for b, cavs in enumerate(self.cavs):
            for cav in cavs:
                getattr(cav, func_name)()





