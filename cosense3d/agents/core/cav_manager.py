import torch

from cosense3d.agents.cav_prototype import get_prototype

class CAVManager:
    def __init__(self, lidar_range, prototype=None, memory_len=1, all_grad=False, num_grad_cav=1, seq_len=0, **kwargs):
        self.lidar_range = torch.tensor(lidar_range)
        self.memory_len = memory_len
        self.all_grad = all_grad
        self.num_grad_cav = num_grad_cav
        self.seq_len = seq_len
        self.kwargs = kwargs
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
                require_grad = True if (i < self.num_grad_cav or self.all_grad) else False
                # pad id with batch idx to avoid duplicated ids across different batches
                cav_id = f'{b}.{cav_id}'
                cav = self.get_cav_with_id(cav_id)
                if not cav:
                    cav = self.prototype(cav_id, i, is_ego,
                                         self.lidar_range,
                                         self.memory_len,
                                         lidar_pose=lidar_poses[b][i],
                                         require_grad=require_grad,
                                         **self.kwargs)
                else:
                    cav.update(lidar_poses[b][i])
                batch_cavs.append(cav)
                cav_dict[cav_id] = (b, i)
            cavs.append(batch_cavs)
        self.cavs = cavs
        self.cav_dict = cav_dict

    def has_cav(self, cav_id):
        return cav_id in self.cav_dict

    def get_cav_with_id(self, id):
        if id not in self.cav_dict:
            return False
        item = self.cav_dict[id]
        if isinstance(item, tuple):
            b, i = item
            return self.cavs[b][i]
        else:
            return item

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
                cav.forward(tasks, training_mode)
                if with_loss and training_mode:
                    cav.loss(tasks)
        return tasks

    def apply_cav_function(self, func_name):
        for b, cavs in enumerate(self.cavs):
            for cav in cavs:
                getattr(cav, func_name)()


class SeqCAVManager:
    def __init__(self, lidar_range, prototype=None, memory_len=1, all_grad=False, num_grad_cav=1, seq_len=0, **kwargs):
        self.lidar_range = torch.tensor(lidar_range)
        self.memory_len = memory_len
        self.all_grad = all_grad
        self.num_grad_cav = num_grad_cav
        self.seq_len = seq_len
        self.kwargs = kwargs
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

    def update_cav_info(self, data):
        seq_len = len(data)
        valid_agent_ids = [data[l]['valid_agent_ids'] for l in range(seq_len)]
        B = len(data[0]['valid_agent_ids'])  # batch_size
        uniq_ids = set(x for d in data for cids in d['valid_agent_ids'] for x in cids)
        cavs = []
        cav_dict = {}
        for l, d in enumerate(data):
            seq_cavs = []
            for b in range(B):
                batch_cavs = []
                for i, cav_id in enumerate(d['valid_agent_ids'][b]):
                    is_ego = True if i == 0 else False  # assume the first car is ego car
                    require_grad = True if (i < self.num_grad_cav or self.all_grad) else False
                    # pad id with batch idx to avoid duplicated ids across different batches
                    cav_id = f'{b}.{cav_id}'
                    cav = self.get_cav_with_id(cav_id)
                    if not cav:
                        cav = self.prototype(cav_id, i, is_ego,
                                             self.lidar_range,
                                             self.memory_len,
                                             require_grad=require_grad,
                                             seq_len=seq_len,
                                             **self.kwargs)
                        cav_dict[cav_id] = cav
                    batch_cavs.append(cav_id)
                seq_cavs.append(batch_cavs)
            cavs.append(seq_cavs)
        self.cavs = cavs
        self.cav_dict = cav_dict

    def has_cav(self, cav_id):
        return cav_id in self.cav_dict

    def get_cav_with_id(self, id):
        if id not in self.cav_dict:
            return False
        item = self.cav_dict[id]
        if isinstance(item, tuple):
            b, i = item
            return self.cavs[b][i]
        else:
            return item

    def send_request(self):
        request = {}
        for cav_id, cav in self.cav_dict.items():
            if cav.is_ego:
                request[cav.id] = cav.get_request_cpm()
        return request

    def receive_request(self, request):
        for req_id, req_cpm in request.items():
            for recv_id, cav in self.cav_dict.items():
                if recv_id.split('.')[0] == req_id.split('.')[0] and req_id != recv_id:
                    cav.receive_request(req_cpm)

    def send_response(self):
        response = {}
        for cav_id, cav in self.cav_dict.items():
            if cav.has_request():
                response[cav_id] = cav.get_response_cpm()
        return response

    def receive_response(self, response):
        for recv_id, cav in self.cav_dict.items():
            if cav.is_ego:
                for i in range(self.seq_len):
                    cav.data[i]['received_response'] = {}
                for resp_id, resp in response.items():
                    if recv_id.split('.')[0] == resp_id.split('.')[0]:
                        cav.receive_response({resp_id: resp})

    def forward(self, training_mode, num_loss_frame):
        tasks = {'with_grad': [], 'no_grad': [], 'loss': []}
        for i in range(self.seq_len):
            with_loss = i >= self.seq_len - num_loss_frame
            for id, cav in self.cav_dict.items():
                if i not in cav.data:
                    continue
                cav.forward(tasks, training_mode, i)
                if with_loss and training_mode:
                    cav.loss(tasks, i)
        return tasks

    def apply_cav_function(self, func_name, seq_idx, **kwargs):
        for cav_id, cav in self.cav_dict.items():
            if seq_idx not in cav.data:
                continue
            getattr(cav, func_name)(seq_idx=seq_idx, **kwargs)





