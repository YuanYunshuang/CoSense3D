import torch


class CAVManager:
    def __init__(self, lidar_range, max_num_cavs=10, **kwargs):
        self.lidar_range = torch.tensor(lidar_range)
        self.cavs = []
        self.cav_dict = {}

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(cavs={self.cav_dict.keys()})'
        return repr_str

    def update_cav_info(self, valid_agent_ids=None, lidar_poses=None, **data):
        B = len(valid_agent_ids)  # batch_size
        del self.cavs
        self.cavs = []
        for b in range(B):
            batch_cavs = []
            for i, cav_id in enumerate(valid_agent_ids[b]):
                is_ego = True if i==0 else False  # assume the first car is ego car
                batch_cavs.append(CAV(cav_id, i, is_ego, lidar_poses[b][i], self.lidar_range))
                self.cav_dict[cav_id] = (b, i)
            self.cavs.append(batch_cavs)

    def get_cav_with_id(self, id):
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

    def reset_cpm_memory(self):
        for b, cavs in enumerate(self.cavs):
            for cav in cavs:
                cav.reset_cpm_memory()

    def forward(self):
        tasks = {'with_grad': [], 'no_grad': []}
        for i, cavs in enumerate(self.cavs):
            for cav in cavs:
                cav.forward_local(tasks)
                cav.forward_fusion(tasks)
                cav.forward_head(tasks)
        return tasks


class CAV:
    def __init__(self, id, mapped_id, is_ego, lidar_pose, lidar_range):
        self.id = id
        self.mapped_id = mapped_id
        self.is_ego = is_ego
        self.lidar_pose = lidar_pose
        self.lidar_range = lidar_range
        self.data = {}

    def reset(self, id, is_ego, mapped_id):
        self.id = id
        self.is_ego = is_ego
        self.mapped_id = mapped_id

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(id={self.id}, '
        repr_str += f'is_ego={self.is_ego}, '
        repr_str += f'data={self.data.keys()})'
        return repr_str

    def apply_data_transform(self):
        if self.is_ego:
            transform = torch.eye(4).to(self.lidar_pose.device)
        else:
            # cav to ego
            request = self.data['received_request']
            transform = request['lidar_pose'].inverse() @ self.lidar_pose
        # augmentation
        if 'rot' in self.data['augment_params']:
            transform = self.data['augment_params']['rot'].to(transform.device) @ transform

        C = self.data['points'].shape[-1]
        points = self.data['points'][:, :3]
        points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1).T
        points_homo = transform @ points_homo

        if 'scale' in self.data['augment_params']:
            points_homo[:2] *= self.data['augment_params']['scale'].item()

        if C > 3:
            self.data['points'] = torch.cat([points_homo[:3].T,
                                             self.data['points'][:, 3:]], dim=-1)
        else:
            self.data['points'] = points_homo[:3].T

    def filter_data_range(self):
        points = self.data['points']
        lr = self.lidar_range.to(points.device)
        mask = (points[:, :3] > lr[:3].view(1, 3)) & (points[:, :3] < lr[3:].view(1, 3))
        self.data['points'] = points[mask.all(dim=-1)]

    def has_request(self):
        if 'received_request' in self.data and self.data['received_request'] is not None:
            return True
        else:
            return False

    def get_request_cpm(self):
        return {'lidar_pose': self.lidar_pose}

    def get_response_cpm(self):
        cpm = {}
        for k in ['pts_feat']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def receive_request(self, request):
        self.data['received_request'] = request

    def receive_response(self, response):
        self.data['received_response'] = response

    def forward_local(self, tasks):
        self.apply_data_transform()  # 1
        self.filter_data_range()  # 2
        if self.is_ego:
            tasks['with_grad'].append((self.id, '3:pts_backbone', {}))
        else:
            tasks['no_grad'].append((self.id, '3:pts_backbone', {}))

    def forward_fusion(self, tasks):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '4:fusion', {}))
            tasks['with_grad'].append((self.id, '5:fusion_neck', {}))
        return tasks

    def forward_head(self, tasks):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '6:bev_head', {}))
            tasks['with_grad'].append((self.id, '7:detection_head', {}))
        return tasks

    def reset_cpm_memory(self):
        self.data.pop('received_request')
        self.data.pop('received_response')


