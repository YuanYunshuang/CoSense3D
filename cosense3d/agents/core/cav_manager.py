import torch


class CAVManager:
    def __init__(self, max_num_cavs=10):
        self.cavs = []
        self.cav_dict = {}

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(cavs={self.cav_dict.keys()})'
        return repr_str

    def update_cav_info(self, valid_agent_ids=None, lidar_poses=None, intrinsics=None, extrinsics=None, **data):
        B = len(valid_agent_ids)  # batch_size
        del self.cavs
        self.cavs = []
        for b in range(B):
            batch_cavs = []
            for i, cav_id in enumerate(valid_agent_ids[b]):
                is_ego = True if i==0 else False  # assume the first car is ego car
                batch_cavs.append(CAV(cav_id, i, is_ego,
                                      lidar_pose=lidar_poses[b][i],
                                      cam_intrinsic=intrinsics[b][i] if intrinsics is not None else None,
                                      cam_extrinsic=extrinsics[b][i] if extrinsics is not None else None,))
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
                    req[cav.id] = {'lidar_pose': cav.lidar_pose}
            request.append(req)
        return request

    def receive_request(self, request):
        for b, req in enumerate(request):
            for ai, req_cpm in req.items():
                for cav in self.cavs[b]:
                    if ai != cav.id:
                        cav.received_request = req_cpm

    def send_response(self):
        response = []
        for b, cavs in enumerate(self.cavs):
            ans = {}
            for cav in cavs:
                if cav.answer is not None:
                    ans[cav.id] = cav.answer
            response.append(ans)
        return response

    def forward(self):
        tasks = {'with_grad': [], 'no_grad': []}
        for i, cavs in enumerate(self.cavs):
            for cav in cavs:
                cav.forward_local(tasks)
                cav.forward_fusion(tasks)
                cav.forward_head(tasks)
        return tasks

class CAV:
    def __init__(self, id=None, mapped_id=None, is_ego=False,
                 pose=None, lidar_pose=None, cam_extrinsic=None, cam_intrinsic=None):
        self.id = id
        self.mapped_id = mapped_id
        self.is_ego = is_ego
        self.pose = pose
        self.lidar_pose = lidar_pose
        self.cam_extrinsic = cam_extrinsic
        self.cam_intrinsic = cam_intrinsic
        self.data = {}
        self.received_request = None
        self.received_cpm = None
        self.local_features = None
        self.fused_features = None

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
            transform = self.received_request['lidar_pose'].inverse() @ self.lidar_pose
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

    def forward_local(self, tasks):
        self.apply_data_transform()
        tasks['no_grad'].append((self.id, '2:filter_range', {}))
        if self.is_ego:
            tasks['with_grad'].append((self.id, '3:pts_backbone', {}))
        else:
            tasks['no_grad'].append((self.id, '3:pts_backbone', {}))

    def forward_fusion(self, tasks):
        if self.is_ego:
            data = {'ego': self.local_features, 'cpm': self.received_cpm}
            tasks['with_grad'].append((self.id, '4:fusion', data))
            tasks['with_grad'].append((self.id, '5:coor_dilation', {}))
        return tasks

    def forward_head(self, tasks):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '6:bev_head', {}))
            tasks['with_grad'].append((self.id, '7:detection_head', {}))
        return tasks


