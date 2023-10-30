

class ProjectToEgo:

    def __init__(self, apply_transform=True):
        self.apply_transform = apply_transform

    def __call__(self, batch_dict):
        Ts_cav2ego = []
        for i, lidar_pose in enumerate(batch_dict['lidar_poses']):
            inds = batch_dict['points_inds'][i]
            ego_pose = batch_dict['ego_poses'][inds[0]][inds[1]]
            Ts_cav2ego.append(ego_pose.inv() @ lidar_pose)
            if self.apply_transform:
                raise NotImplementedError
                # batch_dict['lidar_poses'][i] = batch_dict['ego_pose']
        batch_dict['point_transforms']['proj2ego'] = Ts_cav2ego
        return batch_dict
