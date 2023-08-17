import numpy as np
import torch
import torch.nn.functional as F
from cosense3d.dataset.base_dataset import BaseDataset


class EarlyFusionDataset(BaseDataset):

    def __init__(self, cfgs, mode):
        super().__init__(cfgs, mode)

    @staticmethod
    def collate_batch(data_list):
        ret = {
            'batch_size': len(data_list),
            # data
            'pcds': [],
            'features': [],
            'coords': [],
            'imgs': [],
            # meta
            'scenario': [],
            'frame': [],
            'device_ids': [],
            'objects': [],
            'cam_intrinsics': [],
            'cam_extrinsics': [],
            'lidar_poses': None,
            'map_anchors': None,
            'anchor_offsets': None,
        }

        # extensions
        if data_list[0]['bev_maps'] is not None:
            for k in data_list[0]['bev_maps'].keys():
                ret[f"map_{k}"] = []

        for i in range(len(data_list)):
            # all data in the same frame have the same batch index for early fusion
            if data_list[i]['pcds'] is not None:
                pcds = data_list[i]['pcds']
                pcds[:, 0] = i
                features = data_list[i]['features']
                coords = data_list[i]['coords']
                coords = torch.from_numpy(coords).float()
                coords = F.pad(coords, (1, 0, 0, 0), mode="constant", value=i)
                ret['pcds'].append(torch.from_numpy(pcds).float())
                ret['features'].append(torch.from_numpy(features).float())
                ret['coords'].append(coords)

            objects = data_list[i]['objects']
            if objects is not None and len(objects) > 0:
                objects = torch.from_numpy(objects).float()
                objects = F.pad(objects, (1, 0, 0, 0), mode="constant", value=i)
            else:
                objects = torch.zeros((0, 12))
            ret['objects'].append(objects)

            if data_list[i]['imgs'] is not None:
                ret['imgs'].append(torch.from_numpy(data_list[i]['imgs']).float().unsqueeze(0))
                cam_intrinsic = np.array(data_list[i]['cam_params']['intrinsic'])
                cam_extrinsic = np.array(data_list[i]['cam_params']['extrinsic'])
                ret['cam_intrinsics'].append(torch.from_numpy(cam_intrinsic).float().unsqueeze(0))
                ret['cam_extrinsics'].append(torch.from_numpy(cam_extrinsic).float().unsqueeze(0))

            if data_list[i]['bev_maps'] is not None:
                for k, v in data_list[i]['bev_maps'].items():
                    v_tensor = v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
                    ret[f"map_{k}"].append(v_tensor.unsqueeze(0))


            ret['scenario'].append(data_list[i]['scenario'])
            ret['frame'].append(data_list[i]['frame'])
            ret['device_ids'].append(data_list[i]['device_ids'])

        BaseDataset.cat_data_dict_tensors(ret)

        return ret


