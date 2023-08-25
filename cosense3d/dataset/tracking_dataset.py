import numpy as np
import torch
import torch.nn.functional as F
from cosense3d.dataset.base_dataset import BaseDataset


class TrackingDataset(BaseDataset):

    def __init__(self, cfgs, mode):
        super().__init__(cfgs, mode)
        self.history_len = cfgs.get('history_len', 1)
        self._frame_samples = self.samples
        self.samples = self.samples[self.history_len:]

    def __getitem__(self, item):
        data_list = []
        for offset in range(- self.history_len, 1):
            idx = item + offset
            idx = min(max(idx, 0), len(self.samples) - 1)
            data = self.load_one_sample(idx)
            if self.pre_processes is not None:
                self.pre_processes(data)
            data_list.append(data)
        return data_list

    @staticmethod
    def collate_batch(batch_list):
        ret = {
            'batch_size': len(batch_list),
            'seq_len': len(batch_list[0]),
            # data
            'pcds': [],
            'features': [],
            'coords': [],
            'imgs': [],
            # meta
            'scenario': [],
            'frame': [],
            'device_ids': [],
            'projected': batch_list[0][0]['projected'],
            'objects': [],
            'objects_velo': [],
            'cam_intrinsics': [],
            'cam_extrinsics': [],
            'tf_cav2ego': [],
            'map_anchors': None,
            'anchor_offsets': None,
        }
        seq_len = len(batch_list[0])
        num_cavs = []
        # extensions
        if batch_list[0][0]['bev_maps'] is not None:
            for k in batch_list[0][0]['bev_maps'].keys():
                ret[f"map_{k}"] = []

        for i in range(len(batch_list)):
            for j in range(len(batch_list[i])):
                # all data in the same frame have the same batch index for early fusion
                if batch_list[i][0]['pcds'] is not None:
                    pcds = batch_list[i][j]['pcds']
                    pcds[:, 0] += sum(num_cavs)
                    ret['pcds'].append(torch.from_numpy(pcds).float())
                    tfs = torch.from_numpy(np.stack(batch_list[i][j]['tf_cav2ego'], axis=0)).float()
                    ret['tf_cav2ego'].append(tfs)
                    if 'features' in batch_list[i][j]:
                        features = batch_list[i][j]['features']
                        ret['features'].append(torch.from_numpy(features).float())
                    if 'coords' in batch_list[i][j]:
                        coords = batch_list[i][j]['coords']
                        coords = torch.from_numpy(coords).float()
                        coords = F.pad(coords, (1, 0, 0, 0), mode="constant", value=i * seq_len + j)
                        ret['coords'].append(coords)

                num_cavs.append(len(batch_list[i][0]['device_ids']['lidar']))

                objects = batch_list[i][j]['objects']
                velo = batch_list[i][j]['objects_velo']
                if objects is not None and len(objects) > 0:
                    objects = torch.from_numpy(objects).float()
                    velo = torch.from_numpy(velo).float()
                    objects = F.pad(objects, (1, 0, 0, 0), mode="constant", value=i * seq_len + j)
                else:
                    objects = torch.zeros((0, 12))
                    velo = torch.zeros((0, 2))
                ret['objects'].append(objects)
                ret['objects_velo'].append(velo)

                if batch_list[i][j]['imgs'] is not None:
                    ret['imgs'].append(torch.from_numpy(batch_list[i][j]['imgs']).float().unsqueeze(0))
                    cam_intrinsic = np.array(batch_list[i][j]['cam_params']['intrinsic'])
                    cam_extrinsic = np.array(batch_list[i][j]['cam_params']['extrinsic'])
                    ret['cam_intrinsics'].append(torch.from_numpy(cam_intrinsic).float().unsqueeze(0))
                    ret['cam_extrinsics'].append(torch.from_numpy(cam_extrinsic).float().unsqueeze(0))

                if batch_list[i][j]['bev_maps'] is not None:
                    for k, v in batch_list[i][j]['bev_maps'].items():
                        v_tensor = v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
                        ret[f"map_{k}"].append(v_tensor.unsqueeze(0))

            ret['scenario'].append([x['scenario'] for x in batch_list[i]])
            ret['frame'].append([x['frame'] for x in batch_list[i]])
            ret['device_ids'].append([x['device_ids'] for x in batch_list[i]])

        ret['num_cav'] = num_cavs
        for k, v in ret.items():
            if v == []:
                ret[k] = None
        BaseDataset.cat_data_dict_tensors(ret)

        return ret


