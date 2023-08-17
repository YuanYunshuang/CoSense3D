import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cosense3d.dataset.base_dataset import BaseDataset
from cosense3d.utils.pclib import load_pcd, pose2tf


class NoFusionDataset(BaseDataset):

    def __init__(self, cfgs, mode):
        super().__init__(cfgs, mode)

    def parse_samples(self):
        # list all frames, each frame as a sample
        self.samples = []
        for scenario, scontent in self.meta_dict.items():
            for frame, fdict in scontent.items():
                for agent_id in fdict['agents'].keys():
                    if len(fdict['agents'][agent_id]['lidar']) > 0:
                        self.samples.append([scenario, frame, agent_id])
        self.samples = sorted(self.samples)

        print(f"{self.mode} : {len(self.samples)} samples.")

    def load_one_sample(self, item):
        """
        Load data of the ```item```'th sample.
        Parameters
        ----------
        item : int
            sample index

        Returns
        -------
        batch_dict: dict
            - scenario: str
            - frame: str
            - pcds: np.ndarray [N, 6],
                columns are (x, y, z, intensity, lidar_id, point_cls).
            - imgs: camera data
            - maps: np.ndarray
                HD maps.
            - metas: dict
                all meta info for the current scenario.
        """
        # load meta info
        scenario, frame, agent_id = self.samples[item]
        meta_in = self.meta_dict[scenario][frame]['agents'][agent_id]

        # load ground truth
        if self.mode == 'train':
            gt_boxes = np.array(meta_in['gt_boxes'])
            # select and remap box cls
            gt_boxes = self.remap_box_cls(gt_boxes)
        else:
            gt_boxes = None

        # load data
        device_ids = {'lidar': [], 'cam': []}
        pcds = []
        imgs = []
        cam_params = {'intrinsic': [], 'extrinsic': []}

        # load lidar0 data
        if self.load_lidar:
            for i, (li, ldict) in enumerate(meta_in['lidar'].items()):
                device_ids['lidar'].append(f"{agent_id}.{li}")
                pcd_file = os.path.join(self.cfgs[f'data_path_{self.mode}'], ldict['filename'])
                lidar_dict = load_pcd(pcd_file)
                xyz = lidar_dict['xyz']
                pcd = np.zeros((len(xyz), 6))
                pcd[:, 0] = i
                pcd[:, 1:4] = xyz
                if 'intensity' in lidar_dict:
                    pcd[:, 4] = lidar_dict['intensity'].squeeze()
                if 'label' in lidar_dict:
                    pcd[:, -1] = lidar_dict['label'].squeeze()
                else:
                    # TODO last column is reserved for point_cls of semseg
                    pcd[:, -1] = -1
                if self.load_lidar_time and 'time' in lidar_dict:
                    pcd = np.concatenate([pcd, lidar_dict['time']], axis=1)
                pcds.append(pcd)
        # load cam data
        if self.load_img:
            for ci, cdict in meta_in['camera'].items():
                device_ids['cam'].append(f"{agent_id}.{ci}")
                cam_params['intrinsic'].append(cdict['intrinsic'])
                cam_params['extrinsic'].append(cdict['extrinsic'])
                img_files = [os.path.join(self.cfgs[f'data_path_{self.mode}'], f) for f in cdict['filenames']]
                # img_seq = np.stack([cv2.imread(f.replace('png', 'jpg')) for f in img_files], axis=0)
                # only read the mid. file
                try:
                    img = Image.open(img_files[len(img_files) // 2])
                except:
                    img = Image.open(img_files[len(img_files) // 2].replace('png', 'jpg'))
                imgs.append(img)

        return {
            'scenario': scenario,
            'frame': f"{frame}.{agent_id}",
            'device_ids': device_ids,
            'tf_cav2ego': None,
            'projected': False,
            'objects': gt_boxes,
            'pcds': np.concatenate(pcds, axis=0) if self.load_lidar else None,
            'imgs': imgs if self.load_img else None,
            'cam_params': cam_params if self.load_img else None,
            'bev_maps': None
        }


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
            'tf_cav2ego': None,
            'map_anchors': None,
            'anchor_offsets': None,
        }

        # extensions
        if data_list[0]['bev_maps'] is not None:
            for k in data_list[0]['bev_maps'].keys():
                ret[f"map_{k}"] = []

        num_cav = []
        for i in range(len(data_list)):
            # all data in the same frame have the same batch index for early fusion
            if data_list[i]['pcds'] is not None:
                pcds = data_list[i]['pcds']
                pcds[:, 0] = i
                ret['pcds'].append(torch.from_numpy(pcds).float())
                num_cav.append(1)

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
        ret['num_cav'] = num_cav
        return ret


