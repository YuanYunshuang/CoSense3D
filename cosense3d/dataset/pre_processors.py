import os
import logging
import pickle

import numpy as np
import cv2
from torchvision.transforms import ToTensor
from PIL import Image

from cosense3d.utils import pclib, box_utils
from cosense3d.dataset.toolkit.kitti import type_cosense2kitti, type_kitti2cosense
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs
from cosense3d.dataset.const import CoSenseBenchmarks as bms
from cosense3d.ops.iou3d_nms_utils import boxes_bev_iou_cpu
from cosense3d.utils.pclib import rotate3d


class Compose(object):
    """Composes several pre-processing modules together.
        Take care that these functions modify the input data directly.
    """

    def __init__(self, processes):
        self.processes = processes

    def __call__(self, data_dict):
        for p in self.processes:
            p(data_dict)
        return data_dict


class PreProcessorBase(object):
    def __init__(self, **kwargs):
        logging.info(f"{self.__class__.__name__}:")
        for k, v in kwargs.items():
            setattr(self, k, v)
            logging.info(f"- {k}: {v}")

    def __call__(self, data_dict):
        raise NotImplementedError


class EarlyFusionIn(object):
    def __init__(self):
        logging.info(f"{self.__class__.__name__} as Input.")

    def __call__(self, data_dict):
        pass


class ProjectPointsToEgo(PreProcessorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data_dict):
        pcds = data_dict['pcds']
        tf_matrices = data_dict['tf_cav2ego']
        # crop along axis
        for i, tf in enumerate(tf_matrices):
            mask = pcds[:, 0]==i
            pcd = pcds[mask, 1:4]
            pcd[:, :3] = (tf[:3, :3] @ pcd[:, :3].T).T
            pcd = pcd + tf[:3, 3].reshape(1, 3)
            pcds[mask, 1:4] = pcd
        data_dict['projected'] = True



class CropLidarRange(PreProcessorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data_dict):
        if data_dict['pcds'] is None:
            return
        ##########PCD###########
        mask_pcd = np.ones_like(data_dict['pcds'][:, 0]).astype(bool)
        # crop along axis
        for i, axis in enumerate('xyz'):
            if hasattr(self, axis):
                # shrink the range with 1e-4 to ensure ME coords rounds to the correct idx
                m_pcd = np.logical_and(
                    data_dict['pcds'][:, i + 1] > getattr(self, axis)[0] + 1e-4,
                    data_dict['pcds'][:, i + 1] < getattr(self, axis)[1] - 1e-4
                )
                mask_pcd = np.logical_and(mask_pcd, m_pcd)
        data_dict['pcds'] = data_dict['pcds'][mask_pcd]
        # crop according distance range
        if hasattr(self, 'r'):
            dist = np.linalg.norm(data_dict['pcds'][:, 1:3], axis=1)
            data_dict['pcds'] = data_dict['pcds'][dist < self.r]
        ##########OBJECTS###########
        if data_dict['objects'] is not None:
            mask_obj = np.ones_like(data_dict['objects'][:, 0]).astype(bool)
            for i, axis in enumerate('xyz'):
                if hasattr(self, axis):
                    m_obj = np.logical_and(
                        data_dict['objects'][:, i + 2] > getattr(self, axis)[0] + 1e-4,
                        data_dict['objects'][:, i + 2] < getattr(self, axis)[1] - 1e-4
                    )
                    mask_obj = np.logical_and(mask_obj, m_obj)

            data_dict['objects'] = data_dict['objects'][mask_obj]

            # crop according distance range
            if hasattr(self, 'r'):
                dist = np.linalg.norm(data_dict['objects'][:, 2:4], axis=1)
                data_dict['objects'] = data_dict['objects'][dist < self.r]


class GeoAugmentation(PreProcessorBase):
    """
    This module generates an overall rotation matrix from the sub
    rotation matrices defined by the random rotation around xyz axis,
    random flip along x, y or xy axis, and the random scaling.
    In this module, the out point cloud will be rotated to the global
    augmented coordinate system. However the translation will be unchanged.
    We call this state of point cloud as PC_rot_aug. Once the translation
    is performed, we get PC_aug.
    With the new lidar0 pose (translation from rot_aug to aug, rotation angles
    are all set to zeros), the PC_rot_aug can be translated to PC_aug.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.operations = [
            'rotate',
            'flip',
            'scale',
        ]

    def __call__(self, data_dict):
        # get overall tf from rotate, flip and scale
        aug_tf = np.eye(4)
        for op in self.operations:
            if hasattr(self, op):
                aug_tf = getattr(self, f'fn_{op}')(aug_tf)

        # process lidar0 data
        if data_dict['pcds'] is not None:
            pcds = data_dict['pcds']

            # add noise
            if hasattr(self, 'noise'):
                pcds = self.fn_noise(pcds)

            # rotate pcds to global rotation-coords
            if data_dict['projected'] or data_dict['tf_cav2ego'] is None:
                pcds_global = pcds
            else:
                tfs = data_dict['tf_cav2ego']
                pcds_global = []
                for i in sorted(np.unique(pcds[:, 0])):
                    tf = tfs[int(i)]
                    pcd = pcds[pcds[:, 0] == i]
                    if len(tf) == 4:
                        pcd[:, 1:4] = (tf[:3, :3] @ pcd[:, 1:4].T).T
                        pcd[:, 1:4] = pcd[:, 1:4] + tf[:3, 3].reshape(1, 3)
                    else:
                        pcd[:, 1:4] = pclib.rotate_points_batch(
                            pcd[:, 1:4].reshape(1, -1, 3),
                            tf[3:]
                        )
                        pcd[:, 1:4] = pcd[:, 1:4] + tf[:3].reshape(1, 3)
                    pcds_global.append(pcd)
                pcds_global = np.concatenate(pcds_global, axis=0)
                data_dict['projected'] = True

            # rotate pcds to augmented coords
            pcds_global[:, 1:4] = (aug_tf[:3, :3] @ pcds_global[:, 1:4].T).T
            # lidar_poses[:, :3] = (aug_tf[:3, :3] @ lidar_poses[:, :3].T).T
            # lidar_poses[:, 3:] = 0

            data_dict['pcds'] = pcds_global

        # process object data
        boxes = data_dict['objects']  # in global coords
        boxes_corner = box_utils.boxes_to_corners_3d(boxes[:, 2:])  # (N, 8, 3)
        # rotate bbx to augmented coords
        boxes_corner = (aug_tf[:3, :3] @ boxes_corner.reshape(-1, 3).T
                        ).T.reshape(len(boxes_corner), 8, 3)
        boxes_center = box_utils.corners_to_boxes_3d(boxes_corner)
        boxes[:, 2:] = boxes_center
        data_dict['objects'] = boxes

        # update camera extrinsics
        if data_dict['cam_params'] is not None:
            for i, param in enumerate(data_dict['cam_params']):
                extrinsic = param['extrinsic']
                data_dict['cam_params'][i]['extrinsic'] = \
                    (np.array(extrinsic) @ np.linalg.inv(aug_tf)).tolist()

    def fn_rotate(self, tf):
        # param: [roll, pitch, yaw] in degree
        angles = []
        for angle in self.rotate:
            angle = angle / 180 * np.pi * np.random.random()
            angles.append(angle)
        angles = np.array(angles)
        rot = pclib.rotation_matrix(angles)
        tf[:3, :3] = rot @ tf[:3, :3]
        return tf

    def fn_flip(self, tf):
        rot = np.eye(3)
        flip = np.random.choice(4, 1)
        # flip =1 : flip x
        # flip =2 : flip y
        # flip =3 : flip x & y

        # flip x
        if 'x' in getattr(self, 'flip', 'xy') and (flip == 1 or flip == 3):
            rot[0, 0] *= -1
        # flip y
        if 'y' in getattr(self, 'flip', 'xy') and (flip == 2 or flip == 3):
            rot[1, 1] *= -1
        tf[:3, :3] = rot @ tf[:3, :3]
        return tf

    def fn_scale(self, tf):
        scale = np.eye(3)
        scale_ratio = np.random.uniform(1.0 - self.scale, 1.0 + self.scale, (1, 3))
        scale[[0, 1, 2], [0, 1, 2]] = scale_ratio
        tf[:3, :3] = scale @ tf[:3, :3]
        return tf

    def fn_noise(self, pcds):
        noise = np.random.normal(0, 0.1, (len(pcds), 3))
        pcds[:, 1:4] = pcds[:, 1:4] + noise
        return pcds


class RegisterCoordinates(PreProcessorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data_dict):
        if data_dict['pcds'] is None:
            return
        pcds = data_dict['pcds']  # in local, global_rot or aug_rot
        objects = data_dict['objects']

        if hasattr(self, 'raster_resolution'):
            tfs = data_dict['tf_cav2ego']
            map_anchors = []
            anchor_offsets = []
            poses = []
            for i, tf in enumerate(tfs):
                pcd_mask = pcds[:, 0] == i
                loc = tf[3, :2]
                # find global registration anchor
                map_anchor = np.round(loc / self.raster_resolution) \
                             * self.raster_resolution
                # transform pcd and pose to anchor coords
                offset = loc - map_anchor
                pcds[pcd_mask, 1:3] += offset.reshape(1, 2)

                tf[3, :2] -= offset
                poses.append(pose)
                map_anchors.append(map_anchor)
                anchor_offsets.append(offset)

            data_dict['tf_cav2ego'] = tfs
            data_dict['map_anchors'] = map_anchors
            data_dict['anchor_offsets'] = anchor_offsets

        if hasattr(self, 'height'):
            pcds[:, 3] -= self.height
            if objects is not None:
                objects[:, 4] -= self.height

        data_dict['pcds'] = pcds
        data_dict['objects'] = objects



class ComposePointFeatures(PreProcessorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data_dict):
        """
        Update batch_dict with key "features"  in batch_dict["data"]
        Compose cooordinates and features according to the symbols,
        each valid symbol will be mapped to a self.get_feature_[symbol] function
        to get the corresponding feature in lidar0. Valid symbols are
        - 'x'(coordinate),
        - 'y'(coordinate),
        - 'z'(coordinate),
        - 'i'(intensity),
        - 't'(theta in degree),
        - 'c'(cos(t)),
        - 's'(sin(t)).

        Parameters
        ----------
        data_dict: dict,
                pcds must be np.ndarray [N, 3+c], columns 1-3 are x, y, z,
                if intensity is availble, it should in the 4th column

        Returns
        -------
        """
        if data_dict['pcds'] is None:
            return
        pcds = data_dict['pcds'][:, 1:5]
        features = getattr(self, 'features', "x,y,z")
        features = features.split(',')
        data = [getattr(self, f'get_feature_{f.strip()}')(pcds) for f in features]
        features = np.concatenate(data, axis=1)
        data_dict['features'] = features

        # Feature retrieving functions, input lidar0 columns must be # [x,y,z,i,obj,cls]

    @staticmethod
    def get_feature_x(lidar):
        """x coordinate"""
        return lidar[:, 0].reshape(-1, 1)

    @staticmethod
    def get_feature_y(lidar):
        """y coordinate"""
        return lidar[:, 1].reshape(-1, 1)

    @staticmethod
    def get_feature_z(lidar):
        """z coordinate"""
        return lidar[:, 2].reshape(-1, 1)

    @staticmethod
    def get_feature_i(lidar):
        """intensity"""
        if lidar.shape[1] > 3:
            return lidar[:, 3].reshape(-1, 1)
        else:
            return np.ones_like(lidar[:, 0]).reshape(-1, 1)

    @staticmethod
    def get_feature_t(lidar):
        """orientation"""
        degs = np.rad2deg(np.arctan2(lidar[:, 1], lidar[:, 0]).reshape(-1, 1))
        degs = (degs + 360) % 360
        return degs

    @staticmethod
    def get_feature_d(lidar):
        """distance"""
        return np.linalg.norm(lidar[:, :2], axis=1).reshape(-1, 1)

    @staticmethod
    def get_feature_c(lidar):
        """cosine"""
        return np.cos(np.arctan2(lidar[:, 1], lidar[:, 0])).reshape(-1, 1)

    @staticmethod
    def get_feature_s(lidar):
        """sine"""
        return np.sin(np.arctan2(lidar[:, 1], lidar[:, 0])).reshape(-1, 1)

    @staticmethod
    def get_feature_cs(lidar):
        """normalized coordinate in euclidian system"""
        x_abs = 1 / (np.abs(lidar[:, 1] / (lidar[:, 0] +
                                           (lidar[:, 0] == 0) * 1e-6)) + 1)
        y_abs = 1 - x_abs
        x = x_abs * np.sign(lidar[:, 0])
        y = y_abs * np.sign(lidar[:, 1])
        return np.stack([x, y], axis=1)


class NormalizeImg(PreProcessorBase):
    def __init__(self, cams=None, **kwargs):
        super().__init__(**kwargs)
        self.cams = cams
        if cams is None:
            assert hasattr(self, "mean")
            assert hasattr(self, "std")

    def __call__(self, data_dict):
        cam_ids = data_dict['device_ids']['cam']
        imgs = data_dict['imgs']
        if data_dict['imgs'].max() > 1:
            imgs = imgs / 255
        for i, cam_id in enumerate(cam_ids):
            mean = self.mean if self.cams is None else self.cams[cam_id]['mean']
            std = self.std if self.cams is None else self.cams[cam_id]['std']
            imgs[i] = (imgs[i] - np.array(mean).reshape(1, 1, 3)) / np.array(std).reshape(1, 1, 3)

        data_dict['imgs'] = imgs


class CVTTransform(PreProcessorBase):
    def __init__(self, n_cls, **kwargs):
        super().__init__(**kwargs)
        self.n_cls = n_cls
        self.to_tensor = ToTensor()

    def __call__(self, data_dict):
        bev_maps = data_dict['bev_maps']

        # decode bev
        shift = np.arange(self.n_cls, dtype=np.int32)[None, None]
        bev_maps['bev'] = np.array(bev_maps['bev'])[..., None]
        bev_maps['bev'] = (bev_maps['bev'] >> shift) & 1
        bev_maps['bev'] = (255 * bev_maps['bev']).astype(np.uint8)
        bev_maps['bev'] = self.to_tensor(bev_maps['bev'])

        # visibility
        bev_maps['visibility'] = np.array(bev_maps['visibility'], dtype=np.uint8)

        # center map
        bev_maps['center'] = bev_maps.pop('aux')['aux'][..., 1]
        bev_maps['center'] = self.to_tensor(bev_maps['center'])

        data_dict['bev_maps'] = bev_maps


class CropTopResizeImg(PreProcessorBase):
    def __init__(self, h, w, top_crop, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.w = w
        self.top_crop = top_crop

    def __call__(self, data_dict):
        imgs = data_dict['imgs']
        intrinsics = data_dict['cam_params']['intrinsic']
        imgs_new = []
        Is = []
        for image, I_original in zip(imgs, intrinsics):
            h_resize = self.h + self.top_crop
            w_resize = self.w

            # image_new = cv2.resize(image, (w_resize, h_resize), cv2.INTER_LINEAR)
            # image_new = image_new[self.top_crop:, :]
            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, self.top_crop, image_new.width, image_new.height))

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= self.top_crop

            imgs_new.append(image_new)
            Is.append(I)

        data_dict['imgs'] = np.stack(imgs_new, axis=0)
        data_dict['cam_params']['intrinsic'] = Is


class GTBoxSampler(object):
    def __init__(self,
                 db_path,
                 benchmark,
                 sampler_cfg,
                 num_frames,
                 **kwargs):
        self.db_path = db_path
        self.benchmark = bms[benchmark]
        self.sampler_cfg = sampler_cfg

        # load gt samples
        self.db_infos = {}
        self.num_frames = num_frames
        for id, cosense_cls_names in self.benchmark.items():
            self.db_infos[id] = []
            for cosense_cls in cosense_cls_names:
                kitti_name = type_cosense2kitti[cosense_cls]
            db_info_path = os.path.join(
                    db_path, f"{kitti_name}_samples.pkl"
            )
            with open(db_info_path, 'rb') as f:
                infos = pickle.load(f)
                self.db_infos[id].extend(infos)

        # filter database
        for func_name, val in sampler_cfg['PREPARE'].items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for x in sampler_cfg['SAMPLE_GROUPS']:
            cosense_remapped_id, sample_num = x.split(':')
            cosense_remapped_id = int(cosense_remapped_id)
            if cosense_remapped_id not in self.benchmark.keys():
                continue
            self.sample_class_num[cosense_remapped_id] = sample_num
            self.sample_groups[cosense_remapped_id] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[cosense_remapped_id]),
                'indices': np.arange(len(self.db_infos[cosense_remapped_id]))
            }

        # load planes
        planes_file = os.path.join(self.db_path, "planes_train.pkl")
        with open(planes_file, 'rb') as f:
            self.planes = pickle.load(f)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            this_infos = []
            for info in dinfos:
                if 'difficulty' in info:
                    if info['difficulty'] not in removed_difficulty:
                        this_infos.append(info)
                else:
                    this_infos.append(info)
            new_db_infos[key] = this_infos
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if len(info['points']) >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_plane):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_plane
        center_lidar = gt_boxes[:, 2:5]
        cur_lidar_height = (-d - a * center_lidar[:, 0] - b * center_lidar[:, 1]) / c
        mv_height = gt_boxes[:, 4] - gt_boxes[:, 7] / 2 - cur_lidar_height
        gt_boxes[:, 4] -= mv_height  # lidar0 view
        return gt_boxes, mv_height

    def points_rigid_transform(self,cloud,pose):
        if cloud.shape[0]==0:
            return cloud
        mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
        pose_mat=np.mat(pose)
        mat[:,0:3]=cloud[:,0:3]
        mat=np.mat(mat)
        transformed_mat=pose_mat*mat.T
        T=np.array(transformed_mat.T,dtype=np.float32)
        return T[:,0:3]

    def get_registration_angle(self,mat):

        cos_theta=mat[0,0]
        sin_theta=mat[1,0]

        if  cos_theta < -1:
            cos_theta = -1
        if cos_theta > 1:
            cos_theta = 1

        theta_cos = np.arccos(cos_theta)

        if sin_theta >= 0:
            return theta_cos
        else:
            return 2 * np.pi - theta_cos

    def registration(self,pose, pre_pose, pre_obj_points, pre_box3d_lidar):

        inv_pose_of_last_frame = np.linalg.inv(pose)
        registration_mat = np.matmul(inv_pose_of_last_frame, pre_pose)

        if len(pre_obj_points)!=0:
            pre_obj_points[:, 0:3] = self.points_rigid_transform(pre_obj_points, registration_mat)[:,0:3]
        angle = self.get_registration_angle(registration_mat)
        pre_box3d_lidar[0:3] = self.points_rigid_transform(np.array([pre_box3d_lidar]), registration_mat)[0, 0:3]
        pre_box3d_lidar[6]+=angle

        return pre_obj_points, pre_box3d_lidar

    def add_sampled_boxes_to_scene_multi(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):

        gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        gt_idx = np.array(data_dict['ob_idx'])[gt_boxes_mask]

        if 'gt_tracklets' in data_dict:
            data_dict['gt_tracklets']=data_dict['gt_tracklets'][gt_boxes_mask]
            data_dict['num_bbs_in_tracklets'] = data_dict['num_bbs_in_tracklets'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        ob_index_list =[]
        box3d_lidar_list = []

        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = os.path.join(self.root_path, info['path'])
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            pre_box3d_lidar = info['box3d_lidar']
            id = info['ob_idx']
            seq_idx = info["seq_idx"]
            sample_idx = info['image_idx']

            obj_points_list.append(obj_points)
            ob_index_list.append(str(sample_idx)+'_'+str(seq_idx)+'_'+str(id))
            box3d_lidar_list.append(pre_box3d_lidar)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        ob_idx = np.array(ob_index_list)
        sampled_gt_boxes = np.array(box3d_lidar_list)

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)

        gt_idx = np.concatenate([gt_idx, ob_idx], axis=0)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        data_dict['ob_idx'] = gt_idx.tolist()

        if self.num_frames>1:

            gt_tracklets = np.zeros(shape=(len(total_valid_sampled_dict),7+(self.num_frames-1)*4))

            gt_tracklets[:,0:7]=sampled_gt_boxes[:,0:7]

            num_bbs_in_tracks = np.ones(shape=(len(total_valid_sampled_dict),1))

            for i in range(1,self.num_frames):

                if 'points'+str(-i) not in data_dict:
                    continue
                if 'gt_names'+str(-i) not in data_dict:
                    pre_gt_boxes = np.zeros(shape=(0,7))
                    pre_gt_names = np.zeros(shape=(0,))
                    pre_gt_idx = np.zeros(shape=(0,))
                else:
                    pre_gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names'+str(-i)]], dtype=np.bool_)
                    pre_gt_boxes = data_dict['gt_boxes'+str(-i)][pre_gt_boxes_mask]
                    pre_gt_names = data_dict['gt_names'+str(-i)][pre_gt_boxes_mask]
                    pre_gt_idx = np.array(data_dict['ob_idx' + str(-i)])[pre_gt_boxes_mask]
                pre_points = data_dict['points'+str(-i)]

                pre_obj_points_list = []
                pre_box3d_lidar_list=[]
                pre_sampled_gt_names=[]
                pre_ob_idx_list=[]

                for idx, info in enumerate(total_valid_sampled_dict):

                    if 'box3d_lidar'+str(-i) in info:
                        num_bbs_in_tracks[idx,0]+=1
                        pre_box3d_lidar = np.zeros(shape=info['box3d_lidar'+str(-i)].shape)
                        pre_box3d_lidar[:] = info['box3d_lidar'+str(-i)][:]

                        pre_file_path = os.path.join(self.db_path, info['path'+str(-i)])

                        pre_obj_points = np.fromfile(str(pre_file_path), dtype=np.float32).reshape(
                            [-1, self.sampler_cfg.NUM_POINT_FEATURES])

                        pre_obj_points[:, :3] += pre_box3d_lidar[:3]

                        pose = info['pose']
                        pre_pose = info['pose'+str(-i)]

                        pre_obj_points,pre_box3d_lidar = self.registration(pose,pre_pose,pre_obj_points,pre_box3d_lidar)

                        gt_tracklets[idx,3+i*4:6+i*4]=pre_box3d_lidar[0:3]
                        gt_tracklets[idx, 10+(i-1)*4] = pre_box3d_lidar[6]

                        pre_box3d_lidar_list.append(pre_box3d_lidar)
                        pre_obj_points_list.append(pre_obj_points)
                        pre_sampled_gt_names.append(info['name'])

                        id = info['ob_idx']
                        seq_idx = info["seq_idx"]
                        sample_idx = info['image_idx']
                        pre_ob_idx_list.append(str(sample_idx)+'_'+str(seq_idx)+'_'+str(id))

                if len(pre_obj_points_list)>0:
                    pre_obj_points = np.concatenate(pre_obj_points_list, axis=0)
                    pre_box3d_lidar = np.array(pre_box3d_lidar_list)
                    pre_ob_idx = np.array(pre_ob_idx_list)
                    pre_sampled_gt_names=np.array(pre_sampled_gt_names)

                    pre_points = box_utils.remove_points_in_boxes3d(pre_points, pre_box3d_lidar)
                    pre_points = np.concatenate([pre_points, pre_obj_points], axis=0)
                    pre_gt_names = np.concatenate([pre_gt_names, pre_sampled_gt_names], axis=0)
                    pre_gt_boxes = np.concatenate([pre_gt_boxes, pre_box3d_lidar], axis=0)
                    pre_gt_idx = np.concatenate([pre_gt_idx,pre_ob_idx],0)

                    data_dict['gt_boxes'+str(-i)] = pre_gt_boxes
                    data_dict['gt_names'+str(-i)] = pre_gt_names
                    data_dict['points'+str(-i)] = pre_points
                    data_dict['ob_idx'+str(-i)] = pre_gt_idx.tolist()

            if 'gt_tracklets' in data_dict:
                data_dict["gt_tracklets"]=np.concatenate([data_dict["gt_tracklets"],gt_tracklets], axis=0)
                data_dict["num_bbs_in_tracklets"] = np.concatenate([data_dict["num_bbs_in_tracklets"], num_bbs_in_tracks], axis=0)

        return data_dict

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):

        gt_boxes_mask = np.array([n in self.benchmark.keys() for n in data_dict['objects'][:, 1]],
                                 dtype=np.bool_)
        gt_boxes = data_dict['objects'][gt_boxes_mask]
        gt_classes = data_dict['objects'][:, 1][gt_boxes_mask]
        if 'gt_tracklets' in data_dict:
            data_dict['gt_tracklets']=data_dict['gt_tracklets'][gt_boxes_mask]
        pcd = data_dict['pcds']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, self.planes[data_dict['frame']]
            )

        obj_points_list = []

        for idx, info in enumerate(total_valid_sampled_dict):
            obj_points = info['points']
            obj_points[:, :3] += info['box3d_lidar'][:3] # local coor to lidar0 coor

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, [2, 3, 4, 5, 6, 7, 10]],
            extra_width=self.sampler_cfg['REMOVE_EXTRA_WIDTH']
        )
        pcd = box_utils.remove_points_in_boxes3d(pcd, large_sampled_gt_boxes, x_idx=1)
        points_add = np.zeros((len(obj_points), 6))
        points_add[:, 1:5] = obj_points
        # TODO: point cls should be adapted if semseg is used.
        points_add[:, 5] = -1
        pcd = np.concatenate([points_add, pcd], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['objects'] = gt_boxes
        data_dict['pcds'] = pcd

        return data_dict

    @staticmethod
    def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
        """
        Args:
            boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

        Returns:
            boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        """
        w, l, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:,
                                                                   5:6], boxes3d_lidar[:, 6:7]
        boxes3d_lidar[:, 2] += h[:, 0] / 2
        return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['objects']
        gt_cls_ids = gt_boxes[:, 1]
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []

        for cls_id, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(cls_id == gt_cls_ids)
                sample_group['sample_num'] = str(int(self.sample_class_num[cls_id]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(cls_id, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = self.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, [2, 3, 4, 5, 6, 7, 10]])
                iou2 = boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])

                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = np.zeros((len(valid_mask), 11))
                valid_sampled_boxes[:, [2, 3, 4, 5, 6, 7, 10]] = sampled_boxes[valid_mask]
                valid_sampled_boxes[:, [1]] = cls_id

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            if self.num_frames>1:
                data_dict = self.add_sampled_boxes_to_scene_multi(data_dict, sampled_gt_boxes, total_valid_sampled_dict)
            else:
                data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        return data_dict



