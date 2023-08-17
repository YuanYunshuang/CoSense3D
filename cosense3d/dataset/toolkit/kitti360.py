import glob
import tqdm
import os
from collections import namedtuple
import xml.etree.ElementTree as ET
import numpy as np
try:
    from kitti360scripts.helpers import annotation, ply
except:
    ImportError('kitti360scripts not installed.')

from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs
from cosense3d.utils.box_utils import corners_to_boxes_3d
from cosense3d.utils.misc import save_json
from cosense3d.utils.vislib import o3d_play_sequence

type_kitti2cosense = {
    'car': 'vehicle.car',
    'van': 'vehicle.van',
    'truck': 'vehicle.truck',
    'bus': 'vehicle.bus',
    'bicycle': 'vehicle.cyclist',
    'motorcycle': 'vehicle.motorcycle',
    'person': 'human.pedestrian'
}
type_cosense2kitti = {
    v: k for k, v in type_kitti2cosense.items()
}


class KITTI360(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.meta = {}
        self.scenarios = sorted(os.listdir(os.path.join(root_dir, 'data_3d_raw')))
        self.MAX_N = 1000
        self.load_calib()

    def to_cosense(self, meta_path):
        os.makedirs(meta_path, exist_ok=True)
        for s in self.scenarios:
            scenario_dict = {}
            dynamic_boxes, static_boxes = self.load_3d_bboxes(s)
            static_boxes_centers = np.array([[id] + box['vertices'].mean(axis=0).tolist()
                                             for id, box in static_boxes.items()])
            velo_poses = self.load_velo_poses(s)
            self.meta[s] = {}
            files = sorted(glob.glob(os.path.join(
                self.root_dir, 'data_3d_raw', s, 'velodyne_points/data/*'
            )))
            for f in tqdm.tqdm(files):
                frame = int(os.path.basename(f).split('.')[0])
                cur_boxes = {}
                if frame not in velo_poses:
                    continue
                tf_velo_to_world = velo_poses[frame]
                tf_world_to_velo = np.linalg.inv(tf_velo_to_world)

                if frame in dynamic_boxes:
                    cur_dynamic_boxes = dynamic_boxes[frame]
                    cur_boxes.update(cur_dynamic_boxes)

                mask = np.linalg.norm(static_boxes_centers[:, 1:3]
                                      - tf_velo_to_world[:2, 3],
                                      axis=1) < 100
                cur_static_boxes = {int(id): static_boxes[int(id)]
                                    for id in static_boxes_centers[mask, 0]}
                cur_boxes.update(cur_static_boxes)

                # transform box from world to velo coordinates
                boxes_corner = []
                boxes_id_type = []
                for id, box in cur_boxes.items():
                    box_corner = box['vertices'][[1, 3, 4, 6, 0, 2, 5, 7]]
                    box_corner = tf_world_to_velo[:3, :3] @ box_corner.T \
                                 + tf_world_to_velo[:3, 3:]
                    boxes_id_type.append([id, box['semanticId']])
                    boxes_corner.append(box_corner.T)

                # convert boxes from corners format to cosense3d center format
                if len(boxes_corner) > 0:
                    bbx_center = corners_to_boxes_3d(np.array(boxes_corner))
                    bbx_center = np.concatenate([np.array(boxes_id_type),
                                                 bbx_center], axis=1)
                else:
                    bbx_center = []

                lidar_relative_filename = f.replace(self.root_dir, '')[1:]
                fdict = cs.fdict_template()
                # no cooperation needed, take lidar0 as global for each frame
                if isinstance(bbx_center, np.ndarray):
                    bbx_center = bbx_center.tolist()
                cs.update_frame_bbx(fdict, bbx_center)
                cs.update_agent(fdict, agent_id=0, agent_type='cav')
                cs.update_agent_lidar(fdict, agent_id=0, lidar_id=0,
                                      lidar_file=lidar_relative_filename)
                scenario_dict[frame] = fdict

            self.meta[s] = scenario_dict
            save_json(scenario_dict, os.path.join(meta_path, f"{s}.json"))

    def load_calib(self):
        calib_cam2velo = np.eye(4)
        calib_cam2velo[:3, :] = np.loadtxt(os.path.join(
            self.root_dir, 'calibration', 'calib_cam_to_velo.txt'
        )).reshape(3, 4)
        self.calib_cam2velo = calib_cam2velo
        self.calib_velo2cam = np.linalg.inv(calib_cam2velo)

    def load_cam_poses(self, scenario):
        pose_file = os.path.join(
            self.root_dir, 'data_poses', scenario, 'cam0_to_world.txt'
        )
        cam_poses = np.loadtxt(pose_file)
        cam_poses = {int(p[0]): p[1:].reshape(4, 4) for p in cam_poses}
        return cam_poses

    def load_velo_poses(self, scenario):
        velo_poses = {}
        for frame, pose_cam in self.load_cam_poses(scenario).items():
            velo_poses[frame] = pose_cam @ self.calib_velo2cam
        return velo_poses

    def load_3d_bboxes(self, scenario):
        labelDir = os.path.join(self.root_dir, 'data_3d_bboxes')
        anno3d = annotation.Annotation3D(labelDir, scenario)
        dynamic_boxes = {}
        static_boxes = {}
        for globalId, v in anno3d.objects.items():
            # skip dynamic objects
            # if len(v) > 1:
            #     continue
            for obj in v.values():
                # TODO load other types for future if needed
                if obj.name != 'car':
                    continue
                timestamp = obj.timestamp
                lines = np.array(obj.lines)
                vertices = obj.vertices
                faces = obj.faces
                semanticId, instanceId = annotation.global2local(globalId)

                box = {
                        'semanticId': 0,
                        'vertices': vertices,
                        'lines': lines,
                        'faces': faces,
                        'type': obj.name
                    }

                if timestamp == -1:
                    static_boxes[instanceId] = box
                else:
                    if timestamp not in dynamic_boxes:
                        dynamic_boxes[timestamp] = {}
                    dynamic_boxes[timestamp][instanceId] = box
        return dynamic_boxes, static_boxes


if __name__=="__main__":
    meta_path = "/media/hdd/yuan/CoSense3D/dataset/metas/kitti360"
    root_dir = "/koko/kitti360"
    kitti360 = KITTI360("/koko/kitti360")
    # kitti360.to_cosense(meta_path)
    meta_dict = cs.load_meta(meta_path)
    o3d_play_sequence({'2013_05_28_drive_0002_sync': meta_dict['2013_05_28_drive_0002_sync']},
                      root_dir)