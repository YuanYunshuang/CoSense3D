import os
import yaml
import tqdm
import numpy as np
import time

from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs
from cosense3d.utils.box_utils import boxes_to_corners_3d
from cosense3d.utils import pclib


def label_cosense2opv2v(bbxs, pose, out_file, timestamp=None):
    vehicles = {}
    # only keep car, van, bus, truck
    bbxs = bbxs[bbxs[:, 5] > 2]
    # bbxs = bbxs[np.linalg.norm(bbxs[:, 1:4]) > 2]
    for bbx in bbxs:
        obj_id = int(bbx[0])
        obj_type = int(bbx[1])
        # process the information to the opv2v format
        location = bbx[2:5]
        angle = bbx[[8, 10, 9]] / np.pi * 180
        angle[[0, 2]] *= -1
        extent = bbx[5:8] / 2

        vehicles[int(obj_id)] = {
            'angle': angle.tolist(),
            'center': [0.0] * 3,
            'extent': extent.tolist(),
            'location': location.tolist(),
            'speed': 0,
            'type': obj_type
        }
    if isinstance(pose, np.ndarray):
        pose = pose.tolist()
    yaml_dict = {
        'lidar_pose': pose,
        'true_ego_pos': pose,
        'ego_speed': 0,
        'vehicles': vehicles
    }
    if timestamp is not None:
        # timestamp for ouster is corrected by subtracting a systematic time offset (0.35s)
        yaml_dict['timestamp'] = float(timestamp)
    with open(out_file, 'w') as fh:
        yaml.dump(yaml_dict, fh, default_flow_style=False)


def cosense2sustech(data_dir, dataset, out_dir=None):
    # make out dirs
    out_dir = os.path.join(data_dir, '..', 'sustech_fmt') \
        if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    meta = cs.load_meta(f"../metas/{dataset}")


def cosense2opv2v(data_dir, dataset, out_dir=None):
    # make out dirs
    out_dir = os.path.join(data_dir, '..', 'opv2v_fmt') \
        if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    meta = load_meta(f"../metas/{dataset}")
    for s, sdict in meta.items():
        scenario_dir = os.path.join(out_dir, s)
        os.makedirs(scenario_dir, exist_ok=True)
        for f, fdict in tqdm.tqdm(sdict.items()):
            bbx_global_center = np.array(fdict['meta']['bbx_center_global'])
            # bbx_global_corner = boxes_to_corners_3d(bbx_global_center[:, 2:])
            for a, adict in fdict['agents'].items():
                agent_dir = os.path.join(scenario_dir, a)
                if not os.path.exists(agent_dir):
                    os.makedirs(agent_dir)
                for l, ldict in adict['lidar0'].items():
                    lidar_pose = ldict['pose']
                    filename = ldict['filename'].replace('\\', '/')
                    # TODO rotate points and bbxs
                    pclib.lidar_bin2bin(
                        os.path.join(data_dir, filename),
                        os.path.join(agent_dir, f + '.bin')
                    )
                    # write label file
                    # time1 = time.time()
                    label_cosense2opv2v(bbx_global_center, lidar_pose,
                                       os.path.join(agent_dir, f + '.yaml'))
                    # print(time.time() - time1)


if __name__=="__main__":
    cosense2opv2v("/media/hdd/yuan/koko/data/LUMPI/cosense_fmt",
                  "/media/hdd/yuan/koko/data/LUMPI/opv2v_fmt")
