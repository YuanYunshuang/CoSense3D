import glob
import os
import tqdm
import numpy as np

from cosense3d.utils import pclib
from cosense3d.utils.misc import load_json, save_json
from cosense3d.utils.box_utils import corners_to_boxes_3d
from cosense3d.dataset.toolkit import register_pcds
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs
from cosense3d.utils.pcdio import point_cloud_from_path
from cosense3d.utils.vislib import o3d_draw_frame_data, \
    o3d_draw_agent_data, o3d_draw_pcds_bbxs

global_time_offset = 1.62616 * 1e9

def calib_to_tf_matrix(calib_file):
    calib = load_json(calib_file)
    if 'transform' in calib:
        tf = calib['transform']
    else:
        tf = calib
    tf_matrix = np.eye(4)
    tf_matrix[:3, :3] = np.array(tf['rotation'])
    tf_matrix[:3, 3:] = np.array(tf['translation'])
    if 'relative_error' in calib:
        tf_matrix[0, 3] += calib['relative_error']['delta_x']
        tf_matrix[1, 3] += calib['relative_error']['delta_y']
    return tf_matrix


def load_label(label_file):
    labels = load_json(label_file)
    bbxs_center = []
    bbxs_corner = []
    for l in labels:
        obj_type = {
            'car': 'vehicle.car',
            'van': 'vehicle.van',
            'truck': 'vehicle.truck',
            'bus': 'vehicle.bus',
            'pedestrian': 'human.pedestrian',
            'trafficcone': 'static.trafficcone',
            'motorcyclist': 'vehicle.motorcycle',
            'cyclist': 'vehicle.cyclist',
            'tricyclist': 'vehicle.tricyclist',
            'barrowlist': 'static.barrowlist',
        }[l.get('type', "car").lower()]
        track_id = l.get('track_id', -1)
        bbx = [
            int(track_id),
            cs.OBJ_NAME2ID[obj_type],
            l['3d_location']['x'],
            l['3d_location']['y'],
            l['3d_location']['z'],
            l['3d_dimensions']['l'],
            l['3d_dimensions']['w'],
            l['3d_dimensions']['h'],
            0,
            0,
            l['rotation']
        ]
        bbxs_center.append([float(x) for x in bbx])
        if 'world_8_points' in l:
            bbx_corner = np.array(l['world_8_points'])
            bbx_corner = [bbx.tolist() for bbx in bbx_corner]
            bbxs_corner.append(bbx_corner)
    return bbxs_center, bbxs_corner


def load_info_to_dict(info_file):
    infos = load_json(info_file)
    info_dict = {}
    for info in infos:
        frame = os.path.basename(info['pointcloud_path'][:-4])
        info_dict[frame] = info
    return info_dict


def convert_v2x_c(root_dir, meta_out_dir):
    cvi_path = "cooperative-vehicle-infrastructure"
    infra_path = "infrastructure-side"
    cav_path = "vehicle-side"
    coop_path = "cooperative"
    info_file = "data_info.json"
    inf_lidar_path = "cooperative-vehicle-infrastructure-infrastructure-side-velodyne"
    cav_lidar_path = "cooperative-vehicle-infrastructure-vehicle-side-velodyne"
    new_label_path = "DAIR-V2X-C_Complemented_Anno"
    inf_info_file = os.path.join(root_dir, cvi_path, infra_path, info_file)
    inf_info = load_info_to_dict(inf_info_file)
    veh_info_file = os.path.join(root_dir, cvi_path, cav_path, info_file)
    veh_info = load_info_to_dict(veh_info_file)
    frame_pairs = load_json(os.path.join(root_dir, cvi_path, coop_path, info_file))

    meta_dict = {}
    veh_frames = []
    inf_frames = []
    offsets = []
    for pair in frame_pairs:
        veh_frame = os.path.basename(pair['vehicle_pointcloud_path'][:-4])
        inf_frame = os.path.basename(pair['infrastructure_pointcloud_path'][:-4])
        label_frame = os.path.basename(pair['cooperative_label_path'][:-5])
        assert veh_frame == label_frame
        veh_frames.append(veh_frame)
        inf_frames.append(inf_frame)
        offsets.append(pair['system_error_offset'])

    # load all re-annotated samples
    train = load_json(os.path.join(root_dir, new_label_path, 'train.json'))
    val = load_json(os.path.join(root_dir, new_label_path, 'val.json'))
    split = {
                # 'train': train,
                 'test': val
             }

    for sp, frames in split.items():
        visualization = True
        for frame in tqdm.tqdm(frames):
            cur_veh_info = veh_info[frame]
            scenario = cur_veh_info['batch_id']
            # processing vehicle meta
            tf_novatel2world = calib_to_tf_matrix(
                os.path.join(root_dir, cvi_path, cav_path,
                             cur_veh_info['calib_novatel_to_world_path'])
            )
            tf_lidar2novatel = calib_to_tf_matrix(
                os.path.join(root_dir, cvi_path, cav_path,
                             cur_veh_info['calib_lidar_to_novatel_path'])
            )
            tf_lidar2world = tf_novatel2world @ tf_lidar2novatel
            veh_lidar_pose = pclib.tf2pose(tf_lidar2world)
            veh_pose = pclib.tf2pose(tf_novatel2world)
            veh_lidar_time = float(cur_veh_info['pointcloud_timestamp']) * 1e-6
            veh_lidar_file = os.path.join(cav_lidar_path, frame + '.pcd')
            veh_bbxs_center, _ = load_label(
                os.path.join(root_dir,
                             f"{new_label_path}/new_labels/vehicle-side_label/lidar",
                             frame + '.json'
                             )
            )

            # process infra info
            cur_inf_frame = inf_frames[veh_frames.index(frame)]
            cur_inf_info = inf_info[cur_inf_frame]
            tf_virtuallidar2world = calib_to_tf_matrix(
                os.path.join(root_dir, cvi_path, infra_path,
                             cur_inf_info['calib_virtuallidar_to_world_path'])
            )

            inf_lidar_time = float(cur_inf_info['pointcloud_timestamp']) * 1e-6
            inf_lidar_file = os.path.join(inf_lidar_path, cur_inf_frame + ".pcd")

            # inf_lidar_pose = pclib.tf2pose(tf_infra2ego)
            inf_lidar_pose = pclib.tf2pose(tf_virtuallidar2world)
            inf_label_path = os.path.join(root_dir,
                             f"{cvi_path}/infrastructure-side/label/virtuallidar",
                             cur_inf_frame + '.json')
            inf_bbxs_center, _ = load_label(inf_label_path)

            # process global meta
            coop_label_path = os.path.join(root_dir,
                             f"{new_label_path}/new_labels/cooperative_label/label_world",
                             frame + '.json'
                             )
            world_bbxs_center, world_bbxs_corner = load_label(coop_label_path)
            coop_bbxs_corner =  pclib.rotate_box_corners_with_tf_np(
                np.array(world_bbxs_corner), np.linalg.inv(tf_lidar2world)
            )
            coop_bbxs_center = np.concatenate(
                [np.array(world_bbxs_center)[:, :2],
                 corners_to_boxes_3d(coop_bbxs_corner)],
                axis=1
            ).tolist()

            # if not os.path.exists(inf_label_path):
            #     print('infra label not found.')
            #     inf_bbxs_center = pclib.rotate_box_corners_with_tf_np(
            #         np.array(world_bbxs_corner), np.linalg.inv(tf_virtuallidar2world)
            #     )
            #     inf_bbxs_center = np.concatenate(
            #         [np.array(world_bbxs_center)[:, :2],
            #          corners_to_boxes_3d(inf_bbxs_center)],
            #         axis=1
            #     ).tolist()



            # pcd = point_cloud_from_path(os.path.join(root_dir, veh_lidar_file))
            # points = np.stack([pcd.pc_data[x] for x in 'xyz'], axis=-1)
            # o3d_draw_pcds_bbxs([points], [np.array(veh_bbxs_center)])

            # construct meta dict
            fdict = cs.fdict_template()
            # add cav lidar meta
            cs.update_agent(fdict,
                            agent_id='0',
                            agent_type='cav',
                            agent_pose=veh_pose,
                            gt_boxes=veh_bbxs_center)
            cs.update_agent_lidar(fdict,
                                  agent_id='0',
                                  lidar_id='0',
                                  lidar_pose=veh_lidar_pose,
                                  lidar_time=veh_lidar_time,
                                  lidar_file=veh_lidar_file)
            # add infra lidar meta
            cs.update_agent(fdict,
                            agent_id='1',
                            agent_type='infra',
                            agent_pose=inf_lidar_pose,
                            gt_boxes=inf_bbxs_center)
            cs.update_agent_lidar(fdict,
                                  agent_id='1',
                                  lidar_id='0',
                                  lidar_pose=inf_lidar_pose,
                                  lidar_time=inf_lidar_time,
                                  lidar_file=inf_lidar_file)
            cs.update_frame_bbx(fdict,
                                coop_bbxs_center
                                )# in global coords
            fdict['meta']['ego_id'] = '0'
            fdict['meta']['ego_lidar_pose'] = veh_lidar_pose
            if scenario not in meta_dict:
                meta_dict[scenario] = {}
            meta_dict[scenario][frame] = fdict
        # save meta
        os.makedirs(meta_out_dir, exist_ok=True)
        for scenario, meta in meta_dict.items():
            meta_file = os.path.join(meta_out_dir, f'{scenario}.json')
            save_json(meta, meta_file)
        with open(os.path.join(meta_out_dir, f'{sp}.txt'), 'w') as fh:
            fh.write('\n'.join(list(meta_dict.keys())))


def convert_v2x_seq(root_dir, meta_out_dir):
    split = "test"
    inf_info_file = os.path.join(root_dir, "infrastructure-side/data_info.json")
    inf_info = load_info_to_dict(inf_info_file)
    veh_info_file = os.path.join(root_dir, "vehicle-side/data_info.json")
    veh_info = load_info_to_dict(veh_info_file)
    frame_pairs = load_json(os.path.join(root_dir, "cooperative/data_info.json"))

    meta_dict = {}
    for pdict in frame_pairs:
        scenario = pdict['vehicle_sequence']
        #############################################################
        # processing vehicle meta
        cur_veh_info = veh_info[pdict['vehicle_frame']]
        tf_novatel2world = calib_to_tf_matrix(
            os.path.join(root_dir, "vehicle-side", cur_veh_info['calib_novatel_to_world_path'])
        )
        tf_lidar2novatel = calib_to_tf_matrix(
            os.path.join(root_dir, "vehicle-side", cur_veh_info['calib_lidar_to_novatel_path'])
        )
        tf_lidar2world = tf_novatel2world @ tf_lidar2novatel
        veh_lidar_pose = pclib.tf2pose(tf_lidar2world)
        veh_pose = pclib.tf2pose(tf_novatel2world)

        veh_lidar_time = float(cur_veh_info['pointcloud_timestamp']) * 1e-6
        veh_lidar_file = os.path.join("vehicle-side", cur_veh_info['pointcloud_path'])
        veh_bbxs_center, _ = load_label(
            os.path.join(root_dir, "vehicle-side", cur_veh_info['label_lidar_std_path'])
        )

        ###############################################################
        # process infra info
        cur_inf_info = inf_info[pdict['infrastructure_frame']]
        tf_virtuallidar2world = calib_to_tf_matrix(
            os.path.join(root_dir, "infrastructure-side", cur_inf_info['calib_virtuallidar_to_world_path'])
        )
        inf_lidar_pose = pclib.tf2pose(tf_virtuallidar2world)
        inf_lidar_time = float(cur_inf_info['pointcloud_timestamp']) * 1e-6
        inf_lidar_file = os.path.join("infrastructure-side", cur_inf_info['pointcloud_path'])
        inf_bbxs_center, _ = load_label(
            os.path.join(root_dir, "infrastructure-side", cur_inf_info['label_lidar_std_path'])
        )
        inf_bbxs_center = []

        ###############################################################
        # process global meta
        coop_bbxs_center, _ = load_label(
            os.path.join(root_dir, "cooperative", "label", f"{pdict['vehicle_frame']}.json")
        )

        ###############################################################
        # construct meta dict
        fdict = cs.fdict_template()
        # add cav lidar meta
        cs.update_agent(fdict,
                        agent_id='0',
                        agent_type='cav',
                        agent_pose=veh_pose,
                        gt_boxes=veh_bbxs_center)
        cs.update_agent_lidar(fdict,
                              agent_id='0',
                              lidar_id='0',
                              lidar_pose=veh_lidar_pose,
                              lidar_time=veh_lidar_time,
                              lidar_file=veh_lidar_file)
        # add infra lidar meta
        cs.update_agent(fdict,
                        agent_id='1',
                        agent_type='infra',
                        agent_pose=inf_lidar_pose,
                        gt_boxes=inf_bbxs_center)
        cs.update_agent_lidar(fdict,
                              agent_id='1',
                              lidar_id='0',
                              lidar_pose=inf_lidar_pose,
                              lidar_time=inf_lidar_time,
                              lidar_file=inf_lidar_file)
        cs.update_frame_bbx(fdict,
                            coop_bbxs_center
                            )# in global coords
        fdict['meta']['ego_id'] = '0'
        fdict['meta']['ego_lidar_pose'] = veh_lidar_pose
        if scenario not in meta_dict:
            meta_dict[scenario] = {}
        meta_dict[scenario][pdict['vehicle_frame']] = fdict
    # save meta
    os.makedirs(meta_out_dir, exist_ok=True)
    for scenario, meta in meta_dict.items():
        meta_file = os.path.join(meta_out_dir, f'{scenario}.json')
        save_json(meta, meta_file)
    with open(os.path.join(meta_out_dir, f'{split}.txt'), 'w') as fh:
        fh.write('\n'.join(list(meta_dict.keys())))



if __name__=="__main__":
    root_dir = "/koko/DAIR-V2X"
    meta_out_dir = "/koko/cosense3d/dairv2x"
    # root_dir = "/home/data/DAIR-V2X-Seq/SPD-Example"
    # meta_out_dir = "/home/data/cosense3d/dairv2x_seq"
    convert_v2x_c(root_dir, meta_out_dir)
    # meta_dict = load_meta(os.path.join(meta_out_dir, 'dairv2x'))
    # o3d_play_sequence(meta_dict, root_dir)


