from plyfile import PlyData
from matplotlib import cm

from cosense3d.dataset.toolkit.opv2v import *
from cosense3d.utils.vislib import o3d_draw_pcds_bbxs
from cosense3d.utils.pclib import save_cosense_ply


jet = cm.get_cmap('jet')


def read_ply(filename, properties=None):
    ply = PlyData.read(filename)
    data = ply['vertex']
    properties_from_file = [p.name for p in ply.elements[0].properties]
    if properties is None:
        properties = properties_from_file
    else:
        for p in properties:
            assert p in properties_from_file, f"Property '{p}' not found."
    data_dict = {}
    for p in properties:
        data_dict[p] = np.array(data[p])

    return data_dict


def get_local_boxes3d(objects_dict, ref_pose, order):
    output_dict = {}
    # add ground truth boxes at cav local coordinate
    project_world_objects(objects_dict,
                          output_dict,
                          ref_pose,
                          order)
    boxes_local = []
    velos = []
    for object_id, object_content in output_dict.items():
        if object_content['ass_id'] != -1:
            object_id = object_content['ass_id']
        else:
            object_id = object_id
        object_bbx = object_content['coord']
        if order == 'hwl':
            object_bbx = object_bbx[:, [0, 1, 2, 5, 4, 3, 6]]
        boxes_local.append(
            [object_id, 0, ] +
            object_bbx[0, :6].tolist() +
            [0, 0, object_bbx[0, 6]]
        )
        if 'velo' in object_content and object_content['velo'] is not None:
            velos.append(object_content['velo'].tolist())

    return boxes_local, velos


def read_ply_to_dict(f):
    data = read_ply(f)
    timestamp = os.path.basename(f).split('.')[:-1]
    timestamp = int(timestamp[0]) * 0.05 + int(timestamp[1]) * 0.01
    timestamp = np.ones_like(data['x']) * timestamp
    data['time'] = timestamp.astype(np.float32)
    return data


def read_frame_plys(path, frame, prev_frame=None, time_offset=0):
    data_list = []
    files = []
    if prev_frame is not None:
        files_prev_frame = [f'{prev_frame}.{i}.ply' for i in range(10 - time_offset, 10)]
        files.extend(files_prev_frame)
    files_cur_frame = [f'{frame}.{i}.ply' for i in range(0, 10 - time_offset)]
    files.extend(files_cur_frame)
    for f in files:
        data_list.append(read_ply_to_dict(os.path.join(path, f)))
    data = {k: np.concatenate([d[k] for d in data_list], axis=0) for k in data_list[0]}
    return data


def load_frame_data(scene_dir, cavs, frame):
    ego_id = cavs[0]
    yaml_file = os.path.join(scene_dir, ego_id, f'{frame}.5.yaml')
    meta = load_yaml(yaml_file)
    gt_boxes, velos = get_local_boxes3d(meta['vehicles'],
                                        meta['lidar_pose'], 'lwh')
    ego_pose = meta['lidar_pose']

    points_list = []
    time_list = []
    for cav in cavs:
        cav_dir = os.path.join(scene_dir, cav)
        data = read_frame_plys(cav_dir, frame)
        points = np.stack([data[k] for k in 'xyz'], axis=-1)
        times = (data['time'] - data['time'].min()) * 10
        lidar_pose = load_yaml(
            os.path.join(scene_dir, cav, f'{frame}.5.yaml'))['lidar_pose']
        transform = x1_to_x2(lidar_pose, ego_pose)
        points = (transform[:3, :3] @ points.T + transform[:3, 3].reshape(3, 1)).T
        points_list.append(points)
        time_list.append(times)
    points = np.concatenate(points_list, axis=0)
    times = np.concatenate(time_list, axis=0)
    return points, times, gt_boxes, velos


def opv2vt_to_cosense(data_dir, split, data_out_dir, meta_out_dir):
    order = 'lwh'
    split_dir = os.path.join(data_dir, split)
    scenes = os.listdir(split_dir)
    with open(os.path.join(meta_out_dir, f'{split}.txt'), 'w') as fh:
        fh.write('\n'.join(scenes))
    for s in scenes:
        scene_dir = os.path.join(split_dir, s)
        sdict = {}
        cavs = sorted([x for x in os.listdir(scene_dir)
                       if os.path.isdir(os.path.join(scene_dir, x))])
        time_offset = np.random.randint(0, 5, len(cavs))
        time_offset[0] = 0
        ego_id = cavs[0]
        frames = sorted([x.split(".")[0] for x in os.listdir(
            os.path.join(scene_dir, cavs[0])) if '.0.ply' in x])
        for i, f in tqdm.tqdm(enumerate(frames[1:-1])):
            frame_mid_time = int(f) * 0.05 + 0.05
            fdict = cs.fdict_template()
            ego_lidar_pose = None
            object_id_stack = []
            object_velo_stack = []
            object_stack = []
            for j, cav_id in enumerate(cavs):
                cur_data_out_dir = os.path.join(data_out_dir, split, s, cav_id)
                os.makedirs(cur_data_out_dir, exist_ok=True)
                yaml_file = os.path.join(scene_dir, cav_id, f'{f}.5.yaml')
                params = load_yaml(yaml_file)
                cs.update_agent(fdict, cav_id, agent_type='cav', agent_pose=params['true_ego_pos'])
                # update_cam_params(params, fdict, cav_id, s, f)

                if cav_id == ego_id:
                    ego_lidar_pose = params['lidar_pose']

                # get cav lidar pose in cosense format
                cs.update_agent(fdict, cav_id, 'cav')
                cs.update_agent_lidar(fdict, cav_id, 0,
                                      lidar_pose=opv2v_pose_to_cosense(params['lidar_pose']),
                                      lidar_file=os.path.join(s, cav_id, f'{f}.ply'))
                # save lidar files
                data = read_frame_plys(os.path.join(scene_dir, cav_id), f,
                                       prev_frame=frames[i], time_offset=time_offset[j])
                save_cosense_ply(data, os.path.join(cur_data_out_dir, f'{f}.ply'))

                objects_dict = params['vehicles']
                output_dict = {}
                glob_ref_pose = ego_lidar_pose
                local_ref_pose = params['lidar_pose']

                # update_local_boxes3d(fdict, objects_dict, local_ref_pose, order, split_dir, cav_id)
                # update_2d_bboxes(fdict, cav_id, params['lidar_pose'], data_dir)

                # add gt boxes in ego coordinates as global boxes of cosense3d format
                project_world_objects(objects_dict,
                                      output_dict,
                                      glob_ref_pose,
                                      order)

                for object_id, object_content in output_dict.items():
                    if object_content['ass_id'] != -1:
                        object_id_stack.append(object_content['ass_id'])
                    else:
                        object_id_stack.append(object_id + 100 * int(cav_id))
                    if object_content['velo'] is not None:
                        object_velo_stack.append(object_content['velo'])
                    object_stack.append(object_content['coord'])

            # exclude all repetitive objects
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]
            if len(object_velo_stack) > 0:
                object_velo_stack = np.vstack(object_velo_stack)
                object_velo_stack = object_velo_stack[unique_indices]
            if order == 'hwl':
                object_stack = object_stack[:, [0, 1, 2, 5, 4, 3, 6]]

            cosense_bbx_center = np.zeros((len(object_stack), 11))
            cosense_bbx_center[:, 0] = np.array(object_id_stack)[unique_indices]
            cosense_bbx_center[:, 2:8] = object_stack[:, :6]
            cosense_bbx_center[:, 10] = object_stack[:, 6]
            cs.update_frame_bbx(fdict, cosense_bbx_center.tolist())
            fdict['agents'].pop(0)  # remove template agent

            fdict['meta']['ego_id'] = ego_id
            fdict['meta']['ego_lidar_pose'] = opv2v_pose_to_cosense(ego_lidar_pose)
            fdict['meta']['global_bbox_time'] = np.full(len(cosense_bbx_center), frame_mid_time).tolist()
            if len(object_velo_stack) == len(object_stack):
                fdict['meta']['bbx_velo_global'] = object_velo_stack.tolist()

            sdict[f] = fdict

        save_json(sdict, os.path.join(meta_out_dir, f'{s}.json'))


def vis_frame_data():
    scene_dir = "/koko/OPV2V/temporal_dump/test/2021_08_18_19_48_05"
    cavs = sorted([x for x in os.listdir(scene_dir)
                   if os.path.isdir(os.path.join(scene_dir, x))])
    frames = sorted([x.split(".")[0] for x in os.listdir(
        os.path.join(scene_dir, cavs[0])) if '.0.ply' in x])
    for f in frames[::10]:
        points, times, local_boxes, velos = load_frame_data(scene_dir, cavs, f)
        pcd = o3d.geometry.PointCloud()
        # color_inds = np.round(times).astype(int)
        colors = jet(times)[:, :3]
        o3d_draw_pcds_bbxs([points], [np.array(local_boxes)],
                           pcds_colors=[colors])


if __name__=="__main__":
    opv2vt_to_cosense(
        "/koko/OPV2V/temporal_dump",
        "test",
        "/koko/OPV2V/temporal",
        "/koko/cosense3d/opv2v_temporal"
    )
    # vis_frame_data()