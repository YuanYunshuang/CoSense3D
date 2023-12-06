import random

import torch
from plyfile import PlyData
from matplotlib import colormaps
from multiprocessing import Pool
import torch_scatter

from cosense3d.dataset.toolkit.opv2v import *
from cosense3d.utils.vislib import o3d_draw_pcds_bbxs
from cosense3d.utils.pclib import save_cosense_ply
from cosense3d.utils.misc import load_json, update_dict
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.utils.common import cat_coor_with_idx


# jet = cm.get_cmap('jet')
jet = colormaps['jet']


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
            # TODO adapt velos

    return boxes_local, velos


def fdict_update_local_boxes(fdict, cav_id, local_boxes):
    boxes = [v['box'] for v in local_boxes.values()]
    velos = [v['velo'] for v in local_boxes.values()]
    num_pts = [v['num_pts'] for v in local_boxes.values()]
    cs.update_agent(fdict, cav_id, gt_boxes=boxes)
    cs.update_agent(fdict, cav_id, velos=velos)
    cs.update_agent(fdict, cav_id, num_pts=num_pts)


def read_ply_to_dict(f):
    data = read_ply(f)
    timestamp = os.path.basename(f).split('.')[:-1]
    timestamp = int(timestamp[0]) * 0.05 + int(timestamp[1]) * 0.01
    timestamp = np.ones_like(data['x']) * timestamp
    data['time'] = timestamp.astype(np.float32)
    return data


def read_sub_frame(f):
    pcd_dict = read_ply_to_dict(f + '.ply')
    params_ = load_yaml(f + '_objects.yaml')
    params = load_yaml(f + '.yaml')
    update_dict(params, params_)
    gt_boxes, velos = get_local_boxes3d(params['vehicles'],
                                        params['lidar_pose'], 'lwh')
    gt_boxes = np.array(gt_boxes)
    velos = np.array(velos)
    points = np.stack([pcd_dict[x] for x in 'xyz'], axis=-1)
    points = points[pcd_dict['ObjTag'] == 10]
    return gt_boxes, velos, pcd_dict, points


def parse_sub_frame(f):
    pcd_dict = read_ply_to_dict(f + '.ply')
    params = load_yaml(f + '.yaml')
    gt_boxes, velos = get_local_boxes3d(params['vehicles'],
                                        params['lidar_pose'], 'lwh')
    gt_boxes = np.array(gt_boxes)
    velos = np.array(velos)
    points = np.stack([pcd_dict[x] for x in 'xyz'], axis=-1)
    pts_mask = points_in_boxes_cpu(torch.from_numpy(points),
                                   torch.from_numpy(gt_boxes)[:, [2, 3, 4, 5, 6, 7, 10]])
    num_pts = pts_mask.sum(dim=-1).numpy()
    box_mask = num_pts > 0
    gt_boxes = gt_boxes[box_mask].tolist()
    velos = velos[box_mask].tolist()
    num_pts = num_pts[box_mask].tolist()

    # update boxes dict
    # for i, box in enumerate(gt_boxes):
    #     id = int(box[0])
    #     if id not in boxes:
    #         boxes[id] = {
    #             'box': box,
    #             'velo': velos[i],
    #             'num_pts': num_pts[i]
    #         }
    #     else:
    #         if boxes[id]['num_pts'] < num_pts[i]:
    #             boxes[id] = {
    #                 'box': box,
    #                 'velo': velos[i],
    #                 'num_pts': num_pts[i] + boxes[id]['num_pts']
    #             }
    #         else:
    #             boxes[id]['num_pts'] += num_pts[i]

    return (gt_boxes, velos, num_pts, pcd_dict)


def read_frame_plys_boxes(path, frame, prev_frame=None, time_offset=0, parse_boxes=True):
    data_list = []
    files = []
    if prev_frame is not None:
        files_prev_frame = [f'{prev_frame}.{i}' for i in range(10 - time_offset, 10)]
        files.extend(files_prev_frame)
    files_cur_frame = [f'{frame}.{i}' for i in range(0, 10 - time_offset)]
    files.extend(files_cur_frame)
    files = [os.path.join(path, f) for f in files]
    boxes = {}

    with Pool(5) as p:
        res = p.map(read_sub_frame, files)

    pcd_dict = {k: np.concatenate([d[2][k] for d in res], axis=0) for k in res[0][2]}
    boxes_tensor = cat_coor_with_idx([torch.from_numpy(x[0]) for x in res])
    points_tensor = cat_coor_with_idx([torch.from_numpy(x[3]) for x in res])
    velos = np.stack([x[1] for x in res], axis=0)

    _, pts_idx_of_box = points_in_boxes_gpu(points_tensor.float().cuda(),
                                            boxes_tensor[:, [0, 3, 4, 5, 6, 7, 8, 11]].float().cuda(),
                                            batch_size=len(res))

    pts_idx_of_box = pts_idx_of_box[pts_idx_of_box >= 0]
    cnt = torch.ones_like(pts_idx_of_box)
    num_pts_in_box = cnt.new_zeros(len(boxes_tensor))
    torch_scatter.scatter_add(cnt, pts_idx_of_box, out=num_pts_in_box, dim=0)
    num_pts_in_box = num_pts_in_box.reshape(10, -1).cpu()
    num_pts = num_pts_in_box.sum(dim=0)
    boxes_tensor = boxes_tensor.view(10, -1, boxes_tensor.shape[-1])[..., 1:]
    max_inds = num_pts_in_box.max(dim=0).indices
    boxes_selected = boxes_tensor[max_inds, torch.arange(len(max_inds))]
    velos = velos[max_inds.numpy(), np.arange(velos.shape[1])]

    return pcd_dict, boxes_selected, velos, num_pts


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
        data = read_frame_plys_boxes(cav_dir, frame, parse_boxes=False)[0]
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
    time_offsets = load_json(os.path.join(data_out_dir, 'time_offsets.json'))
    split_dir = os.path.join(data_dir, split)
    scenes = sorted(os.listdir(split_dir))
    with open(os.path.join(meta_out_dir, f'{split}.txt'), 'w') as fh:
        fh.write('\n'.join(scenes))
    for s in scenes:
        print(s)
        scene_dir = os.path.join(split_dir, s)
        sdict = {}
        cavs = sorted([x for x in os.listdir(scene_dir)
                       if os.path.isdir(os.path.join(scene_dir, x))])
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
                data, local_boxes, num_pts = read_frame_plys_boxes(os.path.join(scene_dir, cav_id), f,
                                       prev_frame=frames[i], time_offset=time_offsets[s][cav_id])
                save_cosense_ply(data, os.path.join(cur_data_out_dir, f'{f}.ply'))

                objects_dict = params['vehicles']
                output_dict = {}
                glob_ref_pose = ego_lidar_pose
                local_ref_pose = params['lidar_pose']

                fdict_update_local_boxes(fdict, cav_id, local_boxes)
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
        del sdict


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


def gen_time_offsets(data_dir):
    out_dict = {}
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        scenes = os.listdir(split_dir)
        for s in scenes:
            out_dict[s] = {}
            scene_dir = os.path.join(split_dir, s)
            cavs = sorted([x for x in os.listdir(scene_dir)
                           if os.path.isdir(os.path.join(scene_dir, x))])
            for i, cav in enumerate(cavs):
                if i == 0:
                    out_dict[s][cav] = 0
                else:
                    out_dict[s][cav] = random.randint(0, 5)
    save_json(out_dict, os.path.join(data_dir, f'time_offsets.json'))


if __name__=="__main__":
    # gen_time_offsets("/koko/OPV2V/temporal")
    opv2vt_to_cosense(
        "/koko/OPV2V/temporal_dump",
        "train",
        "/koko/OPV2V/temporal",
        "/koko/cosense3d/opv2v_temporal"
    )
    # vis_frame_data()