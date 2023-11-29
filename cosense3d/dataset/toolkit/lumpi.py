import glob
import os
import shutil

import numpy as np
import tqdm
from plyfile import PlyData, PlyElement
import cv2

from cosense3d.utils.misc import load_json, save_json
from cosense3d.utils import pclib
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs
from cosense3d.dataset.toolkit.kitti import type_cosense2kitti
from cosense3d.dataset.data_utils import save_video_to_imgs

type_lumpi2cosense = {
    0: 'human.pedestrian',
    1: 'vehicle.car',
    2: 'vehicle.cyclist',
    3: 'vehicle.motorcycle',
    4: 'vehicle.bus',
    5: 'vehicle.truck',
    6: 'unknown',
}

H, W = 640, 960
scale_factors = {
    '5': {'fx': W / 1920, 'fy': H / 1080},
    '6': {'fx': W / 1640, 'fy': H / 1232},
    '7': {'fx': W / 1640, 'fy': H / 1232},
    '8': {'fx': W / 1640, 'fy': H / 1232},
    '9': {'fx': W / 1920, 'fy': H / 1080},
    '10': {'fx': W / 1920, 'fy': H / 1080},
}

img_sizes = {
    '5': [1920, 1080],
    '6': [1640, 1232],
    '7': [1640, 1232],
    '8': [1640, 1640],
    '9': [1920, 1080],
    '10': [1920, 1080],
}

cam_nom = {
    '5': {'mean': [0.34550, 0.38377, 0.39296], 'std': [0.24716, 0.22809, 0.21530]},
    '6': {'mean': [0.46485, 0.48167, 0.46741], 'std': [0.23009, 0.22193, 0.22100]},
    '7': {'mean': [0.49440, 0.50039, 0.52180], 'std': [0.19495, 0.17650, 0.17644]}
}

measurement_cams = {
    '0': ['8', '9', '10'],
    '1': ['8', '9', '10'],
    '2': ['8', '9', '10'],
    '3': ['8', '9', '10'],
    '4': ['6', '7'],
    '5': '567',
    '6': '567'
}


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


def read_gt_boxes3d(filename):
    """
    Columns:
    0  - time (sec),
    1  - objId,
    2  - 2dX,
    3  - 2dY,
    4  - 2dWidth,
    5  - 2dHeight,
    6  - confidenceScore,
    7  - classId,
    8  - visibility (always 4),
    9  - 3dX,
    10 - 3dY,
    11 - 3dZ,
    12 - 3dLength,
    13 - 3dWidth,
    14 - 3dHeight,
    15 - heading
    :param filename: input file full path
    :return: a dict in the format of {
    frame (int): 3D bounding boxes with columns [cls, x, y, z, dx, dy, dz, roll, pitch,
    yaw](np.array)
    }.
    """
    data = np.genfromtxt(filename, delimiter=',')
    frames = (data[:, 0] * 10).astype(int)
    mask = frames >= 0
    frames = frames[mask]
    data = data[mask]
    gt_boxes = {}
    for f in np.unique(frames):
        boxes = data[frames == f, 9:]
        id_cls = data[frames == f][:, [1, 7]]
        cls_cosense = np.array([
            cs.OBJ_NAME2ID[type_lumpi2cosense[cls]] \
            for cls in id_cls[:, 1] if cls != -2
        ])
        id_cls[:, 1] = cls_cosense
        zeros = np.zeros_like(boxes[:, :2])
        gt_boxes[f] = np.concatenate(
            [id_cls, boxes[:, :6], zeros, boxes[:, 6:]], axis=1
        )
    return gt_boxes


def train_test_split_tmp():
    root_dir = "/media/hdd/yuan/koko/data/LUMPI/opv2v_fmt/train"
    root_dir2 = "/media/hdd/yuan/koko/data/LUMPI/cosense_fmt/measurement4"
    train_path = "/media/hdd/yuan/koko/data/LUMPI/opv2v_fmt/train"
    test_path = "/media/hdd/yuan/koko/data/LUMPI/opv2v_fmt/test"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    scenarios = ['measurement4_0', 'measurement4_1']
    # for s in scenarios:
    #     for cav in range(5):
    #         os.makedirs(os.path.join(train_path, s, str(cav)), exist_ok=True)
    for i in tqdm.tqdm(range(500, 1000)):

        if i <= 500:
            scn = scenarios[0]
            path = train_path
        else:
            scn = scenarios[1]
            path = test_path
        for cav in range(5):
            pcd_file = os.path.join(root_dir2, str(cav), f"{i:06d}.bin")
            pcd_file_out = os.path.join(path, scn, str(cav), f"{i:06d}.bin")
            shutil.copy(pcd_file, pcd_file_out)
            # shutil.copy(pcd_file.replace('pcd', 'yaml'),
            #             pcd_file_out.replace('pcd', 'yaml'))


def cam_param_from_meta(meta_in):
    cam_params = {}
    for device_id in '567':
        for scenario_id in '456':
            session_id = meta_in['device'][device_id][scenario_id]
            session_meta = meta_in['session'][str(session_id)]
            if scenario_id not in cam_params:
                cam_params[scenario_id] = {}
            cam_params[scenario_id][device_id] = {
                'pose': [x[0] for x in session_meta['tvec'] + session_meta['rvec']],
                'intrinsic': update_cam_intri(session_meta['intrinsic'], device_id),
                'extrinsic': session_meta['extrinsic'],
                'distortion': session_meta['distortion'],
                'fps': session_meta['fps'],
            }
    return cam_params


def parse_img_data(root_dir, data_out_path):
    for s, cam_ids in measurement_cams.items():
        scenario = f'measurement{s}'
        for i in cam_ids:
            os.makedirs(os.path.join(data_out_path, scenario, i), exist_ok=True)
            save_video_to_imgs(
                os.path.join(root_dir, scenario, 'cam', i, 'video.mp4'),
                os.path.join(data_out_path, scenario, i),
                {'fx': 1, 'fy': 1}
            )


def update_cam_intri(intrinsic, cam_id):
    intrinsic[0][0] *= scale_factors[cam_id]['fx']
    intrinsic[1][1] *= scale_factors[cam_id]['fy']
    intrinsic[0][2] *= scale_factors[cam_id]['fx']
    intrinsic[1][2] *= scale_factors[cam_id]['fy']
    return intrinsic


def copy_img_to_sustech(cosense_path, meta_file, sustech_path, measurement):
    os.makedirs(os.path.join(sustech_path, 'calib', 'camera'), exist_ok=True)
    session_meta = load_json("/koko/LUMPI/meta.json")
    cam_params = {}

    for cam in measurement_cams[measurement]:
        if measurement not in session_meta['device'][cam]:
            continue
        os.makedirs(os.path.join(sustech_path, 'camera', cam), exist_ok=True)
        session_id = str(session_meta['device'][cam][measurement])
        intrinsic = session_meta['session'][session_id]['intrinsic']
        intrinsic = update_cam_intri(intrinsic, cam)
        extrinsic = session_meta['session'][session_id]['extrinsic']
        rvec = np.array(session_meta['session'][session_id]['rvec'])
        tvec = np.array(session_meta['session'][session_id]['tvec'])
        cam_params[cam] = session_meta['session'][session_id]

        rot = cv2.Rodrigues(rvec.reshape(3))[0]
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = tvec.reshape(3)
        cam_calib = {
            'extrinsic': [ x for xx in T.tolist() for x in xx],
            'intrinsic': [ x for xx in intrinsic for x in xx]
        }
        save_json(cam_calib, os.path.join(sustech_path, 'calib', 'camera', f'{cam}.json'))

    meta = load_json(meta_file)
    for fi, fdict in tqdm.tqdm(meta.items()):
        m = os.path.basename(meta_file).split('_')[0]
        for cam_id in measurement_cams[measurement]:
            cdict = fdict['agents'].get(cam_id, False)
            if not cdict:
                continue
            else:
                files = cdict['camera']['0']['filenames']
                file_mid = files[len(files) // 2]
                img_file = os.path.join(cosense_path, file_mid.replace('png', 'jpg'))
                # img = cv2.imread(img_file)
                # img = cv2.undistort(img, np.array(cam_params[cam_id]['intrinsic']),
                #                     np.array(cam_params[cam_id]['distortion']))
                # cv2.imwrite(os.path.join(sustech_path, 'camera', cam_id, f'{fi}.jpg'), img)
                shutil.copy(
                    img_file,
                    os.path.join(sustech_path, 'camera', cam_id, f'{fi}.jpg')
                )


def crop_lidar_sustech(sustech_path):
    visibility = cv2.imread("/koko/LUMPI/visibility.png").any(axis=-1)
    lidar_files = glob.glob(os.path.join(sustech_path, 'lidar0', '*.bin'))
    os.makedirs(os.path.join(sustech_path, 'lidar_crop'), exist_ok=True)
    r = 75.2
    for f in lidar_files:
        points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        mask = (points[:, 0] > -r) & (points[:, 0] < r) & \
               (points[:, 1] > -r) & (points[:, 1] < r) & \
               (points[:, 2] > -3) & (points[:, 2] < 1)

        points = points[mask]
        indices = ((points[:, :2] + r) / 0.2).astype(int)
        mask = visibility[indices[:, 0], indices[:, 1]]
        points = points[mask]
        points.tofile(f.replace('lidar0', 'lidar_crop'))


def cal_img_statistics(cosense_dir):
    from cosense3d.utils.data_statistics import StatsRecorder
    for cam in '567':
        stats = StatsRecorder()
        files = glob.glob(os.path.join(cosense_dir, 'measurement5', cam, '*.jpg'))

        for f in tqdm.tqdm(sorted(files)[::30]):
            img = cv2.imread(f) / 255
            stats.update(img.reshape(-1, 3))
        mean_str = (', '.join(['{:.5f}'] * len(stats.mean))).format(*stats.mean)
        std_str = (', '.join(['{:.5f}'] * len(stats.std))).format(*stats.std)
        print(f"Cam{cam} Means: [{mean_str}]")
        print(f"Cam{cam}  Stds: [{std_str}]")


def cal_visibility_map():
    meta = load_json("/koko/LUMPI/meta.json")
    r = 75.2
    x = np.arange(-r + 0.1, r, 0.2)
    z = np.arange(-3 + 0.5, 1, 1)

    xyz = np.stack(np.meshgrid(x, x, z), axis=-1)
    visibility = []

    for cam_id in '567':
        session_id = str(meta['device'][cam_id]['5'])
        intrinsic = np.array(meta['session'][session_id]['intrinsic'])
        intrinsic = update_cam_intri(intrinsic, cam_id)
        extrinsic = np.array(meta['session'][session_id]['extrinsic'])
        dist_coeff = np.array(meta['session'][session_id]['distortion'])
        rvec = np.array(meta['session'][session_id]['rvec'])
        tvec = np.array(meta['session'][session_id]['tvec'])

        # intrinsic[0, 0] *= scale_factors[cam_id]['fx']
        # intrinsic[1, 1] *= scale_factors[cam_id]['fy']
        # intrinsic[0, 2] *= scale_factors[cam_id]['fx']
        # intrinsic[1, 2] *= scale_factors[cam_id]['fy']

        # rot = cv2.Rodrigues(rvec)[0]
        # T = np.eye(4)
        # T[:3, :3] = rot
        # T[:3, 3] = tvec
        #
        # img_points = T @ xyz.T
        # img_points = intrinsic @ img_points[:3]

        imgPoints = cv2.projectPoints(xyz.reshape(-1, 3), rvec, tvec, intrinsic, dist_coeff)
        imgPoints = imgPoints[0][:, 0, :]
        ix = (imgPoints[:, 0] < img_sizes[cam_id][1]) & (imgPoints[:, 0] >= 0)
        iy = (imgPoints[:, 1] < img_sizes[cam_id][0]) & (imgPoints[:, 1] >= 0)

        mask = (ix & iy).reshape(xyz.shape[:3]).all(axis=-1).astype(float)
        visibility.append(mask)

    visibility = np.stack(visibility, axis=-1)
    cv2.imwrite("/koko/LUMPI/visibility.png", visibility)


def update_extrinsics(meta_path):
    meta_files = glob.glob(os.path.join(meta_path, "*.json"))
    for f in meta_files:
        meta = load_json(f)
        for frame, fdict in meta.items():
            for cam in '567':
                if cam in fdict['agents']:
                    params = fdict['agents'][cam]['camera']['0']
                    rvec = np.array(params['pose'][3:])
                    tvec = np.array(params['pose'][:3])
                    rot = cv2.Rodrigues(rvec)[0]
                    T = np.eye(4)
                    T[:3, :3] = rot
                    T[:3, 3] = tvec.reshape(3)
                    fdict['agents'][cam]['camera']['0']['extrinsic_lidar2cam'] = T.tolist()
        save_json(meta, f)


def write_kitti_label(labels, label_out_file):
    with open(label_out_file, 'w') as fh:
        for l in labels:
            # kitti label format
            cosense_type = cs.OBJ_ID2NAME[l[1]]
            type = [type_cosense2kitti[cosense_type]]

            trancated = ['0']
            occluded = ['0']
            alpha = [f"{np.arctan2(l[3], l[2]):.2f}"]
            bbox = ['0'] * 4
            dimensions = [f"{l[x]:.2f}" for x in [7, 6, 5]]  # hwl
            l[4] -= l[7] / 2
            location = [f"{l[x]:.2f}" for x in [2, 3, 4]]  # in cam coor
            rotation_y = [f"{-l[10] - np.pi / 2:.2f}"]
            ls = type + trancated + occluded + alpha + bbox + dimensions + \
                 location + rotation_y
            line = " ".join(ls)
            fh.write(line)
            fh.write('\n')


def load_meta(meta_dir):
    meta = {}
    files = glob.glob(os.path.join(meta_dir, '*.json'))
    for f in files:
        data = load_json(f)
        scenario = os.path.basename(f)[:-5]
        if scenario not in meta:
            meta[scenario] = data
        else:
            meta[scenario].update(data)
    return meta


def update_meta_boxes(meta_old_dir, meta_new_dir):
    meta_old = load_meta(meta_old_dir)
    meta_new = load_meta(meta_new_dir)
    for s, sdict in meta_new.items():
        for f, fdict in sdict.items():
            if s in meta_old and f in meta_old[s]:
                if 'meta' in fdict:
                    meta_old[s][f]['meta']['bbx_center_global'] = \
                        fdict['meta']['bbx_center_global']
                else:
                    meta_old[s][f]['meta']['bbx_center_global'] = []
                for a, adict in fdict['agents'].items():
                    if 'gt_boxes' in adict:
                        meta_old[s][f]['agents'][a].update(adict)
    for s, sdict in meta_old.items():
        save_json(sdict, os.path.join(meta_old_dir, f'{s}.json'))


def convert_to_kitti(root_dir, kitti_dir, parse_img=False):
    meta_in = load_json(os.path.join(root_dir, '../meta.json'))

    # make data split
    split = {
        # 'train': ['measurement4_0'],
        # 'validat': ['measurement4_1'],
        'test': [0, 1, 2, 3],
    }
    os.makedirs(os.path.join(kitti_dir, 'ImageSets'), exist_ok=True)

    for sp, msrs in split.items():
        # make kitti out dirs
        out_dir = os.path.join(kitti_dir, sp + "ing")
        for dir_name in ['velodyne', 'image_2', 'label_2', 'calib']:
            os.makedirs(os.path.join(out_dir, dir_name), exist_ok=True)

        # write split file
        frames = []
        for msr in msrs:
            files = glob.glob(os.path.join(root_dir, f"measurement{msr}", "lidar0", "*.ply"))
            frames.extend([str(msr) + os.path.basename(f)[1:6] for f in files])
        with open(os.path.join(kitti_dir, 'ImageSets', f"{sp}.txt"), 'w') as fh:
            fh.write("\n".join(sorted(frames)))

        # read GT files if exists
        GTs = {}
        for msr in msrs:
            gt_file = os.path.join(root_dir, f"measurement{msr}", "GT.csv")
            if os.path.exists(gt_file):
                GTs[str(msr)] = read_gt_boxes3d(gt_file)

        # create kitti frame data
        for f in tqdm.tqdm(frames):
            msr = f[0]
            frame = f"{int(f[1:]):06d}"
            # copy lidar0 file
            lidar_file = os.path.join(root_dir, f"measurement{msr}", "lidar0", f"{frame}.ply")
            data = read_ply(lidar_file)
            pcd = np.stack([data[t] for t in \
                            ('x', 'y', 'z', 'intensity')], axis=1)
            lidar_file_out = os.path.join(out_dir, "velodyne", f"{f}.bin")
            pcd.tofile(lidar_file_out)
            # TODO: copy image data and calib data
            # copy label
            if msr in GTs.keys():
                labels = GTs[msr][frame]
                kitti_lable_file = os.path.join(out_dir, "label_2", f"{f}.txt")
                write_kitti_label(labels, kitti_lable_file)


def convert_to_cosense3d(data_dir, meta_out_dir, data_out_dir, parse_img=False):
    meta_in = load_json(os.path.join(data_dir, 'meta.json'))
    # meta_in['experiment'] = {'4': meta_in['experiment']['4']}
    for exp, info in meta_in['experiment'].items():
        if exp != '4':
            continue
        # parse devices for current measurement
        device_meta = {}
        session_to_device_id = - np.ones(max([int(x) for x in meta_in['session'].keys()]))
        for device_id, session_id in info.items():
            device_meta[device_id] = meta_in['session'][str(session_id)]
            session_to_device_id[session_id] = int(device_id)
            device_type = meta_in['device'][device_id]['type']
            device_data_dir = os.path.join(data_out_dir, f'measurement{exp}', device_id)
            device_meta[device_id]['out_dir'] = device_data_dir
            os.makedirs(device_data_dir, exist_ok=True)
        if exp == '4':
            device_meta.pop('5')
        cams = {k:v for k, v in device_meta.items() if v['type']=='camera'}
        lidars = {k:v for k, v in device_meta.items() if v['type']=='lidar'}

        # parse lidar files of current measurement
        lidar_dir = os.path.join(data_dir, f'measurement{exp}', 'lidar')
        lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.ply')))
        if exp == '4':
            lidar_files = lidar_files[:1000]
        else:
            lidar_files = lidar_files[:300]

        # for caching cam meta
        for cam_id, cam_info in cams.items():
            cam_info['cam_frames'] = []
            cam_info['lidar_frames'] = []
            cam_info['times'] = []

        # retrive lidar data
        for lf in tqdm.tqdm(lidar_files):
            data = read_ply(lf)
            data['id'] = session_to_device_id[data['id']]
            data.pop('azimuth')
            data.pop('distance')
            assert (data['id'] < 0).sum() == 0
            # save lidar data
            fields = {'x': 'f4', 'y': 'f4', 'z': 'f4', 'intensity': 'u1', 'time': 'u4'}
            for lidar_id in np.unique(data['id']):
                lidar_id = int(lidar_id)
                mask = data['id'] == lidar_id
                vertex_data = list(zip(*[data[k][mask] for k, v in fields.items()]))
                vertex_type = [(k, v) for k, v in fields.items()]
                vertex = np.array(vertex_data, dtype=vertex_type)
                el = PlyElement.describe(vertex, 'vertex')
                PlyData([el]).write(os.path.join(device_meta[str(lidar_id)]['out_dir'], os.path.basename(lf)))

            # get camera frames and times
            for cam_id, cam_info in cams.items():
                cam_frame = round((data['time'].min() + data['time'].max()) / 2 * 1e-6 * cam_info['fps'])
                cam_time = round(cam_frame / cam_info['fps'] * 1e6)
                cam_info['cam_frames'].append(cam_frame)
                cam_info['lidar_frames'].append(os.path.basename(lf)[:-4])
                cam_info['times'].append(cam_time)

        # retrive camera data
        for cam_id, cam_info in cams.items():
            cap = cv2.VideoCapture(os.path.join(data_dir, f'measurement{exp}', 'cam', cam_id, 'video.mp4'))
            # Loop through the video frames
            count = 0
            while cap.isOpened():
                # Read the frame
                print(f'Image: {count}', end='\r')
                ret, img = cap.read()
                if count > max(cam_info['cam_frames']):
                    break
                if count not in cam_info['cam_frames']:
                    print(f'Skip image: {count}', end='\r')
                    count += 1
                    continue

                # If the frame was read successfully
                if ret:
                    lidar_frame = cam_info['lidar_frames'][cam_info['cam_frames'].index(count)]
                    img_file = os.path.join(device_meta[cam_id]['out_dir'], f'{lidar_frame}.jpg')
                    # img = cv2.resize(img, None, fx=scale_factors['fx'], fy=scale_factors['fy'])
                    # if not os.path.exists(img_file):
                    cv2.imwrite(img_file, img)
                    count += 1
                else:
                    # Break the loop if the frame was not read successfully
                    break

            # Release the video capture object and close all windows
            cap.release()

        # construct cosense meta data
        sdict = {}
        for lf in lidar_files:
            frame = os.path.basename(lf)[:-4]
            time = cams
            fdict = cs.fdict_template()
            for lidar_id, lidar_info in lidars.items():
                cs.update_agent(fdict, str(lidar_id), 'infra')
                cs.update_agent_lidar(
                    fdict,
                    agent_id=str(lidar_id),
                    lidar_id=0,
                    lidar_pose=[0,] * 6,
                    lidar_file=os.path.join(f'measurement{exp}', lidar_id, f'{frame}.ply')
                )

            for cam_id, cam_info in cams.items():
                cs.update_agent(fdict, str(cam_id), 'infra')
                intrinsic = cam_info['intrinsic']
                rvec = np.array(cam_info['rvec'])
                tvec = np.array(cam_info['tvec'])

                rot = cv2.Rodrigues(rvec.reshape(3))[0]
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = tvec.reshape(3)
                cam_calib = {
                    'time': cam_info['times'][cam_info['lidar_frames'].index(frame)],
                    'extrinsic': [x for xx in T.tolist() for x in xx],
                    'intrinsic': [x for xx in intrinsic for x in xx],
                    'distortion': cam_info['distortion'],
                    'filenames': [os.path.join(f'measurement{exp}', cam_id, f'{frame}.jpg')],
                }
                cs.add_cam_to_fdict(
                    fdict,
                    str(cam_id),
                    cam_id=0,
                    **cam_calib
                )
                cs.remove_lidar_info(fdict, str(cam_id))
            fdict['agents'].pop(0)
            sdict[frame] = fdict

        # split meta if too large
        frames = sorted(list(sdict.keys()))
        for i in range(0, len(frames) // 500 + 1):
            out_dict = {k: sdict[k] for k in frames[i*500:(i+1)*500]}
            if len(out_dict) > 0:
                save_json(out_dict,
                          os.path.join(meta_out_dir, f'measurement{exp}_{i}.json'))


if __name__=="__main__":
    convert_to_cosense3d(
        '/koko/LUMPI/lumpi-official',
        '/koko/LUMPI/lumpi_selected/meta',
        '/koko/LUMPI/lumpi_selected/data',
        parse_img=True
    )

    # update_meta_boxes(
    #     '/koko/LUMPI/cosense_fmt/meta',
    #     '/mars/projects20/logs/nofusion_lumpi/test/jsons'
    # )

    # sustech_path = "/koko/LUMPI/lumpi_selected_sustech"
    # scenarios = sorted(os.listdir(sustech_path))
    # for s in scenarios:
    #     measurement_idx = s.split('_')[0][-1]
    #     if  measurement_idx != '4':
    #         continue
    #     copy_img_to_sustech(
    #         "/koko/LUMPI/lumpi_selected/data",
    #         os.path.join("/koko/LUMPI/lumpi_selected/meta", f"{s}.json"),
    #         os.path.join(sustech_path, s),
    #         measurement_idx)