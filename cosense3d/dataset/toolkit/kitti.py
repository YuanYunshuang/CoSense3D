import glob
import os
import os.path as osp
import pickle
import re

import cv2
import numpy as np
import tqdm
import open3d as o3d

from cosense3d.utils.misc import load_json, save_json
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs

type_kitti2cosense = {
    'Car': 'vehicle.car',
    'Van': 'vehicle.van',
    'Truck': 'vehicle.truck',
    'Bus': 'vehicle.bus',
    'Tram': 'vehicle.tram',
    'Unknown': 'unknown',
    'Cyclist': 'vehicle.cyclist',
    'MotorcyleRider': 'vehicle.motorcycle',
    'Pedestrian': 'human.pedestrian',
    'Scooterrider': 'vehicle.scooter',
    'DontCare': 'unknown',
    'Misc': 'unknown',
    'Person_sitting': 'human.sitting'
}
type_cosense2kitti = {
    v: k for k, v in type_kitti2cosense.items()
}


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


def get_calib_from_file(filepath):
    ''' Read in a kitti calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''

    data2 = {}
    R0 = np.array([[ 0.99992624,  0.00965411, -0.0072371 ],
                  [-0.00968531,  0.99994343, -0.00433077],
                  [ 0.00719491,  0.00440054,  0.99996366]])
    with open(filepath) as f:
        for line in f.readlines():
            if line[:2] == "P2":
                P2 = re.split(" ", line.strip())
                P2 = np.array(P2[-12:], np.float32)

            if line[:2] == "P3":
                P3 = re.split(" ", line.strip())
                P3 = np.array(P3[-12:], np.float32)

            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                vtc_mat = re.split(" ", line.strip())
                vtc_mat = np.array(vtc_mat[-12:], np.float32)

            if line[:7] == "R0_rect" or line[:6] == "R_rect":
                R0 = re.split(" ", line.strip())
                R0 = np.array(R0[-9:], np.float32)

    data2["P2"]=P2.reshape(3, 4)
    data2["P3"]=P3.reshape(3, 4)
    data2["Tr_velo2cam"]=vtc_mat.reshape(3, 4)
    data2["R0"]=R0.reshape(3, 3)

    return data2


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cat_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cat_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def get_mat_rect_to_lidar(self):
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        mat = np.dot(R0_ext, V2C_ext).T
        return mat

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cat_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cat_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.ob_id = -1
        if len(label)>15:
            self.ob_id=label[-1]
            self.level = self.get_kitti_tracking_obj_level()
        else:
            self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def get_kitti_tracking_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 1 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 2 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str


class KITTI(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.infos = {
            'train': self.read_info('train'),
            'val': self.read_info('val'),
            'test': self.read_info('test')
        }

    def read_info(self, mode):
        # read info file
        info_file = osp.join(self.root_dir,
                             f'kitti_infos_{mode}.pkl')
        if osp.exists(info_file):
            with open(info_file, 'rb') as f:
                infos = pickle.load(f)
        return infos

    def to_cosense(self, meta_path, save_masked_lidar=False):
        os.makedirs(meta_path, exist_ok=True)
        scenario_dict = {}
        planes_dict = {}
        for split, info in self.infos.items():
            split_dir = 'testing' if split == 'test' else 'training'
            # make dir for masked lidar0
            lidar_out_dir = osp.join(self.root_dir, 'cosense3d', split_dir, 'velodyne')
            os.makedirs(lidar_out_dir, exist_ok=True)
            print(split)
            scenarios = []

            for i, info_dict in tqdm.tqdm(enumerate(info)):
                frame = info_dict['point_cloud']['lidar_idx']
                calib = self.get_calib(osp.join(self.root_dir, split_dir, 'calib', f'{frame}.txt'))

                if split_dir == 'training':
                    plane = self.get_plane(osp.join(self.root_dir, 'planes', f'{frame}.txt'))
                    transformation_matrix = calib.get_mat_rect_to_lidar()
                    plane = self.transform_plane(plane, transformation_matrix)
                    planes_dict[frame] = plane
                    # self.draw_points_plane(pts_fov, plane)

                lidar_filename_out = osp.join(lidar_out_dir, f'{frame}.bin')
                lidar_relative_filename = lidar_filename_out.replace(self.root_dir + '/cosense3d/training', '')[1:]

                # mask point cloud with fov
                if save_masked_lidar:
                    lidar_filename_in = osp.join(split_dir, 'velodyne', f'{frame}.bin')
                    points = self.get_lidar(osp.join(self.root_dir, lidar_filename_in))
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, info_dict['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    pts_fov.tofile(lidar_filename_out)

                if 'annos' in info_dict:
                    bbx_center = self.obj_to_cosense(info_dict['annos'])
                else:
                    bbx_center = None
                fdict = cs.fdict_template()
                # no cooperation needed, take lidar0 as global for each frame
                cs.update_frame_bbx(fdict, bbx_center)
                cs.update_agent(fdict, agent_id=0, agent_type='cav')
                cs.update_agent_lidar(fdict, agent_id=0, lidar_id=0,
                                      lidar_file=lidar_relative_filename)
                scenario_dict[frame] = fdict
                if i > 0 and i % 500 == 0:
                    save_json(scenario_dict, osp.join(meta_path, f'kitti_{split}_{i // 500:d}.json'))
                    scenarios.append( f'kitti_{split}_{i // 500:d}')
                    scenario_dict = {}

            if len(scenario_dict) > 0:
                index = int(np.ceil(len(info)/ 500.))
                save_json(scenario_dict, osp.join(meta_path, f'kitti_{split}_{index:d}.json'))
                scenarios.append(f'kitti_{split}_{index:d}')
                with open(os.path.join(meta_path, f'{split}.txt'), 'w') as fh:
                    fh.write('\n'.join(scenarios))

        with open(osp.join(meta_path, 'planes_train_val.pkl'), 'wb') as f:
            pickle.dump(planes_dict, f)

    @staticmethod
    def obj_to_cosense(annos):
        ids = annos['index']
        mask = ids != -1
        names = [type_kitti2cosense[name] for name in annos['name']]
        classes = np.array([cs.OBJ_NAME2ID[n] for n in names])
        bbxs = annos['gt_boxes_lidar']
        bbx_center = np.zeros((len(bbxs), 11))
        bbx_center[:, 0] = ids[mask]
        bbx_center[:, 1] = np.array(classes)[mask]
        bbx_center[:, 2:8] = bbxs[:, :6]
        bbx_center[:, -1] = bbxs[:, -1]
        bbx_center = bbx_center.tolist()
        return bbx_center

    def get_lidar(self, filename):
        assert osp.exists(filename)
        return np.fromfile(str(filename), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, filename):
        assert osp.exists(filename)
        return np.array(cv2.imread(filename).shape[:2], dtype=np.int32)

    def get_label(self, filename):
        assert osp.exists(filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        objects = [Object3d(line) for line in lines]
        if len(objects) == 0:
            return [Object3d(
                'DontCare -1 -1 -4.0061 0.0000 198.4733 416.3764 373.0000 1.5332 1.6821 4.2322 -2.7611 1.6843 4.1515 -4.5719')]

        return objects

    def get_calib(self, calib_src):
        """
        Parameters
        ----------
        calib_src: str or dict

        Returns
        -------
        Calibration object
        """
        if isinstance(calib_src, str):
            assert osp.exists(calib_src)
        return Calibration(calib_src)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    @staticmethod
    def get_plane(filename):
        if not osp.exists(filename):
            return None

        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def draw_points_plane(points, plane):
        pc_range = [0, -40, 70, 40]
        # Create a point cloud from input points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])

        # Define plane dimensions and resolution
        steps_x = pc_range[2] - pc_range[0] + 1
        steps_y = pc_range[3] - pc_range[1] + 1
        x = np.linspace(pc_range[0], pc_range[2], steps_x)
        y = np.linspace(pc_range[1], pc_range[3], steps_y)
        x, y = np.meshgrid(x, y)

        # Compute z-coordinates of the plane based on the input equation
        a, b, c, d = plane
        z = (-a * x - b * y - d) / c

        # Create a triangle mesh from the plane grid
        vertices = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        triangles = []
        for i in range(steps_x - 1):
            for j in range(steps_y - 1):
                idx = i * steps_x + j
                triangles.extend([
                    [idx, idx + steps_x, idx + 1],
                    [idx + 1, idx + steps_x, idx + steps_x + 1]
                ])

        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        plane_mesh.compute_vertex_normals()

        # Visualize the point cloud and the plane
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(point_cloud)
        visualizer.add_geometry(plane_mesh)
        visualizer.run()
        visualizer.destroy_window()

    @staticmethod
    def transform_plane(plane, transformation_matrix):
        a, b, c, d = plane

        # Extract the rotation part of the transformation matrix
        rotation_matrix = transformation_matrix[:3, :3]

        # Compute the normal vector of the plane in coordinate system A
        normal_A = np.array([a, b, c])

        # Transform the normal vector using the rotation matrix
        normal_B = rotation_matrix @ normal_A

        # Find a point on the plane in coordinate system A
        if a != 0:
            point_A = np.array([-d / a, 0, 0])
        elif b != 0:
            point_A = np.array([0, -d / b, 0])
        else:
            point_A = np.array([0, 0, -d / c])

        # Transform the point using the full transformation matrix
        point_B = (transformation_matrix @ np.hstack([point_A, 1]))[:3]

        # Compute the new plane equation in coordinate system B
        a_B, b_B, c_B = normal_B
        d_B = -np.dot(normal_B, point_B)

        plane_B = np.array([a_B, b_B, c_B, d_B])

        return plane_B

    @staticmethod
    def reform_gt_samples(db_info_path, meta_path):
        root_dir = osp.dirname(db_info_path)
        with open(db_info_path, 'rb') as f:
            infos = pickle.load(f)
            db_infos = {}
            for cls, objects in infos.items():
                for obj in objects:
                    points = np.fromfile(
                        osp.join(root_dir, obj.pop('path')),
                        dtype = np.float32
                    ).reshape(-1, 4)
                    obj['points'] = points
                    obj.pop('name')
                    obj.pop('num_points_in_gt')
                db_infos[cls] = objects
                with open(osp.join(meta_path, f'{cls}_samples.pkl'), 'wb') as f:
                    pickle.dump(objects, f)


def convert_kitti():
    db_info_path = "/koko/kitti/kitti_dbinfos_train.pkl"
    meta_path = "/koko/cosense3d/kitti"
    kitti = KITTI("/koko/kitti")
    kitti.to_cosense(meta_path)
    kitti.reform_gt_samples(db_info_path, meta_path)


def convert_waymo():
    meta_path = "/media/hdd/yuan/CoSense3D/dataset/metas/waymo"
    kitti = KITTI("/koko/waymo/kitti")
    kitti.to_cosense(meta_path)


def test_result_kitti_to_sustech():
    test_file_path = "/koko/LUMPI/sustech_fmt/measurement0123/label_kitti"
    label_out_path = "/koko/LUMPI/sustech_fmt/measurement0123/label"

    test_files = sorted(glob.glob(os.path.join(test_file_path, '*.txt')))
    cnt = 0
    for f in tqdm.tqdm(test_files):
        with open(f, 'r') as fh:
            lines = fh.readlines()
        objects = []
        for line in lines:
            obj = Object3d(line)
            theta = ((obj.ry + np.pi) % (2 * np.pi)) - np.pi
            theta = - theta - np.pi / 2
            obj = {
                "obj_id": cnt,
                "obj_type": obj.cls_type,
                "psr": {
                    "position": {"x": float(obj.loc[0]),
                                 "y": float(obj.loc[1]),
                                 "z": float(obj.loc[2] + obj.h / 2)},
                    "rotation": {"x": 0, "y": 0, "z": float(theta)},
                    "scale": {"x": obj.l, "y": obj.w, "z": obj.h}
                }
            }
            objects.append(obj)
            cnt += 1
        save_json(objects, os.path.join(label_out_path,
                                        f"{int(os.path.basename(f)[:-4]):06d}.json"))


if __name__=="__main__":
    convert_kitti()

