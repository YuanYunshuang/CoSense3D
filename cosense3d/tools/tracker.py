import argparse
import os, tqdm, time, glob

import numpy as np
import torch
import open3d as o3d

from cosense3d.model import get_model
from cosense3d.dataset import get_dataloader
from cosense3d.utils import misc, metrics
from cosense3d.utils.train_utils import *
from cosense3d.config import load_config
from cosense3d.utils.box_utils import boxes_to_corners_3d
from cosense3d.utils.vislib import update_lineset_vbo


class Tracker:
    def __init__(self, cfgs, args, use_qt=False):
        self.use_qt = use_qt
        seed_everything(1234)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # set paths
        self.log_dir = cfgs['TRAIN']['log_dir']
        self.out_img_dir = os.path.join(self.log_dir, 'test', 'img')
        self.out_inf_dir = os.path.join(self.log_dir, 'test', 'inf')
        misc.ensure_dir(self.out_img_dir)
        misc.ensure_dir(self.out_inf_dir)
        self.logger = open(os.path.join(self.log_dir, 'test', 'tracker.log'), mode='w')

        # load modules
        cfgs['DATASET']['batch_size_val'] = 1
        cfgs['DATASET']['history_len'] = 0
        self.test_dataloader = get_dataloader(cfgs['DATASET'], mode='val',)
        self.post_processor = getattr(self.test_dataloader.dataset, 'post_processes', None)
        self.post_processor.set_log_dir(os.path.join(self.log_dir, 'test'))
        self.model = get_model(cfgs['MODEL']).to(self.device)

        # load checkpoint
        ckpt = torch.load(os.path.join(self.log_dir, 'last.pth'))
        load_model_dict(self.model, ckpt['model_state_dict'])
        self.model.track_base.tracker = self
        self.model.eval()

        self.state = None
        self.objects = {}
        self.last_assignment = []
        self.last_pcd = None
        self.last_boxes = None
        self.id_ptr = 0
        self.frame_cnt = 0

        self.init_viz(2)

    def init_viz(self, n):
        for i in range(1, n + 1):
            viz = o3d.visualization.Visualizer()
            viz.create_window(width=1000, height=1000)
            viz.get_render_option().background_color = [0.05, 0.05, 0.05]
            viz.get_render_option().point_size = 1.0
            viz.get_render_option().show_coordinate_frame = True
            # vis.get_render_option().line_width = 10.0
            setattr(self, f'viz{i}', viz)
            setattr(self, f'vbo_pcd{i}', o3d.geometry.PointCloud())
            setattr(self, f'vbo_box{i}', o3d.geometry.LineSet())
            setattr(self, f'vbo_match{i}', o3d.geometry.LineSet())
            getattr(self, f'viz{i}').get_render_option().line_width = 5.0

    def update_viz(self, pcd, boxes, viz_idx):
        # draw pcd
        vbo_pcd = getattr(self, f'vbo_pcd{viz_idx}')
        pcd_np = pcd.cpu().numpy()
        pcd_np[:, 0] *= -1
        colors = np.ones_like(pcd_np)
        vbo_pcd.points = o3d.utility.Vector3dVector(pcd_np)
        vbo_pcd.colors = o3d.utility.Vector3dVector(colors)

        # draw boxes
        box_corner = boxes_to_corners_3d(boxes).cpu().numpy()
        vbo_box = getattr(self, f'vbo_box{viz_idx}')
        vbo_box = update_lineset_vbo(vbo_box, box_corner, color=[1, 1, 0])

        if self.frame_cnt <= 1:
            getattr(self, f'viz{viz_idx}').add_geometry(vbo_pcd)
            getattr(self, f'viz{viz_idx}').add_geometry(vbo_box)
            getattr(self, f'viz{viz_idx}').add_geometry(getattr(self, f'vbo_match{viz_idx}'))
        else:
            getattr(self, f'viz{viz_idx}').update_geometry(vbo_pcd)
            getattr(self, f'viz{viz_idx}').update_geometry(vbo_box)

        getattr(self, f'viz{viz_idx}').poll_events()
        getattr(self, f'viz{viz_idx}').update_renderer()
        time.sleep(0.1)

    def viz_highlight_match(self, box1, box2):
        for i, box in enumerate([box1, box2]):
            box_corner = boxes_to_corners_3d(box.unsqueeze(0)).cpu().numpy()
            vbo = getattr(self, f'vbo_match{i + 1}')
            update_lineset_vbo(vbo, box_corner, color=[1, 0, 0])
            getattr(self, f'viz{i + 1}').update_geometry(vbo)
            getattr(self, f'viz{i + 1}').poll_events()
            getattr(self, f'viz{i + 1}').update_renderer()

    def update(self, center_features, center_locs, batch_dict):
        self.state = {
            'centers': center_locs,
            'features': center_features,
            'detections': batch_dict['det_s1'][0],
            'name': batch_dict['scenario'][0][0] + '/' + batch_dict['frame'][0][0],
        }
        if self.frame_cnt == 0:
            for i in range(len(center_features)):
                self.objects[self.id_ptr] = [self.get_tracklet(i)]
                self.last_assignment.append(i)
                self.id_ptr += 1
            self.last_pcd = batch_dict['pcds'][batch_dict['pcds'][:, 0]==0, 1:4]
            self.last_boxes = batch_dict['det_s1'][0]['box']

    def assign_to_tracklet(self, batch_dict):
        if self.frame_cnt > 0:
            dets = self.state['detections']
            cur_pcd = batch_dict['pcds'][batch_dict['pcds'][:, 0] == 0, 1:4]
            self.update_viz(self.last_pcd, self.last_boxes, 1)
            self.update_viz(cur_pcd, dets['box'], 2)

            assign_idx = self.model.track_base.get_assignments(batch_dict)[0]
            new_assignment = []
            for i1, i2 in zip(*assign_idx):
                tracklet_id = self.last_assignment[i1]
                box1 = self.objects[tracklet_id][-1]['box']
                box2 = dets['box'][i2]
                dist = torch.norm(box1[:2] - box2[:2])

                if dist < 2 and tracklet_id >= 0:
                    self.objects[tracklet_id].append(self.get_tracklet(i2))
                    new_assignment.append(tracklet_id)
                    self.viz_highlight_match(box1, box2)
                else:
                    new_assignment.append(self.id_ptr)
                    self.objects[self.id_ptr] = [self.get_tracklet(i2)]
                    self.id_ptr += 1

            self.last_assignment = new_assignment
            self.last_pcd = cur_pcd
            self.last_boxes = dets['box']

    def get_tracklet(self, idx):
        return {
                    'name': self.state['name'],
                    'box': self.state['detections']['box'][idx],
                    'scr': self.state['detections']['scr'][idx],
                    'lbl': self.state['detections']['box'][idx],
                }

    def run(self):
        with torch.no_grad():
            for batch_dict in tqdm.tqdm(self.test_dataloader):
                # if batch_idx > 3:
                #     break
                load_tensors_to_gpu(batch_dict)
                # Forward pass
                self.model(batch_dict)
                self.assign_to_tracklet(batch_dict)

                self.frame_cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--vis-func", type=str) # , default="vis_semantic_unc"
    args = parser.parse_args()

    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_config(args)

    tracker = Tracker(cfgs, args)
    tracker.run()