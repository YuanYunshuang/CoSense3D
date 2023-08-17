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
from cosense3d.dataset.const import CoSenseBenchmarks as csbm
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as csd


class Tracker:
    def __init__(self, log_dir):
        cfgs = load_config(log_dir)
        self.cfgs = cfgs
        seed_everything(1234)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # set paths
        self.log_dir = cfgs['TRAIN']['log_dir']
        # self.logger = open(os.path.join(self.log_dir, 'test', 'tracker.log'), mode='w')

        # load modules
        self.cfgs['DATASET']['batch_size_val'] = 1
        self.cfgs['DATASET']['history_len'] = 1
        self.cfgs['DATASET']['meta_path'] = ''
        # self.cfgs['DATASET']['gt'] = []
        self.test_dataloader = get_dataloader(self.cfgs['DATASET'], mode='val',)
        self.dataset = self.test_dataloader.dataset
        self.post_processor = getattr(self.test_dataloader.dataset, 'post_processes', None)
        self.post_processor.set_log_dir(os.path.join(self.log_dir, 'test'))
        self.model = get_model(cfgs['MODEL'], mode='test').to(self.device)

        # find cls names
        cls_names = []
        for head in cfgs['MODEL']['heads']:
            if 'det_center_sparse' not in head:
                continue
            else:
                for names in head['det_center_sparse']['class_names_each_head']:
                    cls_names.extend(names)
        # get label mapping: centerpoint net format --> consense type id
        self.det_label_to_cosense_typeid = np.array([csd.OBJ_NAME2ID['unknown']] + \
                                           [csd.OBJ_NAME2ID[name] for name in cls_names])
        self.zmin = cfgs['DATASET']['CropLidarRange']['z'][0]


        # load checkpoint
        ckpt = torch.load(os.path.join(self.log_dir, 'last.pth'))
        load_model_dict(self.model, ckpt['model_state_dict'])
        self.model.track_base.tracker = self
        self.model.eval()

        self.state = None
        self.objects = {}
        self.last_assignment = []
        # self.last_pcd = None
        self.last_boxes = None
        self.id_ptr = 0
        self.last_frame_idx = -1
        self.frame_cnt = 0

    def load_dataset(self, meta_dict, scene):
        self.dataset.meta_dict = {scene: meta_dict}
        self.dataset.parse_samples()

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
        if self.last_frame_idx > 0:
            dets = self.state['detections']
            # cur_pcd = batch_dict['pcds'][batch_dict['pcds'][:, 0] == 0, 1:4]

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
                else:
                    new_assignment.append(self.id_ptr)
                    self.objects[self.id_ptr] = [self.get_tracklet(i2)]
                    self.id_ptr += 1

            self.last_assignment = new_assignment
            # self.last_pcd = cur_pcd
            self.last_boxes = dets['box']

    def assign_to_previous(self, batch_dict, index):
        detections = batch_dict['det_s1'][1]
        # use negative ids to indicate new tracklets,
        # finale unique ids will be generated in the interface model
        if index == 0:
            assignment = - np.arange(len(detections['box'])) - 1
        else:
            assignment = []
            gt_objects = batch_dict['objects'][batch_dict['objects'][:, 0] == 0]
            boxes1 = gt_objects[:, [3, 4, 5, 6, 7, 8, 11]]
            ids1 = gt_objects[:, 1]
            assign_idx = self.model.track_base.get_assignments(batch_dict)[0]
            for i1, i2 in zip(*assign_idx):
                tracklet_id = ids1[i1]
                box1 = boxes1[i1]
                box2 = detections['box'][i2]
                dist = torch.norm(box1[:2] - box2[:2])
                if dist < 2 and tracklet_id >=0:
                    assignment.append(tracklet_id.int().item())
                else:
                    self.id_ptr += 1
                    assignment.append(- self.id_ptr)
            detections = {
                'box': detections['box'][assign_idx[1]],
                'scr': detections['scr'][assign_idx[1]],
                'lbl': detections['lbl'][assign_idx[1]]
            }
        return detections, assignment

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

    def inf(self, index):
        with torch.no_grad():
            batch_dict = self.dataset.collate_batch([self.dataset[index]])
            load_tensors_to_gpu(batch_dict)
            # Forward pass
            self.model(batch_dict)
            detections, assignment = self.assign_to_previous(batch_dict, index)

        # to interface format
        box = detections['box'].cpu().numpy()
        lbl = detections['lbl'].cpu().numpy()
        box[:, 2] += self.zmin
        # mask = lbl > 0
        # box = box[mask]
        # lbl = lbl[mask]
        cosense_typeid = self.det_label_to_cosense_typeid[lbl.astype(int)]
        cosense_box = np.concatenate([cosense_typeid[:, None], box[:, :6],
                                      np.zeros_like(box[:, :2]), box[:, 6:]], axis=-1)
        res = {i: b.tolist() for i, b in zip(assignment, cosense_box)}
        return res


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