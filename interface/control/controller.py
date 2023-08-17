import copy
import json
import logging
import math
import os.path
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QFileDialog

from interface.model.scene_manager import Scene
from interface.config import LABEL, INTERFACE
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as csd


# decorator
def _reload_pcds(view_ids):
    def decorate(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            func(*args)
            # self.reload_pcd_to_glwgt(view_ids)
        return wrapper
    return decorate


class Controller:
    MOVEMENT_THRESHOLD = 0.1

    def __init__(self, sensor_cfg=None) -> None:
        """Initializes all controllers and managers."""
        self.scene = Scene(sensor_cfg)
        self.view = None

        # Control states
        self.curr_cursor_pos = None  # updated by mouse movement
        self.last_cursor_pos = None  # updated by mouse click

        # Scene states
        self.curr_frame_id = 0
        self.curr_pcd2_id = 0
        self.curr_view_id = None

        # keyevent state
        self.ctrl_pressed = False
        self.edit_mode = False
        self.moving_pcd = False
        self.moving_pick_point = False

        # drawing state
        self.curr_box_type = 0

    def init_controller(self):
        # load default dataset
        self.scene.load_dataset_by_meta(Path("/koko/LUMPI/cosense/meta"))
        if len(self.scene.scenes) > 0:
            self.view.update_toolbar_info('scene', self.scene.scenes)
        logging.info("Default point cloud folder: ./data")

    def set_trackers(self, tracker, singleton_tracker):
        self.tracker = tracker
        self.singleton_tracker = singleton_tracker

    def track(self):
        res = self.tracker.inf(self.scene.frame_idx)
        self.scene.update_frame_labels(res)
        logging.debug(f'Detected {len(res)} labels in current frame.')
        self.view.glWidget0.updateFrameData(
            self.scene.pcd[:, :3],
            self.scene.label,
            self.scene.label_predecessor
        )

    def track_singleton(self, center):
        center = np.array(center)
        pcds = []
        boxes = []
        for i in range(self.singleton_tracker.history_len + 1):
            frame_idx = max(self.scene.frame_idx - i, 0)
            pcd = self.scene.load_pcd(frame_idx)
            dist = np.linalg.norm(pcd[:, :2] - center[:2].reshape(1, 2), axis=1)
            pcd = pcd[dist < 6]
            pcds.append(pcd)
            if i == 0:
                box = [-1, -1] + center.tolist() + [0, ] * 6
                boxes.append(box)
            else:
                cur_boxes = self.scene.load_label(frame_idx)
                # find closest match
                min_dist = 10000
                matched_id = -1
                for id, box in cur_boxes.items():
                    d_sqr = (box[1] - center[0])**2 + (box[2] - center[1])**2
                    if d_sqr < min_dist:
                        min_dist = d_sqr
                        matched_id = id
                boxes.append([matched_id] + cur_boxes[matched_id])
            # import matplotlib.pyplot as plt
            # plt.plot(pcd[:, 0], pcd[:, 1])
            # plt.show()
            # plt.close()
        res = self.singleton_tracker.inf(pcds, boxes, center.squeeze())
        self.scene.add_label(res)
        logging.debug(f'Detected label {list(res.keys())[0]} in current frame.')
        self.view.glWidget0.updateFrameData(
            self.scene.pcd[:, :3],
            self.scene.label,
            self.scene.label_predecessor
        )

    def change_scene(self, index):
        if index > 0:
            self.scene.set_scene(index)
            frames = [pcd.split('.')[0] for pcd in self.scene.frames]
            self.view.update_toolbar_info('frame', frames)
            self.view.updateScenario(self.scene.frame_viewer_info)
            # meta_dict = {f: self.scene.meta_dict[f] for f in frames}
            # self.tracker.load_dataset(meta_dict, self.scene.scene)

    def change_frame(self, index):
        """
        Args:
            index: index of combobox = self.frame_idx + 1
        """
        if index > 0 and index < len(self.scene.frames) \
                and (index - 1) != self.scene.frame_idx:
            self.scene.set_frame(index)
            self.view.glWidget0.updateFrameData(
                self.scene.pcd,
                self.scene.local_label,
                self.scene.label,
                self.scene.label_predecessor
            )
            self.view.updateFrame(index, self.scene.frame_viewer_info)

    def change_type(self, index):
        self.curr_box_type = index
        self.view.updateType()

    def next_frame(self):
        index = self.scene.frame_idx + 1 + 1
        if index > len(self.scene.frames):
            index = 1
        self.change_frame(index)

    def last_frame(self):
        index = self.scene.frame_idx + 1 - 1
        if index < 1:
            index = len(self.scene.frames)
        self.change_frame(index)

    def change_object(self):
        pass

    def open_dataset(self) -> None:
        path_to_folder = Path(
            QFileDialog.getExistingDirectory(
                self.view,
                "Open data folder",
                "/koko/LUMPI/cosense3d", # /media/hdd/yuan/koko/data/comap/
                QFileDialog.DontUseNativeDialog,
            )
        )
        if not path_to_folder.is_dir():
            logging.warning("Please specify a valid folder path.")
        else:
            self.scene.load_dataset(path_to_folder)
            if len(self.scene.scenes) > 0:
                self.view.update_toolbar_info('scene', self.scene.scenes)
            logging.info("Changed point cloud folder to %s!" % path_to_folder)

    def open_dataset_by_meta(self) -> None:
        meta_folder = Path(
            QFileDialog.getExistingDirectory(
                self.view.menu.win,
                "Open data folder",
                "/koko/LUMPI/cosense_fmt/meta", # /media/hdd/yuan/koko/data/comap/
                QFileDialog.DontUseNativeDialog,
            )
        )
        path_to_folder = meta_folder.parent
        if not meta_folder.is_dir():
            logging.warning("Please specify a valid folder path.")
        else:
            self.scene.load_dataset_by_meta(meta_folder)
            if len(self.scene.scenes) > 0:
                self.view.update_toolbar_info('scene', self.scene.scenes)
            logging.info("Changed point cloud folder to %s!" % path_to_folder)

    def save_frame_labels(self, labels, frame_idx=None):
        self.scene.update_frame_labels(labels, frame_idx)
        frame_idx = self.scene.frame_idx if frame_idx is None else frame_idx
        # self.scene.save_labels()

    def get_batch_plot_data(self):
        # find active box in GLWiewer
        activate_item = self.view.glWidget0.activate_item
        if activate_item is None:
            self.scene.edit_box_id = None
            return None
        else:
            # crop all pcds in the batch data by the box
            batch_view_data = self.scene.batch_data_for_box(activate_item.id)
            self.scene.edit_box_id = activate_item.id
            return batch_view_data

    def saveBatchViewLabels(self, data_dict):
        for frame, label in data_dict.items():
            self.save_frame_labels(label, frame)

    def show_obj_info(self, obj):
        info_dict = {
            'Id': obj.id,
            'Type': obj.typeid
        }
        self.view.object_viewer.updateInfo(info_dict)
        self.view.object_viewer.show()

    def hide_obj_info(self):
        self.view.object_viewer.hide()

    def change_active_object_type(self, index):
        """
        Args:
            index: object type id
        """
        item = self.view.glWidget0.activate_item
        item.typeid = index
        self.scene.update_type(item.id, index)
        self.show_obj_info(item)

    def save(self):
        self.scene.save_labels()

    def frame_viewer_checkbox_changed(self, state, text):
        if state == 2:
            self.view.glWidget0.change_visibility(text, True)
            logging.debug(f"change lidar state: {text} --> show")
        else:
            self.view.glWidget0.change_visibility(text, False)
            logging.debug(f"change lidar state: {text} --> hide")











