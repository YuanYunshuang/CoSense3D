import os, glob
import torch
import numpy as np

from cosense3d.utils.vislib import o3d_draw_pcds_bbxs


class PreidictionVisualiser:
    def __init__(self, data_root, test_path, dataset_name):
        self.data_root = data_root
        self.test_path = test_path
        self.dataset_name = dataset_name

    def visualize_object_detection(self):
        result_files = sorted(glob.glob(os.path.join(
            self.test_path, "inf", "*.pth"
        )))
        assert len(result_files) > 0
        for f in result_files:
            result = torch.load(f)
            frame = result['name'][0]
            lidar = self.get_frame_data(frame)
            pred_boxes = result['pred_boxes'][0].cpu().numpy()
            gt_boxes = result['gt_boxes'][0].cpu().numpy()
            confs = result['confidence']
            o3d_draw_pcds_bbxs([lidar],
                               [pred_boxes, gt_boxes],
                               [(1, 0, 0), (0, 1, 0)])

    def get_frame_data(self, frame):
        if self.dataset_name == 'kitti':
            split = "testing" if "test" in frame else "training"
            lidar_files = [os.path.join(
                self.data_root, split, "velodyne",
                f"{os.path.basename(frame)}.bin"
            )]
        elif self.dataset_name == 'lumpi':
            lidar_files = glob.glob(os.path.join(
                self.data_root, 'measurement4', '*',
                f"{os.path.basename(frame)}.bin"
            ))
        else:
            raise NotImplementedError
        points = []
        for f in lidar_files:
            points.append(
                np.fromfile(f, dtype=np.float32).reshape(-1, 4)
            )
        points = np.concatenate(points, axis=0)
        return points

    def visualize(self, visual_options):
        """

        Parameters
        ----------
        visual_options: list,
            options of the prediction result that needs to be visualized.
        """
        for opt in visual_options:
            visualizer = getattr(self, f"visualize_{opt}", None)
            assert visualizer is not None, f"No function 'visualize_{opt}' is defined to " \
                                           f"visualize the result of {opt}."
            visualizer()


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--dataset_name", type=str)
    args = parser.parse_args()

    pred_visualizer = PreidictionVisualiser(
        data_root=args.data_root,
        test_path=f"{args.log_dir}/test",
        dataset_name=args.dataset_name
    )
    pred_visualizer.visualize([
        "object_detection"
    ])
