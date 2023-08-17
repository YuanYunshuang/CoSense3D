import os.path

import numpy as np
import cv2
import torch

from cosense3d.utils.misc import load_json


def load_meta(meta_path):
    meta_dict = {}
    scenarios = [d[:-5] for d in os.listdir(meta_path) if 'json' in d]
    for scenario in scenarios:
        meta_file = os.path.join(meta_path, f"{scenario}.json")
        scenario_dict = load_json(meta_file)
        meta_dict[scenario] = scenario_dict
    return meta_dict


def project_points_by_matrix(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transformation matrix.
    """

    # convert to homogeneous coordinates via padding 1 at the last dimension.
    if isinstance(points, np.ndarray):
        # (N, 4)
        points_homogeneous = np.concatenate(
            [points, np.ones_like(points[:, :1])], axis=1)
    else:
        points_homogeneous = torch.cat(
            [points, torch.ones_like(points[:, :1])], dim=1)
    # (N, 4)
    projected_points = transformation_matrix @ points_homogeneous.T

    return projected_points[:3].T


def save_video_to_imgs(video_filename, out_dir, scale_factors):
    # Open the video file
    cap = cv2.VideoCapture(video_filename)
    print(f"Saving to {out_dir}")
    # Initialize a counter for the image files
    count = 0

    # Loop through the video frames
    while cap.isOpened():
        # if (not max_frame == None) and count >= max_frame:
        #     break
        print(f'Image: {count}', end='\r')
        # Read the frame
        ret, img = cap.read()

        # If the frame was read successfully
        if ret:
            img_file = os.path.join(out_dir, f'{count:06d}.jpg')
            img = cv2.resize(img, None, fx=scale_factors['fx'], fy=scale_factors['fy'])
            # if not os.path.exists(img_file):
            cv2.imwrite(img_file, img)
            count += 1
        else:
            # Break the loop if the frame was not read successfully
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


