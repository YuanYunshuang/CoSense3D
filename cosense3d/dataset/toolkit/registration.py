"""
Author : Yunshuang Yuan
------------------------
This script registers the source point cloud to the target point cloud with two steps:
1. Visualize the original point clouds
2. Picking points (>=3) from both point clouds for registration
3. Calculate initial transformation based on the picked point pairs and refine the transformation with ICP.
"""

import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def read_bin2o3d(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


def vis_geometry(pcd):
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    o3d.visualization.draw_geometries_with_editing([pcd])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def manual_registration(source, target, trans_init=None):
    if trans_init is None:
        print("Visualization of two point clouds before manual alignment")
        draw_registration_result(source, target, np.identity(4))

        # pick points from two point clouds and builds correspondences
        picked_id_source = pick_points(source)
        picked_id_target = pick_points(target)
        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target

        # estimate rough transformation using correspondences
        print("Compute a rough transform using the correspondences given by user")
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target,
                                                o3d.utility.Vector2iVector(corr))
    print('Initial transformation after point picking: \n', trans_init)

    print("Perform point-to-point ICP refinement")
    # raw alignment
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, 0.6, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # refine alignment
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, 0.1, reg_p2p.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
    return reg_p2p.transformation


if __name__ == "__main__":
    # trans_init = np.array(
    #     [[ 0.08338375, -0.99555312, -0.04383074, 14.96047737],
    #      [ 0.99612891,  0.08204236,  0.0315633 , -4.06152382],
    #      [-0.02782696, -0.04629294,  0.99854025,  0.7442108 ],
    #      [ 0.        ,  0.        ,  0.        ,  1.        ],]
    # )
    manual_registration(
        src_file="/media/cav/Expansion/data/V2Ireal/feb13/rosbags/static/processing/lidar/1676326367_1676326389_infra/000000.bin",
        tgt_file="/media/cav/Expansion/data/V2Ireal/feb13/rosbags/static/processing/lidar/1676326367_1676326389_car/000000.bin",
        # trans_init=trans_init
    )