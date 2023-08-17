import open3d as o3d
import cv2
import os
import numpy as np
import glob
import matplotlib as mlp

cmap = mlp.colors.Colormap('hot', N=256)


def pick_pcd_chessboard_corners(pcd):
    print("")
    print("1) Please pick point using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def calibrate_camera(img_path):
    # termination criteria: type, max cnt, eps
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)...
    objp = np.zeros((7 * 10, 3), np.float32)
    objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(os.path.join(img_path, '*.jpg'))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (21, 21), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 10), corners2, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints,
        gray.shape[::-1],  # image size
        None,  # camera matrix
        None  # distortion coefficients
    )

    # Undistort images
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        cameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, cameraMatrix)
        # crop image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        cv2.imwrite(fname.replace('sync', 'calib_img'), dst)

if __name__=="__main__":
    calibrate_camera("/media/cav/Expansion/data/V2Ireal/calibration/2023-02-21-18-53-10/sync")
    # file = "/media/cav/Expansion/data/V2Ireal/calibration/2023-02-21-18-53-10/sync/000004.txt"
    # points = np.loadtxt(file, skiprows=10, delimiter=' ').astype(float)
    # colors = mlp.colormaps['jet'](points[:, -1])
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # pick_idx = pick_pcd_chessboard_corners(pcd)
    # picked_idx = [62065, 60068, 63711, 62420, 63451, 64331, 64878, 66406,
    #               69369, 67955, 70423, 69386, 68256, 64824, 64347, 63393,
    #               64548, 61730, 62902, 59643, 59177, 60180, 63672, 62449,
    #               65871, 66856, 65334, 66297, 67424, 67916, 70455, 67300,
    #               68284, 67295, 63815, 63142, 64855, 61607, 60755, 59684]