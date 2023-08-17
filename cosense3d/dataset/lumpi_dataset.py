import copy
import json
import cv2
import glob
import numpy as np
from plyfile import PlyData
import open3d as o3d
import pandas as pd
from tqdm import tqdm
import os

from cosense3d.utils import pclib


def project_points(points,cap,t,ses,uniqueIds,ids,colors):
    # augment points
    points = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1).T
    tf = compose_tf()
    points = (tf @ points)

    imgs=[]
    for j in range(len(cap)): ##loop over all cameras
        frame = int(t * ses[j]['fps']) # find frame depending of fps
       # print("Frame:" + str(frame) + "from " + str(cap[j].get(cv2.CAP_PROP_FRAME_COUNT)))
        cap[j].set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ret, img = cap[j].read()
        h, w = img.shape[:2]
        scale_factor_y = 640 / h
        scale_factor_x = 960 / w
        intrinsic = np.array(ses[j]['intrinsic'])
        extrinsic = np.array(ses[j]['extrinsic'])
        rvec = np.array((ses[j]['rvec']))
        tvec = np.array(ses[j]['tvec'])
        distortion = np.array(ses[j]['distortion'])
        intrinsic[0, 0] *= scale_factor_x
        intrinsic[1, 1] *= scale_factor_y
        intrinsic[0, 2] *= scale_factor_x
        intrinsic[1, 2] *= scale_factor_y
        img = cv2.resize(img, None, fx=scale_factor_x, fy=scale_factor_y)

        rot = cv2.Rodrigues(rvec.reshape(3))[0]
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = tvec.reshape(3)
        points_tf = T @ np.linalg.inv(tf) @ points
        # points = rot @ points.T + tvec
        imgPoints = (intrinsic @ points_tf[:3]).T
        imgPoints = (imgPoints / imgPoints[:, 2:]).astype(int)

        # imgPoints = \
        # cv2.projectPoints(points, rvec, tvec, intrinsic,
        #                   None)[0][:, 0, :].astype(int)

        ix=np.where((imgPoints[:,0]<img.shape[1]) &(imgPoints[:, 0] >0)) #and imgPoints[:, 1] < img.shape[1] >0)
        iy=np.where((imgPoints[:,1]<img.shape[0]) &(imgPoints[:, 1] >0) )#and imgPoints[:, 1] > 0 )

        for c in range(len(uniqueIds)):#loop over all lidars
            idc = np.where(ids == uniqueIds[c])
            idc=np.intersect1d(idc[0],ix[0])
            idc = np.intersect1d(idc, iy[0])
            tmpIp = imgPoints[idc, :]
            for i in range(tmpIp.shape[0]):
                    cv2.circle(img, (tmpIp[i, 0], tmpIp[i, 1]), 4, colors[c])
        imgs.append(img)
    return imgs


def compose_tf():
    tf = np.eye(4)
    rotate_param = [0, 0, 180]
    flip_param = 'xy'
    scale_param = 0.05

    # param: [roll, pitch, yaw] in degree
    angles = []
    for angle in rotate_param:
        angle = angle / 180 * np.pi * np.random.random()
        angles.append(angle)
    angles = np.array(angles)
    rot = pclib.rotation_matrix(angles)
    tf[:3, :3] = rot @ tf[:3, :3]

    # fn_flip
    rot = np.eye(3)
    flip = np.random.choice(4, 1)
    # flip =1 : flip x
    # flip =2 : flip y
    # flip =3 : flip x & y

    # flip x
    if 'x' in flip_param and (flip == 1 or flip == 3):
        rot[0, 0] *= -1
    # flip y
    if 'y' in flip_param and (flip == 2 or flip == 3):
        rot[1, 1] *= -1
    tf[:3, :3] = rot @ tf[:3, :3]

    # fn_scale(self, tf):
    scale = np.eye(3)
    scale_ratio = np.random.uniform(0.95, 1.05, (1, 3))
    scale[[0, 1, 2], [0, 1, 2]] = scale_ratio
    tf[:3, :3] = scale @ tf[:3, :3]
    return tf


if __name__ == '__main__':

    projectPath="/koko/LUMPI"
    with open(os.path.join(projectPath, "meta.json"), "r") as f:
        meta = json.load(f)
    tmp = os.path.join(projectPath, "train", "measurement" + str(5), "lidar0","000000.ply")
    plydata = PlyData.read(tmp)
    ply = plydata['vertex']
    points = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()

    ##Start reading data
    for exp in [5]:  #meta["experiment"]:
        cap = []
        maxFPS = 0
        ses = []
        skip=0
        print("loading exp: "+str(exp)+"...")
        for s in meta["session"]:
           if meta["session"][s]["experimentId"]!=exp:
               continue
           if meta["session"][s]["type"]=="lidar0":
                continue
           id = meta["session"][s]["deviceId"]
           # path to input videos
           name = os.path.join(projectPath, "train", "measurement" + str(exp),
                             "cam", str(meta["session"][s]["deviceId"]), "video.mp4")
           if not os.path.isfile(name):
                print(name +" don't exist")
                skip=1
                continue
           cap.append(cv2.VideoCapture(name))
           ses.append(meta["session"][s])
           maxFPS=max(maxFPS, ses[-1]['fps'])##find fastest camera
        if skip==1:
            continue
        ##Path to lidar0 data
        lidarPath=os.path.join(projectPath, "train", "measurement"+str(exp),"lidar0")
        pcFiles = glob.glob(os.path.join(lidarPath, "*.ply"))
        pcFiles.sort()
        print(str(len(pcFiles))+" lidar0 files found")
        colors=[[0,0,255],[255,0,0],[0,255,0],[125,0,0],[0,0,125]]
        for i in tqdm(range(len(pcFiles))):
            print(pcFiles[i])
            plydata = PlyData.read(pcFiles[i])
            ply = plydata['vertex']
            points = pd.DataFrame(ply[["x", "y", "z"]]).to_numpy()
            gray=np.ones((points.shape[0],3))*0.5
            ids = ply["id"]
            uniqIds = np.unique(ids)
            times=ply["time"]*pow(10. ,-6)##time of point in micro seconds
            frames=np.floor(maxFPS*times)##generate maximum frame number
            uniqFrames=np.unique(frames)##getting frames per point cloud
            for j in range(len(uniqFrames)):
                f=uniqFrames[j]
                idx = np.where(frames == f)## active points at this frame
                if(len(idx[0])<1000):
                    continue
                tmpPoints=points[idx[0],:]
                tmpIds=ids[idx[0]]
                t=f/maxFPS
                imgs=project_points( tmpPoints,cap, t+1/50.*2, ses, uniqIds, tmpIds,colors)##find image for each camera and draw colored points
                h, w = 640, 960
                img1 = cv2.resize(imgs[0], (int(w / 2), int(h / 2)))
                img2 = cv2.resize(imgs[1], (int(w / 2), int(h / 2)))
                img3 = cv2.resize(imgs[2], (int(w / 2), int(h / 2)))
                globImg = np.zeros((int(h), int(w), 3)).astype(np.uint8)
                globImg[:int(h / 2), :int(w / 2), :] = img1
                globImg[:int(h / 2), int(w / 2):, :] = img2
                globImg[int(h / 2):, :int(w / 2), :] = img3
                cv2.imshow("LUMPI Visualization",globImg)
                cv2.waitKey(2)

