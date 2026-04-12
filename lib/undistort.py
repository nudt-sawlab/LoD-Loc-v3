import cv2
import numpy as np
import os
from pathlib import Path
import glob
import shutil
def read_intrinsics(camera):
    image_width_px, image_height_px, fx, fy, cx, cy = camera

    

    K = [[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]]
    
    return np.array(K), image_width_px, image_height_px  
def write_intrinsics(intrinsics_path, NewCameraMatrix, w, h, name_list):
    fx, fy, cx, cy = NewCameraMatrix[0][0], NewCameraMatrix[1][1], NewCameraMatrix[0, 2], NewCameraMatrix[1,2]
    
    with open(intrinsics_path, 'w+') as f:
        for name in name_list:
            outline = name + ' ' + 'PINHOLE' +' '+ str(w) + ' ' + str(h) + ' ' + str(fx) + ' ' + str(fy) + ' ' + str(cx) + ' ' + str(cy) + '\n'
            f.write(outline)
    
def main(image,
         query_camera,
         kp,
         ):
    kp = np.array(kp).astype(np.float32)
    CameraMatrix, w, h = read_intrinsics(query_camera)
    NewCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(CameraMatrix, kp, (w, h), 1,  (w, h), 0)

    img_disort = cv2.undistort(image, CameraMatrix, kp,None, NewCameraMatrix) # None, NewCameraMatrix
    print(NewCameraMatrix)
    update_query_camera = [w, h, NewCameraMatrix[0][0], NewCameraMatrix[0][2], NewCameraMatrix[1][2]]
    return img_disort, update_query_camera

       
if __name__ == "__main__":
    kp = [0.3009, -1.1082232, 0.00050171939, 0.00048351417, 1.118214837] #k1k2p1p2k3
    image_path = "/media/ubuntu/XB/DJI_20231204164034_0001_W_frames_1" 
    w_save_path = "/media/ubuntu/XB/undistort_video_frames"
    intrinsics_path = "/home/ubuntu/Documents/code/github/Render2loc/datasets/demo4/queries/query_intrinsics.txt"
    main(image_path, w_save_path, intrinsics_path, kp)      