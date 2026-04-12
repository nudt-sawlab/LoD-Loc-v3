import cv2
import numpy as np
import os
from osgeo import gdal
import time
from pathlib import Path
import torch
import pyproj
import matplotlib
from lib.read_model import parse_pose_list, parse_intrinsic_list
from typing import Dict
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from .transform import qvec2rotmat
matplotlib.use('TKAgg')


class TargetLocation():
    def __init__(self, config: Dict):
        self.config = config
        self.clicked_points = []
        self.query_camera = self.config["render2loc"]["query_camera"]
        
        self.DSM_path = self.config["ray_casting"]["DSM_path"]
        self.DSM_npy_path = self.config["ray_casting"]["DSM_npy_path"]
        self.num_sample = self.config["ray_casting"]["num_sample"]
        self.area_minZ = self.config["ray_casting"]["area_minZ"]
        
        # open DSM map
        self.DSM_map = gdal.Open(self.DSM_path)
        self.geotransform = self.DSM_map.GetGeoTransform()
        self.area = np.load(self.DSM_npy_path)

    def dms_to_dd(self, d, m, s):
        return d + (m / 60) + (s / 3600)

    def dd_to_dms(self, DD):
        degrees = int(DD)
        minutes = int((DD - degrees) * 60)
        seconds = (DD - degrees - minutes / 60) * 3600
        seconds = round(seconds, 2)
        dms = f"{degrees}\u00B0{minutes}\u2032{seconds}\u2033"
        return dms, degrees, minutes, seconds



    def interpolate_along_line(self, area, x, y, num_points):

        # xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        

        sample_values = map_coordinates(area, [y, x], order=1)
        # sample_values = np.max(sample_values, axis=1).reshape(-1, 1)


        sample_array = sample_values.reshape((num_points,))

        return sample_array

    def pixel_to_world_coordinate(self, K, R, t, u, v):

        p_camera = np.array([[u], [v], [1]])
        p_camera = np.linalg.inv(K).dot(p_camera)


        p_world = R.dot(p_camera) + t

        return p_world

    def get_index_array(self, dataset):

        ds = dataset


        geotransform = ds.GetGeoTransform()
        x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform


        rows, cols = ds.RasterYSize, ds.RasterXSize


        x = np.arange(cols) * x_pixel_size + x_origin
        y = np.arange(rows) * y_pixel_size + y_origin

        # index_array = [x, y]

        # xx, yy = np.meshgrid(x, y)
        # index_array = np.stack([yy.ravel(), xx.ravel()], axis=1)
        return x, y

    def line_equation_3d(self, point1, point2):

        p1 = np.array(point1)
        p2 = np.array(point2)


        direction = p2 - p1


        a, b, c = direction
        d = -(a * p1[0] + b * p1[1] + c * p1[2])

        return [a, b, c, d]


    def line_equation_2d(self, x1, y1, x2, y2):
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return [A, B, C]

    def line_equation(self, A, B, Z):

        x, y, z = A
        x1, y1, z1 = B
        t = np.array([x1 - x, y1 - y, z1 - z])
        ray = lambda k: np.array([x, y, z]) + k * t


        k = (Z - z) / t[2]
        projection = ray(k)[:2]

        return ray, projection

    def intersection(self, ray_eqn, Z):

        k = (Z - ray_eqn(0)[2]) / (ray_eqn(1)[2] - ray_eqn(0)[2])
        

        intersection_point = ray_eqn(k)
        
        return intersection_point

    def geo_coords_to_array_index(self, x, y, geotransform):
        x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform
        col = ((x - x_origin) / x_pixel_size).astype(int)
        row = ((y - y_origin) / y_pixel_size).astype(int)
        
        return row, col

    def sample_points_on_line(self, line_equation, num_sample, x_minmax):

        A, B, C = line_equation[0], line_equation[1], line_equation[2]
        x_min, x_max = x_minmax[0], x_minmax[1]


        x = np.linspace(x_min, x_max, num_sample)
        y = (-A/B)*x - (C/B)

        return x, y

    def find_z(self, ray_eqn, points):

        k = (points[0] - ray_eqn(0)[0]) / (ray_eqn(1)[0] - ray_eqn(0)[0])
        # z_values = (-d-a*x-b*y)/c
        

        z_values = [ray_eqn(k_i)[2] for k_i in k]
        
        return z_values

    def caculate_predictXYZ(self, K, pose, objPixelCoords):
        R = pose[:3, :3]
        t = pose[:3, 3].reshape([3,1])

        target = self.pixel_to_world_coordinate(K,R,t,objPixelCoords[0],objPixelCoords[1])

        ray_eqn, projection_eqn = self.line_equation(t, target, self.area_minZ)
        line2D_abcd = self.line_equation_2d(t[0], t[1], target[0], target[1])
        

        intersection_point = self.intersection(ray_eqn, self.area_minZ)
        x_minmax = [t[0],intersection_point[0]]


        x, y = self.sample_points_on_line(line2D_abcd, self.num_sample, x_minmax)


        row, col = self.geo_coords_to_array_index(x, y, self.geotransform)
        sampleHeight = self.interpolate_along_line(self.area, col ,row, self.num_sample)


        z_values = self.find_z(ray_eqn,[x, y])
        z_values = torch.tensor(z_values)
        z_values = z_values.squeeze()


        sampleHeight = torch.tensor(sampleHeight)
        abs_x = torch.abs(z_values - sampleHeight)
        min_val, min_idx = torch.min(abs_x, dim=0)
        # print(min_val)

        # print ("Resulting time cost: ",time.time()-start_time_2,"s") 
        
        return [x[min_idx], y[min_idx], z_values[min_idx]]
    def get_intrinsic(self, camera):
            image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera

            focal_ratio_x = f_mm / sensor_width_mm
            focal_ratio_y = f_mm / sensor_height_mm
            

            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y
            cx = image_width_px / 2
            cy = image_height_px / 2
            

            K = [[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]]
            
            return K
    def get_query_intrinsic(self, camera):
            image_width_px, image_height_px, fx, fy, cx, cy = camera

            

            K = [[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]]
            
            return K 
    def get_pose(self, ret):
        q, t = ret['qvec'], ret['tvec']
        # Convert the quaternion to a rotation matrix
        R = np.asmatrix(qvec2rotmat(q)).transpose()
        
        # Initialize a 4x4 identity matrix
        T = np.identity(4)
        
        # Set the rotation and translation components
        T[0:3, 0:3] = R
        T[0:3, 3] = -R.dot(t)
        
        return T

    def estimate_target_location(self, ret, query_image, clicked_points):

        K_w2c = self.get_query_intrinsic(self.query_camera)
        T_c2w = self.get_pose(ret)
        
        predict_xyz = self.caculate_predictXYZ(K_w2c, T_c2w, clicked_points)

        wgs84 = pyproj.CRS('EPSG:4326')
        cgcs2000 = pyproj.CRS('EPSG:4547')
        
        transformer = pyproj.Transformer.from_crs(cgcs2000, wgs84,always_xy=True)
        lon, lat = transformer.transform(predict_xyz[0], predict_xyz[1])
        dms_longitude, d_lon, m_lon, s_lon = self.dd_to_dms(lon[0])
        dms_latitude, d_lat, m_lat, s_lat = self.dd_to_dms(lat[0])
        coord = [dms_longitude, dms_latitude, predict_xyz[2].item()]
        
        # cv2.circle(query_image, (clicked_points[0], clicked_points[1]), 30, (0, 0, 255), -1)
        # cv2.putText(query_image, f"({d_lon}' {m_lon}' {s_lon}'', {d_lat}' {m_lat}' {s_lat}'')", \
        #     (clicked_points[0] + 60, clicked_points[1] + 60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (0, 0, 255), 10)
        # image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        # plt.figure(figsize=(15, 12))
        # plt.imshow(image_rgb)
        # plt.title("Image with object coordinate")
        # print('object corrdinate:',  coord)
        
        # plt.axis('off')
        # plt.show()
        return [predict_xyz[0].item(), predict_xyz[1].item(), predict_xyz[2].item()]

if __name__ == "__main__":
    
    config = {
    "render2loc": {
        "datasets": "/home/ubuntu/Documents/code/Render2loc/datasets/demo8",
        "image_path":"images/images_upright/query",
        "query_camera": "queries/query_intrinsics.txt",
        "query_pose": "results/1_estimated_pose.txt",
        "ray_casting": {
            "object_name": "pedestrian1",
            "num_sample": 100,
            "DSM_path": "/home/ubuntu/Documents/code/Render2loc/datasets/demo7/texture_model/dsm/DSM_merge.tif",
            "DSM_npy_path": "/mnt/sda/feicuiwan/DSM_array.npy",
            "area_minZ": 20.580223,
            "write_path": "./predictXYZ.txt"
    }
    }
    }
    # main(config)