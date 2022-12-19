
import sys

import open3d as o3d
import cv2
import numpy as np


color_raw = cv2.imread(sys.argv[1])
disp_raw = cv2.imread(sys.argv[2], cv2.IMREAD_UNCHANGED) # .tiff format
print(color_raw.shape)

#disp_raw = disp_raw.astype(np.float16)
disp_raw = disp_raw.astype(np.float32)

# filter left
if False:
    filter_left = 50
    if filter_left > 0:
        disp_raw[:, :filter_left] = 0
        disp_raw[:, -filter_left:] = 0

# disp to depth
baseline_m = 0.07499
focal_x_length = 396.0926513671875 

depth_raw = focal_x_length * baseline_m / disp_raw
depth_resize = cv2.resize(depth_raw, color_raw.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


depth_o3d = o3d.cpu.pybind.geometry.Image(depth_resize)
color_o3d = o3d.cpu.pybind.geometry.Image(color_raw)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=100.0)

intrinsic = [[396.09265137,   0.,         315.01901245],
             [  0.,         396.09265137, 208.98880005],
             [  0.,           0.,           1.        ]]
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    intrinsic=(o3d.camera.PinholeCameraIntrinsic(color_raw.shape[1], color_raw.shape[0], intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2])))
    #o3d.camera.PinholeCameraIntrinsic(
    #    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# translte to suitable position and view
npcd=np.asarray(pcd.points)
z = npcd[:, 2].mean()
pcd.translate((0, 0, -z))
pcd.rotate(pcd.get_rotation_matrix_from_xyz((-np.pi, 0, 0)))

o3d.io.write_point_cloud(sys.argv[2] + '.pcd', pcd, write_ascii=True)

o3d.visualization.draw_geometries([pcd])
