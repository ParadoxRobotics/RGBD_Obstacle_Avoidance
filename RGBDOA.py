# NumPy/Python Obstacle avoidance planner and odometry with RGBD data :
# Author :  MUNCH Quentin 2018/2019

import numpy as np
import cv2
import imutils
import pyrealsense2 as rs
from open3d import *
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Camera intrinsic parameters
fx = 641.66
fy = 641.66
cx = 324.87
cy = 237.38

MIP = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

# Init the D435 pipeline capture
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
# start D435 and depth scale
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# alignement object
align_to = rs.stream.color
align = rs.align(align_to)

while True:
    # Acquire state
    state = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    aligned_state = align.process(state)
    frame_aligned = aligned_state.get_color_frame()
    depth_aligned = aligned_state.get_depth_frame()
    # get reference RGB state
    frame = np.asanyarray(frame_aligned.get_data())
    # get reference Depth state
    depth = np.asanyarray(depth_aligned.get_data())
    print(depth.shape)

    # Init point cloud in cartesian (2D)
    pcc = np.zeros((640,2))
    # Init point cloud in polar value
    pcp = np.zeros((640,2))

    # 2D middle obstacle
    for u in range(640):
        z_val = depth[240-1, u]
        x_val = (u-cx)*((z_val*depth_scale)/fx)
        pcc[u, 1] = z_val
        pcc[u, 0] = x_val

    for v in range(640):
        r_val = np.sqrt(pcc[u,0]**2 + pcc[u,1])
        theta_val = np.degrees(np.arctan2(pcc[u,1], pcc[u,0]))
        pcp[u,1] = r_val
        pcp[u,0] = theta_val


    if cv2.waitKey(1) == 27:
        break

# Release everyone
cv2.destroyAllWindows()
pipeline.stop()
