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

    # Init local occupancy map (2.5D)
    X, Z, GE = [], [], []

    # 2D middle obstacle
    for u in range(640):
        z_val = depth[240-1, u]
        x_val = (u-cx)*((z_val*depth_scale)/fx)
        Z.append(z_val)
        X.append(x_val)

    # 2D ground elevation
    for v in range(480-1, 240-1):
        e_val = depth[v, 320-1]
        GE.append(e_val)

    plt.scatter(X, Z)
    plt.show()

    X, Z = [], []

    if cv2.waitKey(1) == 27:
        break

# Release everyone
cv2.destroyAllWindows()
pipeline.stop()
