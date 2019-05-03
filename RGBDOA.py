# NumPy/Python Obstacle avoidance planner and odometry with RGBD data :
# Author :  MUNCH Quentin 2018/2019

import numpy as np
import cv2
import imutils
import pyrealsense2 as rs
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Camera intrinsic parameters
fx = 384.996
fy = 384.996
cx = 325.85
cy = 237.646

# width: 640, height: 480, ppx: 325.85, ppy: 237.646, fx: 384.996, fy: 384.996, model: Brown Conrady, coeffs: [0, 0, 0, 0, 0]
MIP = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2**3)
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()

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

    # Tell pointcloud object to map to this color frame
    pc.map_to(frame_aligned)

    # Generate the pointcloud and texture mappings
    points = pc.calculate(depth_aligned)
    # return all vertex point without texture
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    # filter point between (Ymin, Ymax) and the min distance
    Ymin = 0.28
    Ymax = 2
    Zmin = 0.30
    filter_index = []
    for i in range(vtx.shape[0]):
        if vtx[i,1] < Ymin or vtx[i,1] > Ymax or vtx[i,2] < Zmin:
            filter_index.append(i)
    vtx = np.delete(vtx, filter_index, 0)

    # Convert the cartesian point cloud to polar space
    PSC = np.zeros((vtx.shape[0],2))
    for j in range(vtx.shape[0]):
        r = np.sqrt(vtx[j,0]**2 + vtx[j,2]**2)
        theta = np.arctan2(vtx[j,2], vtx[j,0])
        PSC[j,0] = theta
        PSC[j,1] = r

    # polar histogram + bubble OA 

    if cv2.waitKey(1) == 27:
        break

# Release everyone
cv2.destroyAllWindows()
pipeline.stop()
