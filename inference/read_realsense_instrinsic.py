import cv2
import pickle
import sys
import os
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
align = rs.align(rs.stream.color)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.color)  # Fetch stream profile for color stream
intr = profile.as_video_stream_profile().get_intrinsics()
print(intr)

