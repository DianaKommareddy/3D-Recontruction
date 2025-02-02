import pyrealsense2 as rs
import numpy as np
import cv2

# Create a RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    # Capture frames for object extraction
    for i in range(10):  # Capture 10 frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Save RGB and depth images
        cv2.imwrite(f'rgb_image_{i}.png', color_image)
        cv2.imwrite(f'depth_image_{i}.png', depth_image)
        print(f'Images {i} captured and saved.')

finally:
    pipeline.stop()
