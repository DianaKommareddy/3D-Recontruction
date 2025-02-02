import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# Directory to save captured images
rgb_save_dir = "camera_images"
depth_save_dir = "Depth_images"
os.makedirs(rgb_save_dir, exist_ok=True)
os.makedirs(depth_save_dir, exist_ok=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Enable depth stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)
print("RealSense D455 Camera is streaming...")

count = 0
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Display the images
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save the images when 's' key is pressed
            color_filename = os.path.join(rgb_save_dir, f'color{count}.jpg')
            depth_filename = os.path.join(depth_save_dir, f'depth{count}.png')
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)
            print(f"Saved {color_filename} and {depth_filename}")
            count += 1
        elif key == ord('q'):  # Exit the loop when 'q' key is pressed
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    print("RealSense D455 Camera streaming stopped.")
