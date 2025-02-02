import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream depth and color frames
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color with increased timeout
        try:
            frames = pipeline.wait_for_frames(timeout_ms=10000)  # Increased timeout to 10 seconds
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap to depth image for better visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            
            # Save images when 's' key is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                cv2.imwrite('color_image.png', color_image)
                cv2.imwrite('depth_image.png', depth_colormap)
                print("Images saved!")
            
            # Break the loop when 'q' key is pressed
            if key & 0xFF == ord('q'):
                break

        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            # Optional: Handle the error (e.g., retry, log, etc.)
            
finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
