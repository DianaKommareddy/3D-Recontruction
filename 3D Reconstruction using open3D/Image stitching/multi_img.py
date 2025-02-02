import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def save_images(color_image, depth_image):
    os.makedirs('Stitch_Img_RGB', exist_ok=True)
    os.makedirs('Stitched_Depth', exist_ok=True)
    cv2.imwrite('Stitch_Img_RGB/color_image_Part_2.png', color_image)
    cv2.imwrite('Stitched_Depth/depth_image_Part_2.png', depth_image)

def main():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of depth and color frames
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()

    # Ensure the camera is ready
    if depth_sensor.supports(rs.option.laser_power):
        depth_sensor.set_option(rs.option.laser_power, 100)

    try:
        retry_count = 5
        while retry_count > 0:
            try:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames(timeout_ms=10000)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    print("Could not get frames from the camera")
                    retry_count -= 1
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Stack both images horizontally
                images = np.hstack((color_image, depth_colormap))

                # Show images
                cv2.imshow('RealSense', images)
                
                # Save images
                save_images(color_image, depth_colormap)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                retry_count -= 1
                time.sleep(1)
                continue

            retry_count = 5

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
