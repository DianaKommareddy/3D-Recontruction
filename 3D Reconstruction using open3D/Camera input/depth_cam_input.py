import cv2
import numpy as np
import pyrealsense2 as rs
import os

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    # Create directories for saving images if they do not exist
    os.makedirs('RGB_Images', exist_ok=True)
    os.makedirs('Depth_Images', exist_ok=True)

    try:
        image_count = 0
        max_images = 10  # Maximum number of images to capture

        while image_count < max_images:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Failed to get frames, retrying...")
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Normalize the depth image for visualization
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_normalized = np.uint8(depth_image_normalized)

            # Apply thresholding to highlight objects
            _, depth_thresh = cv2.threshold(depth_image_normalized, 128, 255, cv2.THRESH_BINARY_INV)

            # Display the RGB and thresholded depth images
            cv2.imshow('RGB Image', color_image)
            cv2.imshow('Depth Image', depth_thresh)

            # Save the RGB and depth images with unique filenames
            rgb_image_path = f'RGB_Images/rgb_image_{image_count + 1}.png'
            depth_image_path = f'Depth_Images/depth_image_thresh_{image_count + 1}.png'

            cv2.imwrite(rgb_image_path, color_image)
            cv2.imwrite(depth_image_path, depth_thresh)

            print(f"Images {image_count + 1} saved successfully.")

            image_count += 1

            # Wait for a key press and break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
