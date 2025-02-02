import pyrealsense2 as rs
import numpy as np
import cv2

# Define the calibration pattern (e.g., checkerboard)
pattern_size = (9, 6)  # Number of inner corners in the calibration pattern
square_size = 0.0254  # Size of each square in meters (1 inch)

# Initialize lists to store calibration images and corresponding object points
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane

# Create a RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    # Capture frames to find the calibration pattern
    for _ in range(30):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_image, pattern_size, None)

        if ret:
            objp = np.zeros((np.prod(pattern_size), 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            objp *= square_size

            obj_points.append(objp)
            img_points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(color_image, pattern_size, corners, ret)
            cv2.imshow('Chessboard Corners', color_image)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, rgb_camera_matrix, rgb_dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_image.shape[::-1], None, None)
    print("RGB Camera Matrix:")
    print(rgb_camera_matrix)

    # Save calibration data
    np.savez('calibration_data.npz', rgb_camera_matrix=rgb_camera_matrix, rgb_dist_coeffs=rgb_dist_coeffs)

finally:
    pipeline.stop()
