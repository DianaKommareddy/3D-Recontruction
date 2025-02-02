import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

def generate_point_cloud(depth_image, color_image, intrinsics):
    # Create point cloud from depth image and color image
    h, w = depth_image.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image / 1000.0  # Convert from mm to meters

    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image.reshape(-1, 3) / 255.0

    # Remove points with zero depth
    mask = depth_image.reshape(-1) > 0
    xyz = xyz[mask]
    colors = colors[mask]

    # Create Open3D Point Cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

# Function to load intrinsics from a saved profile or manually
def load_intrinsics():
    intrinsics = rs.intrinsics()
    intrinsics.width = 640
    intrinsics.height = 480
    intrinsics.ppx = 320.0  # Principal point x
    intrinsics.ppy = 240.0  # Principal point y
    intrinsics.fx = 600.0   # Focal length x
    intrinsics.fy = 600.0   # Focal length y
    intrinsics.model = rs.distortion.brown_conrady
    intrinsics.coeffs = [0, 0, 0, 0, 0]  # Distortion coefficients
    return intrinsics

# Load depth and color images
depth_image = cv2.imread('path/to/depth_image.png', cv2.IMREAD_UNCHANGED)
color_image = cv2.imread('path/to/color_image.png', cv2.IMREAD_COLOR)

# Load intrinsics
intrinsics = load_intrinsics()

# Generate point cloud from the loaded images
point_cloud = generate_point_cloud(depth_image, color_image, intrinsics)
o3d.visualization.draw_geometries([point_cloud])
