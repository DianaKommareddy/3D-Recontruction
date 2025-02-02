import numpy as np
import cv2
import open3d as o3d

def generate_point_cloud(rgb_image, depth_image, intrinsics):
    h, w, _ = rgb_image.shape
    intrinsic_matrix = intrinsics.intrinsic_matrix

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image

    # Extract individual intrinsic parameters
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Reshape x and y to match the shape of z
    x = np.tile(x.reshape(-1, 1), (1, w))
    y = np.tile(y.reshape(-1, 1), (1, w))

    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3) / 255.0

    # Remove points with zero depth
    mask = depth_image.reshape(-1) > 0
    xyz = xyz[mask]
    colors = colors[mask]

    # Create Open3D Point Cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


# Load RGB and depth images
rgb_image = cv2.imread('path_to_rgb_image.png')
depth_image = cv2.imread('path_to_depth_image.png', cv2.IMREAD_UNCHANGED)

# Define camera intrinsics (assuming they are known)
intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) # type: ignore

# Generate point cloud
point_cloud = generate_point_cloud(rgb_image, depth_image, intrinsics)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
