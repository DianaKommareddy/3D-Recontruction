import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

def generate_point_cloud(depth_image, color_image, intrinsics):
    # Ensure depth image is a single-channel image
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]

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

def segment_bottle(point_cloud):
    # Segment plane (e.g., table) from point cloud
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000)
    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    # Perform clustering to find the bottle
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=50, print_progress=True))

    max_label = labels.max()
    bottle_cloud = None
    for i in range(max_label + 1):
        cluster = outlier_cloud.select_by_index(np.where(labels == i)[0])
        if len(cluster.points) > 5000:  # Assumption: bottle will have a significant number of points
            bottle_cloud = cluster
            break

    return bottle_cloud

# Load depth and color images
depth_images = [
    cv2.imread('path/to/depth_image1.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('path/to/depth_image2.png', cv2.IMREAD_UNCHANGED)
    # Add more depth images as needed
]
color_images = [
    cv2.imread('path/to/color_image1.png', cv2.IMREAD_COLOR),
    cv2.imread('path/to/color_image2.png', cv2.IMREAD_COLOR)
    # Add more color images as needed
]

# Load intrinsics
intrinsics = load_intrinsics()

# List of point clouds
point_clouds = []

for depth_image, color_image in zip(depth_images, color_images):
    # Ensure depth image is a single-channel image
    if depth_image is not None and len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]
    pc = generate_point_cloud(depth_image, color_image, intrinsics)
    point_clouds.append(pc)

# Combine point clouds
combined_point_cloud = point_clouds[0]
for pc in point_clouds[1:]:
    combined_point_cloud += pc

# Downsample the combined point cloud
combined_point_cloud = combined_point_cloud.voxel_down_sample(voxel_size=0.005)

# Save the combined point cloud to a file
o3d.io.write_point_cloud('combined_point_cloud.ply', combined_point_cloud)

# Segment the bottle from the combined point cloud
bottle_cloud = segment_bottle(combined_point_cloud)
if bottle_cloud:
    o3d.visualization.draw_geometries([bottle_cloud])
else:
    print("Bottle not found in the point cloud.")
