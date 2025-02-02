import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

def generate_point_cloud(depth_image, color_image, intrinsics):
    """
    Generate a point cloud from a depth image and corresponding color image.
    """
    h, w = depth_image.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    # Create mesh grid for pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image / 1000.0  # Convert depth from mm to meters

    # Compute 3D coordinates
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image.reshape(-1, 3) / 255.0  # Normalize color values

    # Remove points with zero depth
    mask = depth_image.reshape(-1) > 0
    xyz = xyz[mask]
    colors = colors[mask]

    # Create Open3D Point Cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Get camera intrinsics
profile = pipeline.get_active_profile()
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Capture a frame from RealSense camera
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# Stop the pipeline
pipeline.stop()

# Generate point cloud from captured depth and color images
point_cloud = generate_point_cloud(depth_image, color_image, intrinsics)
o3d.visualization.draw_geometries([point_cloud])

# List to store multiple point clouds
point_clouds = []

# Assume we have multiple depth and color images
depth_images = [depth_image1, depth_image2, ...]
color_images = [color_image1, color_image2, ...]

# Generate point clouds for all images
for depth_img, color_img in zip(depth_images, color_images):
    pc = generate_point_cloud(depth_img, color_img, intrinsics)
    point_clouds.append(pc)

# Combine all point clouds into a single point cloud
combined_point_cloud = point_clouds[0]
for pc in point_clouds[1:]:
    combined_point_cloud += pc

# Downsample the combined point cloud for efficiency
combined_point_cloud = combined_point_cloud.voxel_down_sample(voxel_size=0.005)
o3d.visualization.draw_geometries([combined_point_cloud])

def segment_bottle(point_cloud):
    """
    Segment a bottle from the given point cloud by removing the table (plane) and clustering objects.
    """
    # Plane segmentation to remove the table
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000)
    inlier_cloud = point_cloud.select_by_index(inliers)  # Table (Plane)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)  # Objects above the table

    # Cluster remaining points to identify potential objects
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=50, print_progress=True))

    max_label = labels.max()
    bottle_cloud = None

    # Find the largest cluster (assumed to be the bottle)
    for i in range(max_label + 1):
        cluster = outlier_cloud.select_by_index(np.where(labels == i)[0])
        if len(cluster.points) > 5000:  # Assume a bottle has a significant number of points
            bottle_cloud = cluster
            break

    return bottle_cloud

# Segment and visualize the bottle
bottle_cloud = segment_bottle(combined_point_cloud)
if bottle_cloud:
    o3d.visualization.draw_geometries([bottle_cloud])
else:
    print("Bottle not found in the point cloud.")
